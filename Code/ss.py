"""
Routines specific for steady state
"""
import pandas as pd
import numpy as np
from scipy import optimize
import copy

from Code.household import Household
from Code.production import Production
from Code.government import Government
from Code.demographics import Population, load_poptrans
from Code.equilibrium import walras_law, calculate_NFAY_and_Y
from Code.calibration import load_initial_ss


def calculate_ss(hh: Household, popss: Population, govss: Government, prodss: Production, r, gamma, Beq_j_xi=None):
    """Obtain country-level steady state given parameters"""
    solve_flag = Beq_j_xi is None
    prod = prodss.adjust_r(r)
    gov = govss.adjust(popss, prod, hh, r, gamma)

    # now solve for bequests that are self-consistent, if they are not exogenously supplied
    if Beq_j_xi is None:
        beq_error = lambda Beq_xi: hh.ss_givenbequests(hh.bequest_rule(Beq_xi, popss.pij), popss, gov, prod.w_Z, r, gamma)['Beq_xi_implied'] - Beq_xi
        Beq_xi = optimize.fsolve(beq_error, np.zeros(hh.N_xi))
        Beq_j_xi = hh.bequest_rule(Beq_xi, popss.pij)
    ss = hh.ss_givenbequests(Beq_j_xi, popss, gov, prod.w_Z, r, gamma)

    # calculate implied NFA and check Walras' law (if bequests not exogenously supplied)
    if solve_flag:
        NFA_Y, _ = calculate_NFAY_and_Y(ss, hh, popss, gov, prod)
        err = walras_law(hh, ss, popss, gov, prod, NFA_Y, r, gamma)
        assert np.abs(err) < 1E-3, f'Walras law not satisfied: {err}'
    return ss, gov, prod


def solve_world_ss(hh, popss, govss, prodss, gamma, rmin=0.01, rmax=0.06, Beq_j_xi=None):
    def error_NFA(r):
        NFA_agg = 0
        for c in hh:
            Beq_j_xi_c = Beq_j_xi[c] if Beq_j_xi is not None else None
            ss, gov, prod = calculate_ss(hh[c], popss[c], govss[c], prodss[c], r, gamma, Beq_j_xi_c)
            NFA_Y, Y = calculate_NFAY_and_Y(ss, hh[c], popss[c], gov, prod)
            NFA_agg += NFA_Y * Y
        print(f'TRYING r={r}, \t NFA={NFA_agg}')
        return NFA_agg

    print('Solving for world steady state interest rate...')
    return optimize.brentq(error_NFA, rmin, rmax)


def calculate_terminal_world_ss_by_exercise(h_mult=None, fiscal_rule='all', fixed_bequests=False,
        fixed_mortality=False, fixed_retirement=False, norisk=False, soe=False, r_cached=None, validate_r=False,
        calibration_options=None, closed_economy=None, migration=False, fixed_debt=True, alt_debt=False, alt_ret=False, vintage_UNPP=False, fertility_scenario="medium", LIC_wedge=0):
    
    if calibration_options is None:
        calibration_options = {'case': 'homothetic'} if norisk else {}
        
    r_init, gamma, hh, prod_init, pop_init, gov_init, ss_init = load_initial_ss(**calibration_options, LIC_wedge=LIC_wedge)
    
    if h_mult is not None:
        hh = {c: hh[c].update_h(h_mult[c][-1]) for c in hh}

    # if need to compute the case of a closed economy
    if closed_economy is not None:
      hh = {k: v for k, v in hh.items() if k in [closed_economy]}
      prod_init = {k: v for k, v in prod_init.items() if k in [closed_economy]}
      pop_init = {k: v for k, v in pop_init.items() if k in [closed_economy]}
      gov_init = {k: v for k, v in gov_init.items() if k in [closed_economy]}
      ss_init = {k: v for k, v in ss_init.items() if k in [closed_economy]}

    cs = list(hh)

    # create the specific terminal population and government objects depending on exercise
    pop = load_poptrans(fixed_mortality=fixed_mortality,migration=migration,vintage_UNPP=vintage_UNPP,fertility_scenario=fertility_scenario)
    pop_term = {c: pop[c].get_ss_terminal() for c in cs}
    
    # import alternative retirement age path
    ret_path = pd.read_excel("@Import/Data/intermediate_data/_Tr_Scenario1.xlsx")[['isocode','year','Tr_1']]
    year_max, year_min = ret_path['year'].max(), ret_path['year'].min()
    ret_max = ret_path.query('year == @year_max').set_index('isocode')['Tr_1']
    ret_min = ret_path.query('year == @year_min').set_index('isocode')['Tr_1']
    ret_increase = ret_max - ret_min

    gov_term = copy.deepcopy(gov_init)
    for c in gov_term:
        gov_term[c].adjust_rule = fiscal_rule
        if not fixed_retirement:
            years_adj = (5 if not alt_ret and c not in ('LIC', 'IND') 
                        else 0 if not alt_ret and c in ('LIC', 'IND') 
                        else ret_increase[c])
            gov_term[c].rho = gov_term[c].adjust_rho(gov_term[c].rho, years_adj)
            
    if not fixed_debt:
        df = pd.read_csv("@Import/Data/intermediate_data/debt_gdp_w_LIC.csv")
        year_max, year_min = df['year'].max(), df['year'].min()
        B_Y_end = df.query('year == @year_max').set_index('isocode')['debt_gdp']
        B_Y_2016_19 = df.query('@year_min <= year <= @year_min+3').groupby('isocode')['debt_gdp'].mean()
        for c in gov_term:
            if not alt_debt or B_Y_end[c] < B_Y_2016_19[c]:
                gov_term[c] = gov_term[c].update_B_Y(B_Y_end[c])
            else:
                gov_term[c] = gov_term[c].update_B_Y(B_Y_2016_19[c])
    
    # if fixed bequests, will need to supply initial Beq_j_xi exogenously for each country; otherwise indicate 'None'
    Beq_j_xi = {c: hh[c].bequest_rule(ss_init[c]['Beq_xi_implied'], pop_init[c].pij) for c in cs} if fixed_bequests else None

    if not soe:
        if r_cached is not None:
            if not validate_r:
                r_term = r_cached
            else:
                r_term = solve_world_ss(hh, pop_term, gov_term, prod_init, gamma, rmin=-0.01, rmax=r_init+0.01, Beq_j_xi=Beq_j_xi)
                assert np.isclose(r_term, r_cached), 'Cached value incorrect!'
        if r_cached is None:
            r_term = solve_world_ss(hh, pop_term, gov_term, prod_init, gamma, rmin=-0.01, rmax=r_init+0.01, Beq_j_xi=Beq_j_xi)
    else:
        r_term = r_init

    ss_term, prod_term = {}, {}
    for c in cs:
        Beq_j_xi_c = Beq_j_xi[c] if fixed_bequests else None
        ss_term[c], gov_term[c], prod_term[c] = calculate_ss(hh[c], pop_term[c], gov_term[c], prod_init[c], r_term, gamma, Beq_j_xi_c)
    
    return r_term, gamma, hh, prod_term, pop_term, gov_term, ss_term


"""Calculate asset demand and supply semielasticities"""

def calculate_eps_d(hh: Household, popss: Population, govss: Government, prodss: Production, r, gamma, Beq_j_xi=None):
    """Calculate semielasticity of W/Y with respect to r for individual country, and total wealth"""
    h = 1E-5
    r_up, r_dn = r + h, r - h
    ss_up, _, prod_up = calculate_ss(hh, popss, govss, prodss, r_up, gamma, Beq_j_xi)
    ss_dn, _, prod_dn = calculate_ss(hh, popss, govss, prodss, r_dn, gamma, Beq_j_xi)
    dlogW_dr = (np.log(ss_up['W']) - np.log(ss_dn['W'])) / (2 * h)
    dlogY_dr = (np.log(prod_up.Y_ZL) - np.log(prod_dn.Y_ZL)) / (2 * h)

    return dlogW_dr - dlogY_dr


def calculate_eps_s(W_Y, prodss: Production, r):
    return 1 / (r + prodss.delta) * (prodss.K_Y / W_Y)