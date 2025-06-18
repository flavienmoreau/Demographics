"""
Routines specific for transitional dynamics.
"""
import numpy as np
import os

import multiprocessing as mp
from functools import partial


from Code.production import Production
from Code.government import Government
from Code.demographics import Population, load_poptrans
from Code.household import Household
from Code.calibration import load_initial_ss
from Code import ss, jacobians as jac


def solve_world_td_soe(pop, gov, ss_term, cs=None):
    # load initial steady state
    r, gamma, hh, prod_ss, _, _, ss_init = ss.load_initial_ss()
    if cs is None:
        cs = list(hh)


def solve_world_td_full(pop, gov, r_term, ss_term, cs=None, update_Beq_xi=True, norisk=False, closed_economy=None, tol=1E-8, calibration_options=None, h_mult=None, migration=False, LIC_wedge=0, fertility_scenario="medium", alt_debt=False, fixed_debt=False, alt_ret=False, FLFP_gap=None, vintage_UNPP=False, sigma=2, LIC_integration="baseline"):
    if calibration_options is None:
        calibration_options = {'case': 'homothetic'} if norisk else {}

    r_init, gamma, hh, prod_init, _, _, ss_init = load_initial_ss(**calibration_options, LIC_wedge=LIC_wedge)
    
    # if need to compute the case of a closed economy
    if closed_economy is not None:
      hh = {k: v for k, v in hh.items() if k in [closed_economy]}
      prod_init = {k: v for k, v in prod_init.items() if k in [closed_economy]}
      ss_init = {k: v for k, v in ss_init.items() if k in [closed_economy]}
      
    if cs is None:
        cs = list(hh)

    if h_mult is not None:
        # h_mult should be a dict mapping countries to T*J array of multiplicative changes in h
        h_mult_mtx = list(h_mult.values())[0]
        hh = {c: hh[c].update_h(h_mult_mtx[c]) for c in cs}
    
    c_curr = list(pop)[0]    
    T = pop[c_curr].pij.shape[0]

    # load Jacobians for terminal steady state
    fixed_debt = True if np.isscalar(gov[c_curr].B_Y) else False
    
    common_args = {
        'h_mult': h_mult,
        'calibration_options': {'case': 'homothetic'},
        'fixed_debt': fixed_debt,
        'migration': migration,
        'closed_economy': closed_economy,
        'fertility_scenario': fertility_scenario,
        'alt_debt': alt_debt,
        'alt_ret': alt_ret,
        'FLFP_gap': FLFP_gap,
        'vintage_UNPP': vintage_UNPP,
        'LIC_wedge': LIC_wedge,
        'sigma': sigma,
        'LIC_integration': LIC_integration
    }
    Jhh = jac.load_terminal_jacobians(T, norisk=norisk, **common_args)

    # initial guess for r and Beq_xi (latter in each country)
    if closed_economy is not None:
        r_init += LIC_wedge
        wedge_path = {'LIC': np.zeros((T,))}
        wedge_init = {c: 0 for c in cs}
    else:
        y1 = 2016  # Starting year
        y2 = {'LIC': 2035, 'CHN': 2075, 'IND': 2075} if LIC_integration == "faster" else {c: 2075 for c in ['LIC', 'CHN', 'IND']} # Year to reach 0 interest rate wedge
        
        target_years = np.arange(y1,y1 + T)
        
        wedge_path = {c: np.where(
            (target_years>=y1) & (target_years<=y2[c]), LIC_wedge - LIC_wedge * (target_years-y1)/(y2[c]-y1),
            0,
            ) for c in ['LIC', 'CHN', 'IND']}   # specify the wedge path
        if LIC_integration == "slower":
            wedge_path['LIC'] = np.full(T, LIC_wedge)
        
        # wedge_init = {c: 0 if c != 'LIC' else LIC_wedge for c in cs}
        wedge_init = {c: 0 if c not in ['LIC', 'CHN', 'IND'] else LIC_wedge for c in cs}
        #################################################################################################################################
        
    rguess = np.linspace(r_init, r_term, T)[1:]
    
    Beq_xi_guesses = {c: make_Beq_guess(ss_init[c], ss_term[c], T, update_Beq_xi) for c in cs}
    x_cur = {c: np.concatenate((rguess, Beq_xi_guesses[c])) for c in cs}
    
    # Set up multiprocessing pool
    pool = mp.Pool(2)

    # main loop
    Rd = 50
    print(f'Solving world TD with update_Beq_xi = {update_Beq_xi}')
    for rd in range(Rd):
        print(f'STARTING Round {rd}')

        x_cur['LIC'][:T-1] += wedge_path['LIC'][1:]
        if closed_economy is None:
            x_cur['CHN'][:T-1] += wedge_path['CHN'][1:]
            x_cur['IND'][:T-1] += wedge_path['IND'][1:]
        #################################################################################################################################
        # Prepare arguments for each country
        args_list = [(c, x_cur[c], r_init + wedge_init[c], ss_init[c], T, update_Beq_xi, pop[c], hh[c], gov[c], ss_term[c], prod_init[c], gamma) for c in cs]

        # Use the pool to parallelize the loop
        results = pool.starmap(process_country, args_list)
        #results = [process_country(c, x_cur[c], r_init, ss_init[c], T, update_Beq_xi, pop[c], hh[c], gov[c], ss_term[c], prod_init[c], gamma) for c in cs]

        # Process the results
        NFA = np.zeros(T-1)
        Y, W_Y, NFA_Y, K_Y, PB_Y, C_Y, Inc_Y, G_Y, tau, d_bar, prod, Beq_err, dY_dr = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for c, Y_c, W_Y_c, NFA_Y_c, K_Y_c, PB_Y_c, C_Y_c, Inc_Y_c, G_Y_c, tau_c, d_bar_c, prod_c, Beq_err_c, dY_dr_c in results:
            Y[c], W_Y[c], NFA_Y[c], K_Y[c], PB_Y[c], C_Y[c], Inc_Y[c], G_Y[c], tau[c], d_bar[c], prod[c], Beq_err[c], dY_dr[c] = Y_c, W_Y_c, NFA_Y_c, K_Y_c, PB_Y_c, C_Y_c, Inc_Y_c, G_Y_c, tau_c, d_bar_c, prod_c, Beq_err_c, dY_dr_c
            NFA += Y[c][1:] * NFA_Y[c][1:]

        # Adjust NFA by initial period
        NFA += sum(Y[c][0] * NFA_Y[c][0] for c in cs)

        NFA_error = np.max(np.abs(NFA)) if closed_economy is None else np.max(np.abs(NFA[:50]))                # some numerical error with the autarky case, but not important
        Beq_error = np.max([np.max(np.abs(Beq_err[c])) for c in cs]) if closed_economy else np.max([np.max(np.abs(Beq_err[c][:50])) for c in cs])
        print(f'Round {rd}: NFA error = {NFA_error}, Beq error = {Beq_error}')

        if NFA_error < tol:
            break
            # if not update_Beq_xi:
            #     break
            # elif Beq_error < tol:
            #     break

        # get full country-level Jacobian by subtracting asset supply vs r Jacobian
        Jfull = {c: Jhh[c].copy() for c in cs}
        for c in cs:
            Jfull[c][:T-1, :T-1] -= jac.make_As_jacobian(rguess, prod[c])

        # update step: get Jacobians and inverses, solve for new x
        x_old = x_cur
        Js = jac.get_world_jacobian_submatrices(Jfull, Y, NFA_Y, dY_dr)
        if update_Beq_xi:
            Jinvs = jac.get_inverse_jacobian_submatrices(*Js)
            x_cur = jac.update_x(NFA, Beq_err, x_old, Jinvs)
        else:
            J_NFA_r = Js[0]
            dr = -np.linalg.solve(J_NFA_r, NFA)
            for c in x_cur:
                x_cur[c][:T-1] += dr

    else:  # no break
        pool.close()
        pool.join()
        raise ValueError(f'Failed to converge after {Rd} rounds, max NFA error = {np.max(np.abs(NFA))}')

    pool.close()
    pool.join()

    r = {c: np.concatenate(([r_init + wedge_init[c]], x_cur[c][:T-1])) for c in cs}

    # return final guess
    return r, W_Y, NFA_Y, K_Y, Y, PB_Y, C_Y, Inc_Y, G_Y, tau, d_bar


def process_country(c, x_cur_c, r_init, ss_init_c, T, update_Beq_xi, pop_c, hh_c, gov_c, ss_term_c, prod_init_c, gamma):
    print(f'Country {c}')
    r, Beq_xi = extract_paths(x_cur_c, r_init, ss_init_c, T)
    
    if not update_Beq_xi:
        Beq_j_xi = hh_c.bequest_rule(Beq_xi[0], pop_c.pij[0])
        Beq_j_xi = np.tile(Beq_j_xi, (T, 1, 1))
    else:
        Beq_j_xi = hh_c.bequest_rule(Beq_xi, pop_c.pij)
    prod_c, _, td, W_Y_c, NFA_Y_c, K_Y_c, Y_c, PB_Y_c, C_Y_c, Inc_Y_c, G_Y_c, tau_c, d_bar_c = calculate_country_td_given_bequests(r, Beq_j_xi, pop_c, gov_c, ss_init_c['D'], ss_term_c['Va'], hh_c, prod_init_c, gamma)
    dY_dr_c = jac.get_dlogY_dr(r, prod_c) * Y_c
    Beq_err_c = (td['Beq_xi_implied'] - Beq_xi)[1:].T.ravel()
    return c, Y_c, W_Y_c, NFA_Y_c, K_Y_c, PB_Y_c, C_Y_c, Inc_Y_c, G_Y_c, tau_c, d_bar_c, prod_c, Beq_err_c, dY_dr_c


# def evaluate_td_guess(x_cur, cs, pop, gov, hh, r_init, prod_init, ss_init, ss_term, gamma, update_Beq_xi=True):
#     T = len(pop[cs[0]].pij)
#     NFA = np.zeros(T-1)
#     Y, W_Y, NFA_Y, prod, Beq_err, dY_dr = {}, {}, {}, {}, {}, {}
#     for c in cs:
#         print(f'Country {c}')
#         r, Beq_xi = extract_paths(x_cur[c], r_init, ss_init[c], T)
#         if not update_Beq_xi:
#             Beq_j_xi = hh[c].bequest_rule(Beq_xi[0], pop[c].pij[0])
#             Beq_j_xi = np.tile(Beq_j_xi, (T, 1, 1))
#         else:
#             Beq_j_xi = hh[c].bequest_rule(Beq_xi, pop[c].pij)
#         prod[c], _, td, W_Y[c], NFA_Y[c], Y[c] = calculate_country_td_given_bequests(r, Beq_j_xi, pop[c], gov[c], ss_init[c]['D'], ss_term[c]['Va'], hh[c], prod_init[c], gamma)
#         dY_dr[c] = jac.get_dlogY_dr(r, prod[c])*Y[c]
#         Beq_err[c] = (td['Beq_xi_implied'] - Beq_xi)[1:].T.ravel()
#         NFA += Y[c][1:]*NFA_Y[c][1:]
    
#     # adjust NFA by initial period; since full collection of countries calibrated to zero NFA, only relevant if doing subset
#     NFA += sum(Y[c][0]*NFA_Y[c][0]  for c in cs)
#     return NFA, Y, W_Y, NFA_Y, prod, Beq_err, dY_dr


def make_Beq_guess(ss_init, ss_term, T, update_Beq_xi=True):
    N_xi = len(ss_init['Beq_xi_implied'])
    if not update_Beq_xi:
        ss_term = ss_init # just use initial steady state throughout
    Beq_xi_guesses = np.tile(ss_init['Beq_xi_implied'], (T-1, 1))
    return np.concatenate(Beq_xi_guesses)


def calculate_country_td_given_bequests(r, Beq_j_xi, pop: Population, gov: Government, D_init, Va_term, hh: Household, prodss: Production, gamma):
    prod = prodss.adjust_r(r)
    gov = gov.adjust(pop, prod, hh, r, gamma)
    td = hh.td_givenbequests(r, Beq_j_xi, prod.w_Z, pop, gov, D_init, Va_term, gamma)
    NFA_Y, Y = ss.calculate_NFAY_and_Y(td, hh, pop, gov, prod)
    W_Y = NFA_Y + prod.K_Y + gov.B_Y if np.isscalar(gov.B_Y) else NFA_Y + prod.K_Y + gov.B_Y[:-1]
    
    L, Nret = pop.get_L(hh, gov.rho), pop.get_Nret(gov.rho)
    w_Z = prod.w_Z

    PB_Y = gov.primary_balance(L, w_Z, Nret, prod.Y_ZL * L)
    C_Y = td['C'] / (prod.Y_ZL * L / pop.N)
    Inc_Y = (w_Z * (L * (1-gov.tau) + Nret * gov.d_bar) + r * td['W']) / (prod.Y_ZL * L)

    return prod, gov, td, W_Y, NFA_Y, prod.K_Y, Y, PB_Y, C_Y, Inc_Y, gov.G_Y, gov.tau, gov.d_bar


def calculate_country_soe_td(r, pop: Population, gov: Government, ss_init, ss_term, hh: Household, prodss: Production, gamma, J_BeqBeq, tol=1E-8):
    T = len(r)
    Beq_xi_guess_flat = np.zeros((T-1)*hh.N_xi)

    Rd = 30
    print('Solving country-level SOE TD')
    for rd in range(Rd):
        Beq_xi = extract_Beq_xi(Beq_xi_guess_flat, ss_init, T)
        Beq_j_xi = hh.bequest_rule(Beq_xi, pop.pij)
        _, _, td, W_Y, NFA_Y, K_Y, Y, PB_Y, C_Y, Inc_Y, G_Y, tau, d_bar = calculate_country_td_given_bequests(r, Beq_j_xi, pop, gov, ss_init['D'], ss_term['Va'], hh, prodss, gamma)
        Beq_err_flat = (td['Beq_xi_implied'] - Beq_xi)[1:].T.ravel()
        
        err = np.max(np.abs(Beq_err_flat))
        print(f'Round {rd}: max Beq error = {err}')
        if err < tol:
            break

        Beq_xi_guess_flat -= np.linalg.solve(J_BeqBeq, Beq_err_flat)
    else:
        raise ValueError(f'Failed to converge after {Rd} rounds, max Beq error = {err}')
    
    return W_Y, NFA_Y, K_Y, Y, PB_Y, C_Y, Inc_Y, G_Y, tau, d_bar


def extract_Beq_xi(Beq_xi_flat, ss_init, T):
    # extract Beq_xi from a flattened x when we're only iterating over one country's Beq_xi
    Beq_xi = np.empty((T, len(ss_init['Beq_xi_implied'])))
    Beq_xi[0] = ss_init['Beq_xi_implied']
    Beq_xi[1:] = Beq_xi_flat.reshape((-1, T-1)).T
    return Beq_xi


def extract_paths(x, r_init, ss_init, T):
    """Extracts r and Beq_xi from a flattened x we're using to iterate"""
    r = np.concatenate(([r_init], x[:T-1]))
    N_xi = len(x) // (T - 1) - 1
    Beq_xi = np.empty((T, N_xi))
    Beq_xi[0] = ss_init['Beq_xi_implied']
    Beq_xi[1:] = x[T-1:].reshape((-1, T-1)).T
    return r, Beq_xi


# def calculate_world_td_by_exercise(r_term, ss_term, fiscal_rule='all', fixed_bequests=False,
#                     fixed_mortality=False, fixed_retirement=False, norisk=False, soe=False, cs=None, tol=1E-8,
#                     calibration_options=None, closed_economy=None, migration=False):
    
#     if calibration_options is None:
#         calibration_options = {'case': 'norisk'} if norisk else {}

#     r_init, gamma, hh, prod_init, pop_init, gov_init, ss_init = load_initial_ss(**calibration_options)

#     # if need to compute the case of a closed economy
#     if closed_economy is not None:
#       hh = {k: v for k, v in hh.items() if k in [closed_economy]}
#       prod_init = {k: v for k, v in prod_init.items() if k in [closed_economy]}
#       pop_init = {k: v for k, v in pop_init.items() if k in [closed_economy]}
#       gov_init = {k: v for k, v in gov_init.items() if k in [closed_economy]}
#       ss_init = {k: v for k, v in ss_init.items() if k in [closed_economy]}

#     if cs is None:
#         cs = list(hh)
    
#     # create specific population and government paths
#     pop = load_poptrans(fixed_mortality,migration)
#     T = len(pop[cs[0]].pij)

#     gov = {}
#     for c in cs:
#         if fixed_retirement:
#             gov[c] = gov_init[c].ss_to_td(T)
#         else:
#             gov[c] = gov_init[c].ss_to_td(T, age_increase=5, years_increase=60)
#         gov[c].adjust_rule = fiscal_rule

#     Jhh = jac.load_terminal_jacobians(T, norisk, closed_economy,migration=migration)
#     if soe:
#         r = np.full(T, r_init)
#         W_Y, NFA_Y, K_Y, Y, PB = {}, {}, {}, {}, {}
#         if not fixed_bequests:
#             for c in cs:
#                 print(c)
#                 W_Y[c], NFA_Y[c], K_Y[c], Y[c], PB[c] = calculate_country_soe_td(r, pop[c], gov[c], ss_init[c], ss_term[c], hh[c], prod_init[c], gamma, Jhh[c][T-1:, T-1:], tol)
#         else:
#             for c in cs:
#                 print(c)
#                 Beq_xi_init = ss_init[c]['Beq_xi_implied']
#                 Beq_j_xi_init = hh[c].bequest_rule(Beq_xi_init, pop_init[c].pij)
#                 Beq_j_xi = np.tile(Beq_j_xi_init, (T, 1, 1))
#                 *_, W_Y[c], NFA_Y[c], K_Y[c], Y[c], PB[c] = calculate_country_td_given_bequests(r, Beq_j_xi, pop[c], gov[c], ss_init[c]['D'], ss_term[c]['Va'], hh[c], prod_init[c], gamma)
#     else:
#         if not fixed_bequests:
#             r, W_Y, NFA_Y, K_Y, Y, PB_Y = solve_world_td_full(pop, gov, r_term, ss_term, cs, tol=tol, norisk=norisk, calibration_options = calibration_options)
#         else:
#             r, W_Y, NFA_Y, K_Y, Y, PB_Y = solve_world_td_full(pop, gov, r_term, ss_term, cs, update_Beq_xi=False, tol=tol, norisk=norisk, calibration_options = calibration_options)

#     return r, W_Y, NFA_Y, K_Y, Y, PB_Y


def Deltabar_log_WY(Y, W_Y):
    Delta_log_WY = {c: np.log(W_Y[c]) - np.log(W_Y[c][0]) for c in W_Y}
    W0s = {c: W_Y[c][0]*Y[c][0] for c in W_Y}
    return sum(Delta_log_WY[c]*W0s[c] for c in W0s) / sum(W0s.values())


def Delta_comp(poptrans, h, ss_init):
    Delta_comp = poptrans.log_shift_share(ss_init['W_j']) - poptrans.log_shift_share(h)
    return Delta_comp
