import numpy as np
import pandas as pd
from scipy import optimize

from Code.utils import path_intermediates, path_calibration_targets
from Code.demographics import Population
from Code.production import Production
from Code.government import Government
from Code.equilibrium import walras_law
from Code.household import Household, import_inputs_bdist # TODO: clean this up and make it part of calibration


def load_calibration_dataframes():
    df_world = pd.read_csv(path_calibration_targets + "targets_global.csv")
    df_beqineq = pd.read_csv(path_calibration_targets + "targets_beqineq.csv")
    df_beqdistribution = pd.read_csv(path_calibration_targets + "targets_beqdistribution.csv")
    df_countries = pd.read_csv(path_calibration_targets + "targets_countries.csv").set_index('country')
    df_countries_age = pd.read_csv(path_calibration_targets + "targets_countries_age.csv")

    return df_world, df_beqineq, df_beqdistribution, df_countries, df_countries_age


def load_processed_calibration_data(norisk=False, beq_discount=True, low_socsec=False, LIC_wedge=0):
    df_world, _, df_beqdistribution, df_countries, df_countries_age = load_calibration_dataframes()
    countries = list(df_countries.index)

    # load world parameters
    r, delta, gamma, Tw, T = [df_world[k][0] for k in ('r', 'delta', 'gamma', 'Tw', 'T')]
    if 'gamma' in df_countries:
        gamma_init = df_countries['gamma'].to_dict()
    else:
        gamma_init = {c: gamma for c in countries}

    # obtain population and age-productivity data by country
    popss, h, h_adj, rho = {}, {}, {}, {}
    for c in countries:
        data = df_countries_age.loc[(df_countries_age['country'] == c)]
        phi, pij, h[c], h_adj[c], rho[c] = (data[k].values for k in ('phi', 'pij', 'h', 'h_adj', 'rho'))
        assert np.allclose(h[c], h_adj[c] * (1-rho[c])) # h is total labor income, h_adj is if not retired
        phi[-1] = 0  # in year T, 0 chance of survival to T+1
        popss[c] = Population(pij, phi, df_countries.loc[c, 'n'])
    
    # calibrate production in every country
    # wedge = {c: 0 if c != 'LIC' else LIC_wedge for c in countries}
    wedge = {c: 0 if c not in ['LIC', 'CHN', 'IND'] else LIC_wedge for c in countries}
    #################################################################################################################################
    prod = {c: Production.calibrate(df_countries.loc[c, 'KY_target'], df_countries.loc[c, 'Y'], r + wedge[c], delta) for c in countries}

    # construct household objects in every country (still require further calibration after this)
    hh_calib_params = {k: df_world[k][0] for k in ('sigma', 'rho_eps', 'sigma_eps', 'rho_xi', 'sigma_xi')}
    if norisk:
        hh_calib_params.update(dict(N_eps=1, N_xi=1))
    beq_dist = import_inputs_bdist(df_beqdistribution, T, Tw=Tw)  # bequests distribution rule
    hh = {}
    for c in df_countries.index:
        hh[c] = Household(h_adj[c], beq_dist, T, Tw=Tw, beq_discount=beq_discount, **hh_calib_params)

    # obtain government data in every country
    govss = {}
    for c in df_countries.index:
        tgt = {k: df_countries.loc[c, k] for k in ['benefits_target', 'tau', 'BY_target']}
        if low_socsec:
            tgt['benefits_target'] /= 10
        tgt['rho'] = rho[c]
        govss[c] = Government.calibrate_ss(hh[c], prod[c], popss[c], r + wedge[c], gamma_init[c], **tgt)

    # h should come normalized so that it's = 1 in every country, and agree with h_adj prior to retirement age
    for c in countries:
        assert np.max(np.abs(popss[c].get_L(hh[c], rho[c]) - 1)) < 1E-12

    return r, gamma, hh, prod, popss, govss, gamma_init


def load_full_calibration(case='all', vary='xi', beq_discount=True, low_socsec=False, LIC_wedge=0):
    """Load both exogenous calibration and calibrated household parameters"""
    r, gamma, hh, prod, pop, gov, gamma_init = load_processed_calibration_data((case=='norisk'), beq_discount, low_socsec, LIC_wedge)  
    suffix = calibration_suffix(case, vary, beq_discount, low_socsec)
    df_calib = pd.read_csv("@Import/Data/calibrated_parameters/" + f"params_calib{suffix}.csv").set_index('country')
    for c in hh:
        hh[c].update_params({k: df_calib.loc[c, k] for k in ['beta_bar', 'beta_xi', 'upsilon', 'nu']})
    return r, gamma, hh, prod, pop, gov, gamma_init


def load_initial_ss(case='all', vary='xi', beq_discount=True, low_socsec=False, LIC_wedge=0):
    """Load fully computed steady state"""
    r, gamma, hh, prod, pop, gov, gamma_init = load_full_calibration(case, vary, beq_discount, low_socsec, LIC_wedge)
    
    suffix = calibration_suffix(case, vary, beq_discount, low_socsec)
    df_calib = pd.read_csv("@Import/Data/calibrated_parameters/" + f"params_calib{suffix}.csv").set_index('country')
    ss = {}
    # wedge = {c: 0 if c != 'LIC' else LIC_wedge for c in hh}
    wedge = {c: 0 if c not in ['LIC', 'CHN', 'IND'] else LIC_wedge for c in hh}
    ################################################################################################################################
    for c in hh:
        Beq_xi_received = df_calib.loc[c, [f'Beq{i+1}' for i in range(hh[c].N_xi)]].values
        Beq_j_xi = hh[c].bequest_rule(Beq_xi_received, pop[c].pij)
        ss[c] = hh[c].ss_givenbequests(Beq_j_xi, pop[c], gov[c], prod[c].w_Z, r + wedge[c], gamma_init[c])

    return r, gamma, hh, prod, pop, gov, ss


def calibration_suffix(case, vary, beq_discount=True, low_socsec=False):
    if case == 'all' and vary == 'xi' and beq_discount and not low_socsec:
        return ""
    
    main = f'_{case}_vary{vary}' if case != 'noxiupsilon' else '_noxiupsilon'
    if not beq_discount:
        main += '_nobeqdiscount'
    if low_socsec:
        main += '_lowsocsec'
    return main


def calib_ss(case='all', vary='xi', nu=None, beq_discount=True, low_socsec=False, LIC_wedge=0):
    r, gamma, hh, prod, popss, govss, gamma_init = load_processed_calibration_data((case=='norisk'), beq_discount, low_socsec, LIC_wedge=LIC_wedge)
    
    _, df_beqineq, _, df_countries, df_countries_age = load_calibration_dataframes()

    # get WY_j_target and WY_j_target_age for each country
    WY_j_target, WY_j_target_age = {}, {}
    for c in df_countries.index:
        data = df_countries_age.loc[df_countries_age['country'] == c]

        if c == 'NLD':
            # manually drop highest age group for Netherlands, extremely high & hard to match
            data.loc[data['age'] > 90, 'WY_j_target'] = np.nan

        WY_j_target[c] = data['WY_j_target'].dropna().values
        WY_j_target_age[c] = data['age'].values[data['WY_j_target'].notna()]

    beq_pctl, beq_pctl_targets = df_beqineq['beq_dist_pct'].values, df_beqineq['beq_dist'].values

    Beq_xi_received = {}

    # first calibrate the US household side, in main case to get the nu and upsilon
    tgt = {'WY_j_target': WY_j_target['USA'], 'WY_j_target_age': WY_j_target_age['USA'],
           'beq_dist': beq_pctl_targets, 'beq_dist_pct': beq_pctl, 'WY_target': df_countries.loc['USA', 'WY_target']}
    c = 'USA'
    hh[c], ss = calib_usa(hh[c], popss[c], govss[c], prod[c].w_Z, r, gamma_init['USA'], prod[c].Y_ZL, tgt, case=case, nu=nu)
    Beq_xi_received[c] = ss['Beq_xi_implied']
    err = walras_law(hh[c], ss, popss[c], govss[c], prod[c], df_countries['NFAY_target'][c], r, gamma)  # check Walras' law
    print(f'WALRAS LAW USA: {err}')

    # now calibrate the other countries
    vary = None if case == 'noxiupsilon' else vary # nothing to vary if xi=upsilon=0
    for c in df_countries.index[::-1]:
        if c == 'USA':
            continue
        tgt = {'WY_j_target': WY_j_target[c], 'WY_j_target_age': WY_j_target_age[c],
               'WY_target': df_countries.loc[c, 'WY_target']}
        # if c == 'LIC':
        if c == 'LIC' or c == 'CHN' or c == 'IND':      ###########################################################################################################
            hh[c], ss = calib_notusa(hh['USA'], hh[c], popss[c], govss[c], prod[c].w_Z, r + LIC_wedge, gamma_init[c], prod[c].Y_ZL, tgt, vary=vary)
        else:
            hh[c], ss = calib_notusa(hh['USA'], hh[c], popss[c], govss[c], prod[c].w_Z, r, gamma_init[c], prod[c].Y_ZL, tgt, vary=vary)
        Beq_xi_received[c] = ss['Beq_xi_implied']

        err = walras_law(hh[c], ss, popss[c], govss[c], prod[c], df_countries['NFAY_target'][c], r, gamma_init[c])  # check Walras' law
        print(f'WALRAS LAW {c}: {err}')
    
    # build a pandas dataframe to store the four calibrated parameters and Beq_xi_received
    df_calib = pd.DataFrame(index=df_countries.index, columns=['beta_bar', 'beta_xi', 'upsilon', 'nu'] + [f'Beq{i+1}' for i in range(hh[c].N_xi)])
    for c in df_countries.index:
        df_calib.loc[c] = [hh[c].beta_bar, hh[c].beta_xi, hh[c].upsilon, hh[c].nu, *Beq_xi_received[c]]
    suffix = calibration_suffix(case, vary, beq_discount, low_socsec)
    
    df_calib.to_csv("@Import/Data/calibrated_parameters/" + f"params_calib{suffix}.csv")
    
    return df_calib


def calibrate_WZ_bequests(hh, popss, govss, w, r, gamma, WZ_target):
    """Hit fixed point for bequests and calibrate 'beta_bar' to hit W/Z exactly"""
    
    def error_WZ_bequests(z): 
        hh.beta_bar, *Beq_xi = z
        #print(f'  trying beta_bar = {hh.beta_bar:.6f}')
        #hh.beta_bar = np.exp(log_beta_bar)
        Beq_j_xi = hh.bequest_rule(np.array(Beq_xi), popss.pij)
        ss_c = hh.ss_givenbequests(Beq_j_xi, popss, govss, w, r, gamma)
        errs = np.empty_like(z)
        errs[0] = (ss_c['W'] - WZ_target)
        errs[1:] = ss_c['Beq_xi_implied'] - np.array(Beq_xi)
        if np.isnan(errs).any():
            raise ValueError('NaN in calibration')
        return errs
    
    z_guess = np.array([1] + [0]*hh.N_xi)
    res = optimize.fsolve(error_WZ_bequests, z_guess)
    if np.isnan(res[0]):
        raise ValueError('NaN in calibration')
    hh.update_params({'beta_bar': res[0]})

    # Compute steady state at these parameters
    Beq_j_xi = hh.bequest_rule(res[1:], popss.pij)
    return hh, hh.ss_givenbequests(Beq_j_xi, popss, govss, w, r, gamma)


def calib_usa(hh, popss, govss, w_Z, r, gamma, Y_Z, targets_country, case, nu=None):
    """Calibration routine for the USA."""
    def assign_params(x):
        if case=='all':
            hh.beta_xi, hh.upsilon, hh.nu = x[0], np.exp(x[1]), x[2]
        elif case=='norisk':
            assert nu is not None, 'need to exogenously set nu in no-risk case'
            hh.beta_xi, hh.upsilon, hh.nu = x[0], np.exp(x[1]), nu
        elif case == 'noxi':
            hh.beta_xi, hh.upsilon, hh.nu = 0, np.exp(x[0]), x[1]
        elif case == 'noupsilon':
            hh.beta_xi, hh.upsilon, hh.nu = x[0], 1E-5, hh.sigma
        elif case == 'noxiupsilon':
            hh.beta_xi, hh.upsilon, hh.nu = 0, 1E-5, hh.sigma
        elif case == 'homothetic':
            hh.beta_xi, hh.upsilon, hh.nu = x[0], np.exp(x[1]), hh.sigma
        else:
            raise ValueError('Invalid case')
    
    guesses = {'xi': 0, 'upsilon': np.log(100), 'nu': 1.5}
    guesses_by_case = {'all': [guesses['xi'], guesses['upsilon'], guesses['nu']],
                       'norisk': [guesses['xi'], guesses['upsilon']],
                       'noxi': [guesses['upsilon'], guesses['nu']],
                       'noupsilon': [guesses['xi'], guesses['nu']],
                       'homothetic': [guesses['xi'], guesses['upsilon']], # for homothetic case, nu is fixed at sigma
    }

    def errors_tomin(x):
        """Returns sum of squared errors of wealth profile and Beq/Y at given (beta_bar, beta_xi, upsilon, nu)"""
        assign_params(x)

        _, ss_c = calibrate_WZ_bequests(hh, popss, govss, w_Z, r, gamma, targets_country['WY_target']*Y_Z)

        # Compute sum of squared-errors of wealth profile and bequest distribution
        errs_WY_j = (ss_c['A_j'][targets_country['WY_j_target_age']] / Y_Z - targets_country['WY_j_target'])
        errors_Bpctls = hh.pctl_bequests(targets_country['beq_dist_pct'], popss, ss_c) - targets_country['beq_dist']

        # error to minimize
        if case == 'all' or case == 'noxi':
            # here can target both wealth profile and bequest distribution
            to_min = np.hstack((errs_WY_j, errors_Bpctls))
        elif case == 'noupsilon' or case == 'norisk' or case == 'homothetic':
            # without utility from bequests or without risk or with nu fixed, can't target bequest distribution
            to_min = errs_WY_j

        print(f'(beta_bar, beta_xi, Upsilon, nu) = ({hh.beta_bar:.6f}, {hh.beta_xi:.6f}, {hh.upsilon:.6f}, {hh.nu:.6f})')
        print(f'Err = {to_min @ to_min:.6f}')
        print('___________________________________________________________')
        return to_min

    if case != 'noxiupsilon':
        res = optimize.least_squares(errors_tomin, np.array(guesses_by_case[case]), method='lm')
        assign_params(res.x)
        print(res.message)
    else:
        assign_params(None)

    hh, ss = calibrate_WZ_bequests(hh, popss, govss, w_Z, r, gamma, targets_country['WY_target']*Y_Z)

    return hh, ss


def calib_notusa(hh_usa, hh, popss, govss, w_Z, r, gamma, Y_Z, targets_country, vary):
    """Calibration routines for countries other than USA."""

    def assign_params(x):
        hh.nu = hh_usa.nu
        if vary == 'xi':
            hh.beta_xi, hh.upsilon = x[0], hh_usa.upsilon
        elif vary == 'upsilon':
            hh.beta_xi, hh.upsilon = hh_usa.beta_xi, np.exp(x[0])
        elif vary == 'xiupsilon':
            hh.beta_xi, hh.upsilon = x[0], np.exp(x[1])
        elif vary is None:
            hh.beta_xi, hh.upsilon = hh_usa.beta_xi, hh_usa.upsilon
        else:
            raise ValueError('Invalid "vary" option')

    def errors_tomin(x):
        """Returns sum of squared errors of wealth profile and Beq/Y at given (beta_bar, beta_xi, upsilon nu)"""
        assign_params(x)
        print(f'  trying beta_xi = {hh.beta_xi:.6f} and upsilon = {hh.upsilon:.6f}')
        _, ss_c = calibrate_WZ_bequests(hh, popss, govss, w_Z, r, gamma, targets_country['WY_target']*Y_Z)

        # Compute sum of squared-errors of wealth profile, bequests-to-GDP ratio and bequests distribution
        to_min = (ss_c['W_j'][targets_country['WY_j_target_age']] / Y_Z - targets_country['WY_j_target'])

        print(f'    (beta_bar, beta_xi, upsilon) = ({hh.beta_bar:.6f}, {hh.beta_xi:.6f}, {hh.upsilon:.6f})')
        print(f'    W/Y = {ss_c["W"] / Y_Z :.3f} vs data {targets_country["WY_target"]:.3f}')
        print(f'    Err = {to_min @ to_min:.6f}')
        print('___________________________________________________________')

        return to_min

    # Perform minimization
    guesses_by_vary = {'xi': [0], 'upsilon': [np.log(100)], 'xiupsilon': [0, np.log(100)]}
    if vary is not None:
        res = optimize.least_squares(errors_tomin, np.array(guesses_by_vary[vary]), method='lm')
        assign_params(res.x)
    else:
        assign_params(None)

    hh, ss = calibrate_WZ_bequests(hh, popss, govss, w_Z, r, gamma, targets_country['WY_target']*Y_Z)

    return hh, ss