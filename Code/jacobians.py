import numpy as np
import copy
import pandas as pd

from Code.utils import path_cached_results
from Code import ss
from Code.table4 import exercises


"""Convenience functions to calculate cached terminal fake news and then load Jacobians from these"""

def calculate_terminal_fakenews(h_mult=None, norisk=False, closed_economy=None, calibration_options=None, migration=False, fixed_debt=True, fertility_scenario="medium", alt_debt=False, alt_ret=False, FLFP_gap=None, vintage_UNPP=False, LIC_wedge=0, sigma=2, LIC_integration="baseline"):
    # TODO: here just assume baseline or norisk exercise and take that, later generalize to allow for every exercise
    ex = 'Baseline' if not norisk else 'Adjust bequests received'
    if closed_economy is not None:
      ex = 'Autarky'
    
    if h_mult is not None:
        h_mult_scenario = list(h_mult.keys())[0]
        h_mult_mtx = list(h_mult.values())[0]
    else:
        h_mult_scenario = ""
        h_mult_mtx = None
        
    r_cached = pd.read_csv("@Import/Data/cached_results/cached_r_terms.csv", header=None, index_col=0)[1].to_dict()[ex]

    r_term, gamma, hh, prod_term, pop_term, gov_term, ss_term = ss.calculate_terminal_world_ss_by_exercise(h_mult=h_mult_mtx,closed_economy=closed_economy,r_cached=r_cached,calibration_options=calibration_options,migration=migration, fixed_debt=fixed_debt, alt_debt=alt_debt, alt_ret=alt_ret, vintage_UNPP=vintage_UNPP, fertility_scenario=fertility_scenario, LIC_wedge=LIC_wedge)
    Fs = {c: make_household_fakenews(r_term, gamma, hh[c], prod_term[c], pop_term[c], gov_term[c], ss_term[c]) for c in hh}
    
    suffix = '_mig' if migration is True else ''
    suffix += '_autarky' if closed_economy is not None else ''
    suffix += '_' + h_mult_scenario.replace(" ", "_") if h_mult_scenario != 'Baseline' else ''
    suffix += '_FLFP' + str(FLFP_gap) if FLFP_gap is not None else ''
    suffix += f"_{fertility_scenario}" if fertility_scenario != "medium" else ""
    suffix += '_altdebt' if alt_debt is True else ''
    suffix += '_altret' if alt_ret is True else ''
    suffix += '_vintage_UNPP' if vintage_UNPP is True else ''
    suffix += f"_{LIC_integration}_integration" if LIC_integration != "baseline" else ''
    suffix += f'_sigma{sigma}' if sigma != 2 else ''
  
    np.savez(path_cached_results + f"fakenews{suffix}.npz", **Fs)
    return Fs


def load_terminal_jacobians(T, h_mult=None, norisk=False, closed_economy=None, calibration_options=None, migration=False, fixed_debt=True, fertility_scenario="medium", alt_debt=False, alt_ret=False, FLFP_gap=None, vintage_UNPP=False, LIC_wedge=0, sigma=2, LIC_integration="baseline"):
    ex = 'Baseline' if not norisk else 'Adjust bequests received'
    if closed_economy is not None:
      ex = 'Autarky' 
      
    if h_mult is not None:
        h_mult_scenario = list(h_mult.keys())[0]
        h_mult_mtx = list(h_mult.values())[0]
    else:
        h_mult_scenario = ""
        h_mult_mtx = None
        
    r_cached = pd.read_csv("@Import/Data/cached_results/cached_r_terms.csv", header=None, index_col=0)[1].to_dict()[ex]

    r_term, _, hh, prod_term, pop_term, gov_term, ss_term = ss.calculate_terminal_world_ss_by_exercise(h_mult=h_mult_mtx,closed_economy=closed_economy, r_cached=r_cached,calibration_options=calibration_options,migration=migration, fixed_debt=fixed_debt, alt_debt=alt_debt, alt_ret=alt_ret, vintage_UNPP=vintage_UNPP, fertility_scenario=fertility_scenario, LIC_wedge=LIC_wedge)
    
    suffix = '_mig' if migration is True else ''
    suffix += '_autarky' if closed_economy is not None else ''
    suffix += '_' + h_mult_scenario.replace(" ", "_") if h_mult_scenario != 'Baseline' else ''
    suffix += '_FLFP' + str(FLFP_gap) if FLFP_gap is not None else ''
    suffix += f"_{fertility_scenario}" if fertility_scenario != "medium" else ""
    suffix += '_altdebt' if alt_debt is True else ''
    suffix += '_altret' if alt_ret is True else ''
    suffix += '_vintage_UNPP' if vintage_UNPP is True else ''
    suffix += f"_{LIC_integration}_integration" if LIC_integration != "baseline" else ''
    suffix += f'_sigma{sigma}' if sigma != 2 else ''
      
    Fs = np.load(path_cached_results + f"fakenews{suffix}.npz")
    return {c: make_household_jacobian(Fs[c], r_term, ss_term[c]['W'], hh[c], gov_term[c], pop_term[c], prod_term[c], T) for c in prod_term}


def get_prod_sensitivity_r(r, prodss):
    h = 1E-4
    T = 5 # number of periods, after t=1 used as check
    r_up = r + h*(np.arange(5)==1)
    r_dn = r - h*(np.arange(5)==1)
    prod_up = prodss.adjust_r(r_up)
    prod_dn = prodss.adjust_r(r_dn)
    dw = (prod_up.w_Z - prod_dn.w_Z) / (2*h)
    dKY = (prod_up.K_Y - prod_dn.K_Y) / (2*h)
    dY = (prod_up.Y_ZL - prod_dn.Y_ZL) / (2*h)

    # verify only contemporaneous effects
    for dX in (dw, dKY, dY):
        assert np.allclose(dX[[0, 2, 3, 4]], 0)

    return dw[1], dKY[1], dY[1]
    

def get_gov_sensitivity_r(r, gamma, hh, prodss, popss, govss):
    h = 1E-4
    T = 5 # number of periods, after t=1 used as check
    pop = popss.ss_to_td(T) 
    gov = govss.ss_to_td(T)

    r_up = r + h*(np.arange(5)==1)
    r_dn = r - h*(np.arange(5)==1)
    prod_up = prodss.adjust_r(r_up)
    prod_dn = prodss.adjust_r(r_dn)
    gov_up = gov.adjust(pop, prod_up, hh, r_up, gamma)
    gov_dn = gov.adjust(pop, prod_dn, hh, r_dn, gamma)
    
    ddbar = (gov_up.d_bar - gov_dn.d_bar) / (2*h)
    dtau = (gov_up.tau - gov_dn.tau) / (2*h)

    for dX in (ddbar, dtau):
        # only anticipatory and contemporaneous effects
        assert np.allclose(dX[2:], 0)

    return ddbar[:2], dtau[:2]


def input_perturbation_from_r(r, gamma, hh, prodss, popss, govss, ss, h = 1E-5):
    # calculate perturbed inputs to household problem from r perturbation
    dw_dr, _, _ = get_prod_sensitivity_r(r, prodss)
    ddbar_dr, dtau_dr = get_gov_sensitivity_r(r, gamma, hh, prodss, popss, govss)
    
    T = hh.T - hh.Tw + 2 # maximum horizon is T-Tw+1
    Beq_xi = np.tile(ss['Beq_xi_implied'], (T, 1)) # bequests unchanged

    # add direct and indirect effects of final period r perturbation
    dr = h*(np.arange(T) == (T - 1))
    r_up = r + dr # perturb last r
    w_Z = prodss.w_Z + dw_dr * dr

    # very ugly, we're using our knowledge of what hh needs from gov here
    # come back to this someday and make it nicer
    gov_up = copy.copy(govss)
    gov_up.d_bar = gov_up.d_bar + 0*dr # get right dimensions
    gov_up.d_bar[-2:] += ddbar_dr * h
    gov_up.tau = gov_up.tau + 0*dr
    gov_up.tau[-2:] += dtau_dr * h
    gov_up.rho = gov_up.rho + 0*dr[:, np.newaxis]

    return r_up, Beq_xi, w_Z, gov_up


def input_perturbation_from_Beq(r, hh, prodss, govss, ss, h = 1E-5):
    # calculate perturbed inputs to household problem from Beq perturbation
    T = hh.T - hh.Tw + 2 # maximum horizon is T-Tw+1
    dzero = np.zeros(T)
    r_up = r + dzero          # r unchanged
    w_Z = prodss.w_Z + dzero  # w unchanged

    # also ugly, again fix this with better gov object
    gov_up = copy.copy(govss)
    gov_up.d_bar = gov_up.d_bar + dzero # d_bar unchanged
    gov_up.tau = gov_up.tau + dzero     # tau unchanged
    gov_up.rho = gov_up.rho + dzero[:, np.newaxis]       # Tr unchanged

    Beq_xi = np.tile(ss['Beq_xi_implied'], (T, 1))
    Beq_xi[T-1, :] += h

    return r_up, Beq_xi, w_Z, gov_up


def calculate_dD_dshocks(r, gamma, hh, prodss, popss, govss, ss, h = 1E-5):
    T = hh.T - hh.Tw + 2 # maximum horizon is T-Tw+1
    pop = popss.ss_to_td(T)

    shocks = [input_perturbation_from_r(r, gamma, hh, prodss, popss, govss, ss, h),
              input_perturbation_from_Beq(r, hh, prodss, govss, ss, h)]
    
    dD_dshocks = []
    for (r_up, Beq_xi, w_Z, gov_up) in shocks:
        Beq_j_xi = hh.bequest_rule(Beq_xi, pop.pij)
        a_up = hh.td_givenbequests(r_up, Beq_j_xi, w_Z, pop, gov_up, ss['D'], ss['Va'], gamma)['a']
        a_up = a_up[::-1, :] # reverse time so that we get change in policy by horizon of shock

        D_forward_up = np.zeros((T, * ss['D'].shape))
        for t in range(T):
            # for each horizon of shock, use the perturbed policy to iterate distribution forward from steady-state
            for j in range(hh.Tw, hh.T+1): # later, need last age to account for bequest effects
                D_asset = hh.forward_asset(ss['D'][j], a_up[t, j])
                D_forward_up[t, j + 1] = hh.forward_eps(D_asset)

        dD_dshock = (D_forward_up - ss['D']) / h
        dD_dshock[:, :hh.Tw+1] = 0 # no perturbation for agents who haven't worked yet
        dD_dshocks.append(dD_dshock)

    return dD_dshocks


def make_household_fakenews(r, gamma, hh, prodss, popss, govss, ss):
    """Household fake news matrix of AW with respect to shocks"""
    dD_r, dD_Beq = calculate_dD_dshocks(r, gamma, hh, prodss, popss, govss, ss)
    curlyE = hh.expectation_functions(ss['a'])

    curlyE_W = hh.scale_exp_by_age(curlyE, np.append(popss.pij, [0]))
    curlyE_Beq = hh.scale_exp_by_age(curlyE, popss.get_ingoing_deaths())
    curlyE_Beq /= hh.pi_xi[:, np.newaxis, np.newaxis] # Beq is normalized by pi_xi
    
    H = hh.T - hh.Tw + 2  # maximum horizon for anticipation or bequest effects is H-1

    # part 1: effect of r on everything else
    F_W_r = curlyE_W.reshape(H, -1) @ dD_r.reshape(H, -1).T
    F_Beqs_r = [curlyE_Beq[:, :, xi].reshape(H, -1) @ dD_r[:, :, xi].reshape(H, -1).T for xi in range(hh.N_xi)]

    # part 2: effect of Beq on everything else, noting that Beq only directly affects within xi type (before applying Pi_xi)
    F_W_Beqs    = [curlyE_W[:, :, xi].reshape(H, -1)   @ dD_Beq[:, :, xi].reshape(H, -1).T for xi in range(hh.N_xi)]
    F_Beqs_Beqs = [curlyE_Beq[:, :, xi].reshape(H, -1) @ dD_Beq[:, :, xi].reshape(H, -1).T for xi in range(hh.N_xi)]

    # part 3: now put in big (1 + N_xi) * (1 + N_xi) array of Fs
    F = np.zeros((1 + hh.N_xi, 1 + hh.N_xi, H+1, H))

    # effect of r on everything
    F[0, 0] = pad_zeros_first_row(F_W_r)
    for i, Fxi in enumerate(F_Beqs_r):
        F[i+1, 0] = pad_zeros_first_row(Fxi)

    # effect of Beqs on W
    for i, Fxi in enumerate(F_W_Beqs):
        F[0, i+1] = pad_zeros_first_row(Fxi)
        F[0, i+1, 0, 0] += hh.pi_xi[i] # special case: contemporaneous effect of Beq shock on W
    
    # effect of Beqs on itself (exploiting diagonality)
    for i, Fxi in enumerate(F_Beqs_Beqs):
        F[i+1, i+1] = pad_zeros_first_row(Fxi)

    # part 4: reallocate Beqs to descendants
    F[1:] *= hh.pi_xi[:, np.newaxis, np.newaxis, np.newaxis] # scale by pi_xi to get mass
    F[1:] = (hh.Pi_xi.T @ F[1:].reshape(hh.N_xi, -1)).reshape(F[1:].shape) # reallocate via Pi_xi
    F[1:] /= hh.pi_xi[:, np.newaxis, np.newaxis, np.newaxis] # renormalize by pi_xi

    return F


def pad_zeros_first_row(F):
    F = np.vstack((np.zeros(F.shape[1]), F))
    return F


def J_from_F(F, T):
    # embed smaller F into larger J
    J = np.zeros((T, T))
    J[:F.shape[0], :F.shape[1]] = F
    for t in range(1, T):
        J[1:, t] += J[:-1, t-1]
    return J


def make_As_jacobian(r, prodss):
    dKY_dr = -prodss.alpha/(r + prodss.delta)**2
    return np.diag(dKY_dr)

def get_dlogY_dr(r, prodss):
    alpha, delta = prodss.alpha, prodss.delta
    dlogKY_dr = - 1 / (r+delta)
    dlogY_dr = alpha / (1 - alpha) * dlogKY_dr
    return dlogY_dr

def manual_ss_test(r, prodss):
    # TODO: call somewhere
    h = 1E-5
    prod_up = prodss.adjust_r(r + h)
    prod_dn = prodss.adjust_r(r - h)
    dlogY = (prod_up.Y_ZL - prod_dn.Y_ZL) / (2*h) / prodss.Y_ZL
    assert np.isclose(dlogY, get_dlogY_dr(r, prodss))

def make_household_jacobian(F, r, W_ZN, hh, govss, popss, prodss, T):
    """Jacobian for entire system except need to subtract asset supply As from first output"""
    N_xi = F.shape[0] - 1
    J = np.empty((1 + N_xi, 1 + N_xi, T, T))
    for i1 in range(1 + N_xi):
        for i2 in range(1 + N_xi):
            J[i1, i2] = J_from_F(F[i1, i2], T)
    
    # normalize all effects on assets by Y_ZN
    L_N = popss.get_L(hh, govss.rho) / popss.N
    Y_ZN = prodss.Y_ZL * L_N
    J[0] /= Y_ZN
    
    # additionally, add GDP effect from r on ratio 
    _, _, dYZL_dr = get_prod_sensitivity_r(r, prodss)
    J[0, 0] -= W_ZN * np.eye(T) * dYZL_dr * L_N / Y_ZN**2

    # finally, subtract identity matrix to get derivatives of Beq_implied - Beq 
    for i in range(N_xi):
        J[i+1, i+1] -= np.eye(T)
    
    # return as 2D matrix, ignore first period since we won't iterate over that
    return J[:, :, 1:, 1:].swapaxes(1, 2).reshape((1+N_xi)*(T-1), (1+N_xi)*(T-1))


def broyden_update(J, dx, dy):
    """Broyden's method for updating a Jacobian"""
    return J + np.outer(dy - J @ dx, dx) / (dx @ dx)


def get_world_jacobian_submatrices(Jfull, Y, NFA_Y, dY_dr):
    """Calculate submatrices of world Jacobian"""
    # note: Y, NFA_Y, dY_dr all have first period, need to remove
    c_curr = list(Y)[0] 
    T = len(Y[c_curr])

    # *** Special attention: J_NFA_r, which has two terms ***
    # term 1: direct GDP-weighted aggregation of J_NFAY_r^c
    J_NFA_r_1 = sum(Y[c][1:, np.newaxis] * Jfull[c][:T-1, :T-1] for c in Y)

    # term 2: interaction between changes in Y and existing NFA path
    J_NFA_r_2 = np.diag(sum(dY_dr[c][1:] * NFA_Y[c][1:] for c in Y))

    J_NFA_r = J_NFA_r_1 + J_NFA_r_2

    # ** Remaining matrices **
    J_NFA_Beq = {c: Y[c][1:, np.newaxis] * Jfull[c][:T-1, T-1:] for c in Y}
    J_Beq_r = {c: Jfull[c][T-1:, :T-1] for c in Y}
    J_Beq_Beq = {c: Jfull[c][T-1:, T-1:] for c in Y}

    return J_NFA_r, J_NFA_Beq, J_Beq_r, J_Beq_Beq


def build_full_jacobian(J_NFA_r, J_NFA_Beq, J_Beq_r, J_Beq_Beq):
    """Recombine submatrices into full world Jacobian"""
    T = J_NFA_r.shape[0] + 1
    C = len(J_NFA_Beq)
    N_xi = J_NFA_Beq[list(J_NFA_Beq)[0]].shape[1] // (T-1) 

    Jfull = np.zeros(((T-1)*(N_xi*C+1), (T-1)*(N_xi*C+1)))
    print(f'N_xi: {N_xi}')

    Jfull[:T-1, :T-1] = J_NFA_r
    for i, c in enumerate(J_NFA_Beq):
        Jfull[:T-1, (N_xi*i+1)*(T-1):(N_xi*(i+1)+1)*(T-1)] = J_NFA_Beq[c]
    
    for i, c in enumerate(J_Beq_r):
        Jfull[(N_xi*i+1)*(T-1):(N_xi*(i+1)+1)*(T-1), :T-1] = J_Beq_r[c]
    
    for i, c1 in enumerate(J_Beq_Beq):
        Jfull[(N_xi*i+1)*(T-1):(N_xi*(i+1)+1)*(T-1), (N_xi*i+1)*(T-1):(N_xi*(i+1)+1)*(T-1)] = J_Beq_Beq[c1]
    return Jfull


def get_inverse_jacobian_submatrices(J_Ar, J_Ays, J_grs, J_gys):
    # using notation from "Jacobian inverse structure.pdf", tested in "jacobian inverse structure test.ipynb"
    cs = list(J_grs)

    key_sum = sum(J_Ays[c] @ np.linalg.solve(J_gys[c], J_grs[c]) for c in cs)
    Jinv_rA = np.linalg.inv(J_Ar - key_sum)

    Jinv_yAs = {c: -np.linalg.solve(J_gys[c], J_grs[c]) @ Jinv_rA for c in cs}

    Jinv_rgs = {c: -Jinv_rA @ J_Ays[c] @ np.linalg.inv(J_gys[c]) for c in cs}

    Jyg_invs = {c: np.linalg.inv(J_gys[c]) for c in cs}
    
    right_factors = {c: Jinv_rA @ J_Ays[c] @ Jyg_invs[c] for c in cs}
    left_factors = {c: np.linalg.solve(J_gys[c], J_grs[c]) for c in cs}

    Jinv_yk_gjs = {}
    for k in cs:
        Jinv_yk = {}
        for j in cs:
            Jform = left_factors[k] @ right_factors[j]
            if k == j:
                Jform += Jyg_invs[k]
            Jinv_yk[j] = Jform
        Jinv_yk_gjs[k] = Jinv_yk
    
    return Jinv_rA, Jinv_rgs, Jinv_yAs, Jinv_yk_gjs


def update_x(NFA, Beq_err, x_old, Jinvs):
    Jinv_r_NFA, Jinv_r_Beq, Jinv_Beq_NFA, Jinv_Beq_Beq = Jinvs

    T = len(NFA) + 1

    # what is the current guess?
    c_curr = list(x_old)[0] 
    r = x_old[c_curr][:T-1]
    Beq_xi = {c: x_old[c][T-1:] for c in x_old}

    # calculate implied update on dr and update
    dr = - (Jinv_r_NFA @ NFA + sum(Jinv_r_Beq[c] @ Beq_err[c] for c in Beq_err))

    # calculate implied update on Beq_xi country-by-country
    dBeq_xi = {}
    for c in x_old:
        dBeq_xi[c] = -Jinv_Beq_NFA[c] @ NFA
        for c2 in x_old:
            dBeq_xi[c] -= Jinv_Beq_Beq[c][c2] @ Beq_err[c2]

    # create x_new
    x_new = {}
    for c in x_old:
        x_new[c] = np.concatenate((r + dr, Beq_xi[c] + dBeq_xi[c]))
    
    return x_new


def update_x_alternate(NFA, Beq_err, x_old, Jgiant):
    T = len(NFA)
    err = np.concatenate((NFA, np.concatenate([Beq_err[c] for c in Beq_err])))
    dx = -np.linalg.solve(Jgiant, err)
    C = len(Beq_err)
    
    dr = dx[:T]
    dBeq_xi = dx[T:].reshape(C, -1)
    
    dx_c = {c: np.concatenate((dr, dBeq_xi[i])) for i, c in enumerate(Beq_err)}
    x_new = {c: x_old[c] + dx_c[c] for c in x_old}
    return x_new
