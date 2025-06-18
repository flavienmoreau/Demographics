#%%
import pickle
import copy 
from datetime import datetime
import numpy as np
import pandas as pd

from Code import ss as steady_state, td as transition
from Code.calibration import load_initial_ss
from Code import production, government, demographics, household, equilibrium
from Code.utils import path_data_inputs

def evaluate_exercise_slow(ex, t, r_cached=None, cs=None, tol=1E-8):
    # evaluate exercise and produce relevant row of Table 4

    # first: main GE exercise
    r_term, *_, ss_term = steady_state.calculate_terminal_world_ss_by_exercise(**exercises[ex], r_cached=r_cached)
    r, W_Y, NFA_Y, K_Y, Y      = transition.calculate_world_td_by_exercise(r_term, ss_term, **exercises[ex], cs=cs, tol=tol)

    # second: SOE
    r_term_soe, *_, ss_term_soe = steady_state.calculate_terminal_world_ss_by_exercise(**exercises[ex], soe=True)
    r_soe, W_Y_soe, NFA_Y_soe, K_Y_soe, Y_soe = transition.calculate_world_td_by_exercise(r_term_soe, ss_term_soe, soe=True, **exercises[ex], cs=cs, tol=tol)

    Delta_r         = r[t] - r[0]
    Deltabar_log_WY = transition.Deltabar_log_WY(Y, W_Y)[t]
    Deltabar_soe    = transition.Deltabar_log_WY(Y_soe, W_Y_soe)[t]

    extra = dict(r=r, W_Y=W_Y, NFA_Y=NFA_Y, Y=Y, r_soe=r_soe, W_Y_soe=W_Y_soe, NFA_Y_soe=NFA_Y_soe, Y_soe=Y_soe)

    return Delta_r, Deltabar_log_WY, Deltabar_soe, extra # epsbar_d, epsbar_s, extra


def evaluate_exercise_fast(ex, t, cs=None):

    norisk = exercises[ex].get("norisk", False)
     
    calibration_options = {'case': 'norisk'} if norisk else {}
    
    r_init, gamma, hh, prod_init, pop_init, gov_init, ss_init = load_initial_ss(**calibration_options)

    # if need to compute the case of a closed economy
    closed_economy = exercises[ex].get("closed_economy")
    if closed_economy is not None:
      hh = {k: v for k, v in hh.items() if k in [closed_economy]}
      prod_init = {k: v for k, v in prod_init.items() if k in [closed_economy]}
      pop_init = {k: v for k, v in pop_init.items() if k in [closed_economy]}
      gov_init = {k: v for k, v in gov_init.items() if k in [closed_economy]}
      ss_init = {k: v for k, v in ss_init.items() if k in [closed_economy]}
    
    poptrans = demographics.load_poptrans(fixed_mortality=exercises[ex].get('fixed_mortality', False))
    
    if cs is None:
        cs = list(hh)
    Ls                = {c: pop_init[c].get_L(hh[c], gov_init[c].rho) for c in cs} # All ones
    Ys                = {c: prod_init[c].Y_L * Ls[c] for c in cs}
    WYs              = {c: equilibrium.get_WY(ss_init[c], hh[c], pop_init[c], gov_init[c], prod_init[c]) for c in cs}
    W0s               = {c: Ys[c] * WYs[c] for c in cs}

    Delta_comp        = {c: transition.Delta_comp(poptrans[c], hh[c].h_adj * (1 - gov_init[c].rho), ss_init[c]) for c in cs}
    Deltabar_comp     = sum(W0s[c] * Delta_comp[c] for c in cs) / sum(W0s.values())
    Deltabar_comp     = Deltabar_comp[t]

    pop2100 = {c: poptrans[c].get_ss_t(t) for c in poptrans}

    gov2100 = copy.deepcopy(gov_init)
    for c in cs:
        if not exercises[ex].get("fixed_retirement", False):
             gov2100[c].rho = gov2100[c].adjust_rho(gov2100[c].rho, 5)
        if exercises[ex].get("fiscal_rule", False):
             gov2100[c].adjust_rule = exercises[ex]['fiscal_rule']
   
    if exercises[ex].get('fixed_bequests', False):
        Beq_j_xi = {c: hh[c].bequest_rule(ss_init[c]['Beq_xi_implied'], pop_init[c].pij) for c in cs}
    else:
        Beq_j_xi = {c: None for c in hh}
    
    eps_d = {c: steady_state.calculate_eps_d(hh[c], pop2100[c], gov2100[c], prod_init[c], r_init, gamma, Beq_j_xi=Beq_j_xi[c]) for c in cs}
    eps_s = {c: steady_state.calculate_eps_s(WYs[c], prod_init[c], r_init) for c in cs}

    epsbar_d = sum(W0s[c] * eps_d[c] for c in cs) / sum(W0s.values())
    epsbar_s = sum(W0s[c] * eps_s[c] for c in cs) / sum(W0s.values())

    extra_fast = dict(Delta_comp = Delta_comp, eps_d = eps_d, eps_s = eps_s)
   
    return Deltabar_comp, epsbar_d, epsbar_s, extra_fast 


# Part 1: specify exercises and solve for terminal steady state interest rates (or cache)

exercises = {'Baseline': {},
            'Autarky': dict(closed_economy='LIC')}
# exercises = {'Baseline': {},
#             'Drop annuities, add bequests': dict(fixed_retirement=True, fiscal_rule='G_Y', fixed_mortality=True, norisk = True, fixed_bequests=True),
#             'Adjust bequests received': dict(fixed_retirement=True,  fiscal_rule='G_Y', fixed_mortality=True, norisk=True),
#             'Add income risk': dict(fixed_retirement=True,  fiscal_rule='G_Y', fixed_mortality=True),
#             'Change perceived mortality': dict(fiscal_rule='G_Y', fixed_retirement=True),
#             'Increase retirement age ': dict(fiscal_rule='G_Y'),
#             'Only lower expenditures': dict(fiscal_rule='G_Y'),
#             'Only higher taxes': dict(fiscal_rule='tau'),
#             'Only lower benefits': dict(fiscal_rule='d_bar'),
#             'Autarky': dict(closed_economy='NGA')}
