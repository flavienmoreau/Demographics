"""
Routines specific for government.
"""
import numpy as np, pandas as pd
from Code.demographics import Population
from Code.production import Production

import copy

class Government:
    """Start with baseline levels of social security tax, income tax, and benefits, then given taxable income
    and mass of retirees, follow ss adjustment rule to get adjusted levels that balance ss budget"""

    def __init__(self, tau, d_bar, G_Y, B_Y, rho, adjust_rule='all'):
        # note: if all scalars except B_Y, will assume B_Y is 1 entry longer
        #tau, d_bar, G_Y, B_Y = regularize_shapes(tau, d_bar, G_Y, B_Y)
        
        self.tau = tau  # baseline tax
        self.d_bar = d_bar  # baseline benefits
        self.G_Y = G_Y  # baseline expenditures
        self.B_Y = B_Y  # baseline debt
        self.rho = rho # percent retired by age

        self.adjust_rule = adjust_rule  # adjustment rule chosen (if this is reference steady state)
        self.adjust_dispatch = {'tau': self.adjust_tau, 'd_bar': self.adjust_d_bar,
                                'G_Y': self.adjust_G_Y, 'all': self.adjust_all}

    @staticmethod
    def calibrate_ss(hh, prod, pop, r, gamma, tau, rho, BY_target, benefits_target):
        Y_ZL, w_Z = prod.Y_ZL, prod.w_Z
        L, Nret = pop.get_L(hh, rho), pop.get_Nret(rho)
        d_bar = benefits_target * Y_ZL * L / (w_Z * Nret)

        gov = Government(tau, d_bar, 0, BY_target, rho, 'G_Y')
        return gov.adjust(pop, prod, hh, r, gamma)
        
    def get_ss_terminal(self):
        govss = Government(self.tau, self.d_bar, self.G_Y[-1], self.B_Y[-1], self.rho)
        return govss

    def adjust(self, pop: Population, prod: Production, hh, r, gamma):
        """Do fiscal adjustment given baseline parameters, dispatching to correct method, to find SS gov rules"""
        # everything here is normalized by Z
        L, Nret = pop.get_L(hh, self.rho), pop.get_Nret(self.rho)

        if np.isscalar(r):
            g_ante = (1+gamma) * (1+pop.n) - 1
        else:
            Y = prod.Y_L * L
            g_ante = (1 + gamma) * Y[1:] / Y[:-1] - 1
            g_ante = np.append(g_ante, g_ante[-1]) 

        newgov = self.adjust_dispatch[self.adjust_rule](L, Nret, prod.w_Z, r, g_ante, prod.Y_ZL*L)

        # check for budget balance
        new_surplus = newgov.surplus_Y(L, Nret, prod.w_Z, r, g_ante, prod.Y_ZL*L)
        if not np.allclose(new_surplus, 0, atol=1E-14):
            raise ValueError(f'Budget balance does not hold after adjustment: {new_surplus}')

        return newgov

    def update_B_Y(self, B_Y_new):
        gov = copy.deepcopy(self)
        gov.B_Y = B_Y_new
        return gov

    def net_taxes(self, L, w_Z, mass_retirees):
        # means-testing tax on income post-retirement age is 100%
        return (self.tau * L - mass_retirees * self.d_bar) * w_Z

    def primary_balance(self, L, w_Z, mass_retirees, Y_Z):
      surplus_Y_excluding_B = self.net_taxes(L, w_Z, mass_retirees) / Y_Z - self.G_Y
      return surplus_Y_excluding_B

    def surplus_Y(self, L, mass_retirees, w_Z, r, g_ante, Y_Z):
        surplus_Y_excluding_B = self.net_taxes(L, w_Z, mass_retirees) / Y_Z - self.G_Y
        if np.isscalar(self.B_Y):
            return surplus_Y_excluding_B - self.B_Y * (r - g_ante)
        else:
            #assert False, 'Do not have B_Y adjustment now, below seems old / a bit wrong, missing (1+g) factor on chg in B_Y'
            #return surplus_Y_excluding_B - self.B_Y[:-1] * (r - g_ante) + (self.B_Y[1:] - self.B_Y[:-1])
            surplus_Y = surplus_Y_excluding_B - (1 + r) * self.B_Y[:-1] + (1 + g_ante) * self.B_Y[1:]
            return surplus_Y_excluding_B - (1 + r) * self.B_Y[:-1] + (1 + g_ante) * self.B_Y[1:]

    def deriv_surplus_tau(self, L, w, Y):
        return L * w / Y

    def deriv_surplus_d_bar(self, mass_retirees, w, Y):
        return - mass_retirees * w / Y

    def adjust_tau(self, L, mass_retirees, w, r, g_ante, Y):
        tau = self.tau - self.surplus_Y(L, mass_retirees, w, r, g_ante, Y) / self.deriv_surplus_tau(L, w, Y)
        return Government(tau, self.d_bar, self.G_Y, self.B_Y, self.rho)

    def adjust_d_bar(self, L, mass_retirees, w, r, g_ante, Y):
        d_bar = self.d_bar - self.surplus_Y(L, mass_retirees, w, r, g_ante, Y) / self.deriv_surplus_d_bar(mass_retirees, w, Y)
        return Government(self.tau, d_bar, self.G_Y, self.B_Y, self.rho)

    def adjust_G_Y(self, L, mass_retirees, w, r, g_ante, Y):
        G_Y = self.G_Y + self.surplus_Y(L, mass_retirees, w, r, g_ante, Y)
        return Government(self.tau, self.d_bar, G_Y, self.B_Y, self.rho)

    def adjust_all(self, L, mass_retirees, w, r, g_ante, Y):
        surplus = self.surplus_Y(L, mass_retirees, w, r, g_ante, Y)
        tau = self.tau - 1/3 * surplus / self.deriv_surplus_tau(L, w, Y)
        d_bar = self.d_bar - 1/3 * surplus / self.deriv_surplus_d_bar(mass_retirees, w, Y)
        G_Y = self.G_Y + 1/3 * surplus
        return Government(tau, d_bar, G_Y, self.B_Y, self.rho)

    def ss_to_td(self, Ttrans, age_increase=None, years_increase=None, B_Y=None, ret_path=None):
        # not super robust but after calling adjust should work?
        assert np.isscalar(self.tau) # should start with ss
        gov = copy.deepcopy(self)
        gov.d_bar = np.full(Ttrans, 1.*gov.d_bar)
        gov.tau = np.full(Ttrans, 1.*gov.tau)
        gov.G_Y = np.full(Ttrans, 1.*gov.G_Y)
        # assuming B_Y is a constant, so won't expand it here

        if age_increase is not None and years_increase is not None:
            gov.rho = np.empty((Ttrans, len(self.rho)))
            # increase retirement age by `age_increase` linearly over `years_increase` years
            for t in range(Ttrans):
                gov.rho[t] = gov.adjust_rho(self.rho, np.minimum(t * age_increase / years_increase, age_increase))
            assert years_increase < Ttrans
        elif ret_path is not None:
            gov.rho = np.empty((Ttrans, len(self.rho)))
            for t in range(Ttrans):
                gov.rho[t] = gov.adjust_rho(self.rho, ret_path[t] - ret_path[0])
        else:
            gov.rho = np.tile(self.rho, (Ttrans, 1))
        
        if B_Y is not None:
            gov = gov.update_B_Y(B_Y)
        return gov

    @staticmethod
    def adjust_rho(rho, shift):
        """move vector rho to the right by shift"""
        assert shift >= 0, 'adjust_rho shift must be non-negative'
        
        # use convex combination of two nearest integers to represent shift
        shift_floor = int(np.floor(shift))
        shift_ceil = shift_floor + 1
        floor_frac = shift_ceil - shift
        
        if shift_floor == 0:
            rho_shift_floor = rho
        else: 
            rho_shift_floor = np.concatenate((np.zeros(shift_floor), rho[:-shift_floor]))
        
        rho_shift_ceil = np.concatenate((np.zeros(shift_ceil), rho[:-shift_ceil]))
        return floor_frac * rho_shift_floor + (1 - floor_frac) * rho_shift_ceil
