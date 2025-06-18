"""
Defines the household class containing routines for household problem
"""
import numpy as np
import copy
from scipy.interpolate import interp1d

from Code.interpolation import interpolate_coord, interpolate_y, forward_a, expectation_a
from Code.household_utils import markov_incomes, agrid

class Household:
    def __init__(self, h_adj, beq_dist, T, Tw, sigma, nu=None, upsilon=None,
                 abar=0.0, N_a=200, amax=1000, # changing to grid of 1000 now
                 rho_eps=0.91, sigma_eps=0.92, N_eps=11, beta_bar=None, beta_xi=None,
                 rho_xi=0.677, sigma_xi=0.37**0.5, N_xi=3, beq_discount=True):
        # Age structure parameters
        self.Tw = Tw           # Working life starts at this year
        self.T = T             # Last year of life, i.e. survive to T+1 with probability 0

        # Bequest distribution rule: how an amount of total bequests is distributed across ages
        self.b_dist = beq_dist

        # preference parameters
        self.sigma = sigma        # Curvature of own utility (inverse EIS)
        self.nu = nu              # Curvature of bequest utility
        self.upsilon = upsilon    # Bequest scaling factor, to calibrate desire for bequest
        self.beta_bar = beta_bar  # Subjective discount factor
        self.beta_xi = beta_xi    # Subjective discount factor
        self.beq_discount = beq_discount  # Whether bequests should be discounted

        # Asset grid parameters
        self.abar = abar        # Minimum asset position (borrowing constraint)
        self.amax = amax        # Maximum asset position
        self.N_a = N_a          # Number of asset gridpoints

        # Income process parameters
        self.rho_xi = rho_xi        # Intergenerational shock persistence parameter
        self.sigma_xi = sigma_xi    # Intergenerational shock standard deviation
        self.N_xi = N_xi            # Number of intergenerational shock states

        self.rho_eps = rho_eps      # Idiosyncratic shock persistence parameter
        self.sigma_eps = sigma_eps  # Idiosyncratic shock standard deviation
        self.N_eps = N_eps          # Number of idiosyncratic shock states

        #self.h = h                  # Productivity scaling factor by age
        self.h_adj = h_adj          # Differs from h after initial retirement age, use this for "non-retired"

        # Vector of ages of life from 0 to T
        self.jvec = np.arange(0, T + 1)  # vector of ages

        # Asset grid derived
        self.a = agrid(amin=abar, amax=amax, N=N_a)

        # Discretized processes for xi and eps
        self.y_xi, self.pi_xi, self.Pi_xi = markov_incomes(rho=rho_xi, sigma_y=sigma_xi, N=N_xi)
        self.y_eps, self.pi_eps, self.Pi_eps = markov_incomes(rho=rho_eps, sigma_y=sigma_eps, N=N_eps)

        # Combined values, frequencies, number of states
        # BY DEFAULT EVERYTHING WILL BE in FULL s=(xi, eps) COORDINATES NOW
        self.y_s = self.y_xi[:, np.newaxis] * self.y_eps[np.newaxis, :]
        self.pi_s = self.pi_xi[:, np.newaxis] * self.pi_eps[np.newaxis, :]
        self.N_s = N_xi * N_eps

        # Include productivity shifter and get labor income by (age, xi, eps)
        # This is only relevant for income, where after retirement there is 100% means testing
        # UPDATE: now allow for this to have (time, age, xi, eps) dimensions too for transition...
        self.y = self.h_adj[..., np.newaxis, np.newaxis] * self.y_s

        # Transition matrix WITHIN GENERATION (xi does not change)
        self.Pi = np.kron(np.eye(N_xi), self.Pi_eps)

        # Transition matrix ACROSS GENERATIONS (xi follows its process, random draw from eps)
        self.Pi_gen = np.kron(self.Pi_xi, np.tile(self.pi_eps, (N_eps, 1)))

        # check mean of productivities, not including age-shifter, is 1
        assert np.isclose(np.vdot(self.pi_s, self.y_s), 1)

    def update_h(self, h_mult):
        # updates h, will turn from ss to td if h_mult 2-dim
        hh = copy.deepcopy(self)
        hh.h_adj = h_mult * self.h_adj
        hh.y = hh.h_adj[..., np.newaxis, np.newaxis] * hh.y_s
        return hh
    
    # def get_ss_terminal(self):
    #     assert self.h_adj.ndim == 4 # only makes sense if transition now
    #     hh = copy.deepcopy(self)
    #     hh.h_adj = self.h_adj[-1]
    #     return hh

    def expect_eps(self, X):
        """Take expectations of X(xi, eps, a) with respect to eps transition."""
        # equivalent to np.einsum('ef,tfa->tea', self.Pi_eps, X)
        X_eps_xia = np.swapaxes(X, 0, 1).reshape(len(self.Pi_eps), -1)           # change X from (xi, eps, a) to (eps, xi*a)
        X_eps_xia = self.Pi_eps @ X_eps_xia                                      # take expectation over eps
        return np.swapaxes(X_eps_xia.reshape(X.shape[1], X.shape[0], -1), 0, 1)  # change back to (xi, eps, a)

    def forward_eps(self, D):
        """Update distribution D(xi, eps, a) with transition of eps."""
        # equivalent to np.einsum('ef,tfa->tea', self.Pi_eps.T, D)
        D_eps_xia = np.swapaxes(D, 0, 1).reshape(len(self.Pi_eps), -1)           # change D from (xi, eps, a) to (eps, xi*a)
        D_eps_xia = self.Pi_eps.T @ D_eps_xia                                    # iterate forward over eps
        return np.swapaxes(D_eps_xia.reshape(D.shape[1], D.shape[0], -1), 0, 1)  # change back to (xi, eps, a)

    def forward_asset(self, D, a_pol):
        """For D with dims (..., a), and a_pol with same dims, update distribution with a policy."""
        a_pol_i, a_pol_pi = interpolate_coord(self.a, a_pol)
        # a only takes a single state dimension other than a, so need to reshape back and forth
        return forward_a(D.reshape((-1, self.N_a)),
                         a_pol_i.reshape((-1, self.N_a)), a_pol_pi.reshape((-1, self.N_a))).reshape(D.shape)

    def expectation_asset_from_lottery(self, X, a_pol_i, a_pol_pi):
        return expectation_a(X.reshape((-1, self.N_a)),
                         a_pol_i.reshape((-1, self.N_a)), a_pol_pi.reshape((-1, self.N_a))).reshape(X.shape)

    def expectation_functions(self, a_pol):
        # one-time prep: get interpolated representation
        a_pol_i, a_pol_pi = interpolate_coord(self.a, a_pol)

        # up to maximum horizon "T", where are you in expectation?
        # note we need expectation function for beginning of period assets even in J+1 (certain death)
        T = self.T - self.Tw + 1
        curlyE = np.zeros((T+1, a_pol.shape[0]+1, *a_pol.shape[1:]))

        curlyE[0] = np.broadcast_to(self.a, curlyE.shape[1:])

        for t in range(1, T+1):
            # looking t periods in future
            for j in range(self.Tw, self.T + 2 - t):
                # effective_survival = pij[j+1] / pij[j] # includes migrants
                mid = self.expect_eps(curlyE[t-1, j+1])
                curlyE[t, j] = self.expectation_asset_from_lottery(mid, a_pol_i[j], a_pol_pi[j])
        
        return curlyE
    
    def scale_exp_by_age(self, curlyE, scale_j):
        # scale_j is a vector of length J+2 (0 to J+1) by which we want to scale expectations
        # we scale by your age in the future (at the date we're taking expectations of)
        curlyE = curlyE.copy()
        for t in range(len(curlyE)):
            for j in range(self.Tw, self.T + 2 - t):
                # you're age j today, we're taking expectation of you when you're age j+t 
                curlyE[t, j] *= scale_j[t + j]
        return curlyE

    def coh(self, income, r, beq):
        return (income[:, :, np.newaxis] + (1 + r) * (self.a + beq[:, np.newaxis, np.newaxis]))

    def income(self, w, gov):
        """Get (age, xi, eps) or (time, age, xi, eps) array of after-tax-and-transfer lifetime incomes."""
        if np.asarray(w).ndim == 0:
            # steady state
            rho = gov.rho[:, np.newaxis, np.newaxis]
            incomes = w * ((1 - rho) * (1 - gov.tau) * self.y + rho * gov.d_bar * self.y_xi[:, np.newaxis])
        else:
            # transition
            incomes = np.empty((len(w), self.T + 1, self.N_xi, self.N_eps))
            for t in range(len(w)):
                y = self.y[t] if self.y.ndim == 4 else self.y # allow y to be time-varying
                rho = gov.rho[t][:, np.newaxis, np.newaxis]
                incomes[t] = w[t] * ((1 - rho) * (1 - gov.tau[t]) * y + rho * gov.d_bar[t] * self.y_xi[:, np.newaxis])
        return incomes

    def beta_j(self):
        jvec = np.append(self.jvec, self.jvec[-1]+1)
        beta = np.minimum(np.exp(np.log(self.beta_bar) * jvec + self.beta_xi * (jvec - 40) ** 2), 1e10)
        # normalize so that level is 1 at age 95, only matters when beq_discount is False so this gives units of upsilon
        # this normalization ensures that when beq_discount=False, more patient betabar also means more bequests
        beta /= beta[self.T] 
        return beta

    def make_v_a_core(self):
        a_trunc = self.a.copy()  # Avoid division by zero when raised to a negative power
        a_trunc[self.a <= 1E-5] = 1E-5
        return self.upsilon * a_trunc ** (-self.nu)

    def cons_from_euler(self, Vap, betaratio, beta_j, gamma, phi):
        """Get consumption today on grid for assets tomorrow from Euler equation"""
        Vap_exp = self.expect_eps(Vap)
        if not hasattr(self, 'v_a_core') or self.last_nu != self.nu or self.last_upsilon != self.upsilon:
            self.v_a_core = self.make_v_a_core()
            self.last_nu = self.nu
            self.last_upsilon = self.upsilon
        
        v_a = (1 - phi) * (1 + gamma) ** (-self.nu) * self.v_a_core
        if not self.beq_discount:
            v_a /= beta_j # undo effect of discounting bequests, shifts v_a up by 1/beta_j
        return 1 / ((phi * betaratio * (1 + gamma) ** (-self.sigma)  * Vap_exp + v_a)) ** (1 / self.sigma)

    def backward_iterate(self, Vap, coh, r, betaratio, beta_j, gamma, phi):
        # Implied consumption today consistent with Euler equation (on tomorrow's grid for assets a')
        c_nextgrid = self.cons_from_euler(Vap, betaratio, beta_j, gamma, phi)

        # We have consumption today for each a' tomorrow (a mapping from total cash on hand today c'+(1+gamma)*a' 
        # to a' tomorrow). Interpolate to get mapping of actual cash on hand in each state to assets tomorrow a'
        a = interpolate_y(c_nextgrid + (1+gamma)*self.a, coh, self.a)

        # Borrowing constraint should never bind bc of Inada on bequests, but numerically might
        a = np.maximum(a, self.abar)
        
        c = coh - (1+gamma)*a
        uc = 1 / c ** self.sigma
        Va = (1+r) * uc
        return Va, a, c

    def bequest_rule(self, beq_xi, pij):
        """Given aggregate bequests by xi, how much do you receive by (age, xi) or (time, age, xi)?"""
        pij = (pij == 0)*1E-6 + pij # avoid division by zero, zeros in late years should have b_dist == 0 anyway
        if pij.ndim == 1:
            beq = (self.b_dist / pij)[:, np.newaxis] * beq_xi
            beq[:self.Tw, :] = 0.0
        elif pij.ndim == 2:
            if hasattr(self, 'constant_beq_r'):
                if self.constant_beq_r:
                    # Here use the first distribution only to distribute bequests
                    beq = (self.b_dist / pij[0,:][np.newaxis, :])[:, :, np.newaxis] * beq_xi[:, np.newaxis, :]
                    beq[:, :self.Tw, :] = 0.0
            else:
                beq = (self.b_dist / pij)[:, :, np.newaxis] * beq_xi[:, np.newaxis, :]
                beq[:, :self.Tw, :] = 0.0
        return beq

    @property
    def D_zero(self):
        """Asset distribution that is degenerate at zero, only nontrivial if zero not on grid."""
        D = np.zeros(self.N_a)
        if self.a[0] == 0:
            D[0] = 1
        else:
            izero = max(np.argmax(self.a > 0) - 1, 0)  # first index less than zero
            D[izero] = self.a[izero + 1] / (self.a[izero + 1] - self.a[izero])
            D[izero + 1] = 1 - D[izero]
        return D

    def ss_givenbequests(self, Beq_j_xi, popss, govss, w, r, gamma):
        """Household steady state given aggregate bequests received by xi."""
        income = self.income(w, govss)
        beta_j = self.beta_j()

        Va, c, a, D, C_j, A_j, Beq_j = self.ss_household_solution(
                                                popss.phi, income, beta_j, Beq_j_xi, r, gamma)

        # Calculate implied bequests received within each type 'xi'
        ingoing_deaths_j = popss.get_ingoing_deaths()
        Beq_xi_given_implied = np.einsum('j,jxea,a->x', ingoing_deaths_j, D, self.a)
        Beq_xi_implied = (Beq_xi_given_implied @ self.Pi_xi) / self.pi_xi

        A, Beq, C = popss.pij @ A_j, popss.pij @ Beq_j, popss.pij @ C_j
        W_j = A_j + Beq_j
        W = A + Beq

        return {'Va': Va, 'D': D, 'c': c, 'a': a, 'D': D, 'Beq_xi_implied': Beq_xi_implied, 
                'C_j': C_j, 'A_j': A_j, 'Beq_j': Beq_j, 'W_j': W_j,
                'Beq': Beq, 'A': A, 'C': C, 'W': W}
    
    def ss_household_solution(self, phi, income, beta_j, Beq_j_xi, r, gamma):
        """"Fully PE" household-level steady-state solution, where households take income and bequests
        received as given, and we calculate their steady-state policy functions and distribution by age,
        without knowing the population distribution or doing any aggregation"""
        # Initialize arrays to store results and enforce 0 before age Tw
        # need to know savings of people who are alive at T and die at T+1, so need T+2 periods for D
        Va, c, a = (np.empty((self.T + 1, self.N_xi, self.N_eps, self.N_a)) for _ in range(3))
        D = np.empty((self.T + 2, self.N_xi, self.N_eps, self.N_a))
        Va[:self.Tw], c[:self.Tw], a[:self.Tw], D[:self.Tw] = 0.0, 0.0, 0.0, 0.0  # Ensure 0 before Tw

        # backward iteration to get policies by age
        Vap = np.zeros((self.N_xi, self.N_eps, self.N_a))
        for j in reversed(range(self.Tw, self.T + 1)):
            coh = self.coh(income[j], r, Beq_j_xi[j])  # Cash-on-hand in age j
            Va[j], a[j], c[j] = self.backward_iterate(
                Vap, coh, r, beta_j[j+1]/beta_j[j], beta_j[j], gamma, phi[j])
            Vap = Va[j]

        # forward iteration to get distribution by age
        # we emphasize the "by age" because D[j,...] is the **conditional** distribution given age j
        D[self.Tw] = self.pi_s[:, :, np.newaxis] * self.D_zero
        for j in range(self.Tw, self.T + 1):
            D_asset = self.forward_asset(D[j], a[j])
            D[j + 1] = self.forward_eps(D_asset)

        # aggregate consumption, assets, and bequests received by age
        C_j = np.einsum('jxea,jxea->j', c, D[:-1])
        A_j = np.einsum('a,jxea->j', self.a, D[:-1])
        Beq_j = np.einsum('jx,x->j', Beq_j_xi, self.pi_xi)

        return Va, c, a, D, C_j, A_j, Beq_j

    def td_givenbequests(self, r, Beq_j_xi, w, pop, gov, D_init, Va_term, gamma):
        """Household transitional dynamics given population, government, and production parameters."""
        # Get path of income given path of real wage from interest rate and social security parameters
        income = self.income(w, gov)

        # Get vector beta_j
        beta_j = self.beta_j()

        # Run transitional dynamics
        _, c, a, D, C_j, A_j, Beq_j = self.td_household_solution(
                        r, Beq_j_xi, income, pop.phi, D_init, Va_term, beta_j, gamma)

        # Calculate implied bequests received within each type 'xi'
        ingoing_deaths_j = pop.get_ingoing_deaths()
        Beq_xi_given_implied = np.einsum('tj,tjxea,a->tx', ingoing_deaths_j, D, self.a)
        Beq_xi_implied = (Beq_xi_given_implied @ self.Pi_xi) / self.pi_xi # intergenerational transmission

        # aggregation 
        A, Beq, C = np.sum(pop.pij*A_j, axis=1), np.sum(pop.pij*Beq_j, axis=1), np.sum(pop.pij*C_j, axis=1)
        W_j = A_j + Beq_j
        W = A + Beq

        return {'D': D, 'c': c, 'a': a, 'D': D, 'Beq_xi_implied': Beq_xi_implied, 
                'C_j': C_j, 'A_j': A_j, 'Beq_j': Beq_j, 'W_j': W_j,
                'Beq': Beq, 'A': A, 'C': C, 'W': W}

    def td_household_solution(self, r, Beq_j_xi, income, phi, D_init, Va_term, beta_j, gamma):
        """"Fully PE" transitional dynamics calculation, given paths of demographic parameters (survival probabilities,
        population distributions), initial distribution, and paths for income, interest rate and bequests received.

        Used as inner routine for 'td_givenbequests'."""
        Ttrans = len(r)

        # Initialize arrays to store results and enforce 0 before age Tw
        Va, c, a = (np.empty((Ttrans, self.T + 1, self.N_xi, self.N_eps, self.N_a)) for _ in range(3))
        D = np.empty((Ttrans, self.T + 2, self.N_xi, self.N_eps, self.N_a))
        Va[:, self.Tw], c[:, :self.Tw], a[:, :self.Tw], D[:, :self.Tw] = 0.0, 0.0, 0.0, 0.0

        # Backward iteration to get policies by year and age
        for t in reversed(range(Ttrans)):
            for j in reversed(range(self.Tw, self.T + 1)):
                coh = self.coh(income[t, j], r[t], Beq_j_xi[t, j])
                if j == self.T:
                    Vap = np.zeros_like(Va[0, 0])
                elif t == Ttrans - 1:
                    Vap = Va_term[j + 1]
                else:
                    Vap = Va[t + 1, j + 1]

                Va[t, j], a[t, j], c[t, j] = self.backward_iterate(
                    Vap, coh, r[t], beta_j[j+1]/beta_j[j], beta_j[j], gamma, phi[t, j])

        # Forward iteration to get distribution by age
        # D[t, j,...] is the conditional distribution at time t given age j

        # initial conditions coming into date 0 and beginning of working life
        D[0] = D_init
        for t in range(1, Ttrans):
            D[t, self.Tw] = self.pi_s[:, :, np.newaxis] * self.D_zero

        # go forward using policy from above
        for t in range(Ttrans-1):
            for j in range(self.Tw, self.T + 1):
                D_asset = self.forward_asset(D[t, j], a[t, j])
                D[t + 1, j + 1] = self.forward_eps(D_asset)

        # aggregate consumption, assets, and bequests received by age
        C_j = np.einsum('tjxea,tjxea->tj', c, D[:, :-1])
        A_j = np.einsum('a,tjxea->tj', self.a, D[:, :-1])
        Beq_j = np.einsum('tjx,x->tj', Beq_j_xi, self.pi_xi)

        return Va, c, a, D, C_j, A_j, Beq_j

    def pctl_bequests(self, percentiles_desired, popss, ss):
        """Returns holdings of bequests at desired percentiles divided by average holdings of bequests.
        """
        ingoing_deaths_j = popss.get_ingoing_deaths()
        D_ja = ss['D'].sum(axis=(1, 2)) # don't care about xi, eps states
        D_beq = ingoing_deaths_j @ D_ja
        D_beq /= np.sum(D_beq)
        
        cum_D_beq = np.concatenate(([0], np.cumsum(D_beq)))
        a_mid = 0.5 * (self.a[1:] + self.a[:-1])
        a = np.concatenate(([self.abar], a_mid, [self.amax]))

        beq_mean = D_beq @ self.a

        return interpolate_y(cum_D_beq, percentiles_desired, a) / beq_mean


    def update_params(self, THETA):
        """Updates parameter values with values contained in a dict 'THETA'."""
        for key, value in THETA.items():
            setattr(self, key, value)


def shiftshare(Ay_j, Ay_j_proj, pij, pij_proj, h, h_proj):
    """
    This function computes the statistics Delta_pi, Delta_a and Delta both in levels and from the
    formula for W/Y:
        (W/Y)(t) = sum{pij(t) * (a/y)(t)} / sum{pij(t) * h(t)}
    where 'pij' is the population distribution, 'a/y' is the age-wealth profile (normalized by ZN) 'a'
    divided by output per person 'y', and 'h' is the age-efficiency profile.

    Parameters
    ----------
    Ay_j: np.ndarray
        Age-wealth (A/ZN) profile over GDP per capita y(r) in the base year
    Ay_j_proj: np.ndarray
        Age-wealth (A/ZN) profile over GDP per capita y(r) in the projection
        year, or path from base year to projection year
    pij: np.ndarray
        Population shares in the base year
    pij_proj: np.ndarray
        Population shares in the projection year, or path from
        base year to projection year
    h: np.ndarray
        Efficiency profile in the base year
    h_proj: dict
        Efficiency profile in the projection year, or path from
        base year to projection year

    Returns
    -------
    delta: dict
        Dict containing statistics

    """
    # Wealth-to-GDP in the base year
    WY = np.sum(Ay_j * pij) / np.sum(h * pij)
    # Wealth-to-GDP in the projection years
    WY_proj = np.sum((Ay_j_proj * pij_proj).T, axis=0) / np.sum((h_proj * pij_proj).T, axis=0)

    # Delta_a:
    d_Aj = Ay_j_proj - Ay_j
    Delta_a = np.sum((pij * d_Aj).T, axis=0) / np.sum((pij * h).T, axis=0)

    # Delta_h:
    Delta_h = (np.sum((pij * Ay_j).T, axis=0) / np.sum((pij * h_proj).T, axis=0)
               - np.sum((pij * Ay_j).T, axis=0) / np.sum((pij * h).T, axis=0))

    # Delta_pi:
    Delta_pi = (np.sum((pij_proj * Ay_j).T, axis=0) / np.sum((pij_proj * h).T, axis=0)
                - np.sum((pij * Ay_j).T, axis=0) / np.sum((pij * h).T, axis=0))
    Delta_pi_log = (np.log(np.sum((pij_proj * Ay_j).T, axis=0) / np.sum((pij_proj * h).T, axis=0))
                    - np.log(np.sum((pij * Ay_j).T, axis=0) / np.sum((pij * h).T, axis=0)))

    # Delta:
    Delta = WY_proj - WY
    Delta_log = np.log(WY_proj) - np.log(WY)

    return {'Delta_pi': Delta_pi, 'Delta_pi_log': Delta_pi_log, 'Delta_h': Delta_h,  'Delta_a': Delta_a,
            'Delta': Delta, 'Delta_log': Delta_log, 'Delta_check_WY': WY_proj}


def shiftshare_fromtd(countries, td_outcomes, Ttrans):
    Delta = {}
    for country in countries:
        # Adjust 'h' with retirement age for albor supply profiles
        h_proj = td_outcomes[country]['h_gross'][:Ttrans]
        # Store wealth profiles that are s.t. W/Y = sum(a*pi) / sum(h*pi)
        Ay_j_proj = td_outcomes[country]['W_j'] / (td_outcomes[country]['Y'][:Ttrans] /
                                                   td_outcomes[country]['L'][:Ttrans])[:, np.newaxis]
        # Perform decomposition
        Delta[country] = shiftshare(
            Ay_j_proj[0], Ay_j_proj, td_outcomes[country]['pij'][0], td_outcomes[country]['pij'][:Ttrans],
            h_proj[0], h_proj)
        td_outcomes[country].update(Delta[country])

    return td_outcomes


def import_inputs_bdist(df, T, Tw=20):
    """Import rule for distribution of bequests by age."""
    # Given a total of bequests to distribute, 'beq_dist' specifies how to distribute by age
    b_data = df['dist'].loc[df['age_bin'] >= 20].values
    b_data_age = df['age_bin'].loc[df['age_bin'] >= 20].values

    # Use Cubic interpolation
    b_f = interp1d(b_data_age, b_data, 'cubic', fill_value="extrapolate")
    b_n = np.concatenate([np.zeros(Tw),b_f(np.arange(Tw,b_data_age[-1])),np.zeros(T + 1 - b_data_age[-1])])

    # Make sure sums to 1
    beq_dist = b_n / np.sum(b_n)

    return beq_dist


