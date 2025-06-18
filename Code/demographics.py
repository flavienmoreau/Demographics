"""
Routines specific to demographics
"""
import numpy as np
import pandas as pd
from Code.utils import path_data_inputs, path_intermediates

class Population:
    """Demographic variables for a population

    Attributes
    ----------
    pij : np.ndarray
        pij[j] is percent of population at age j
    phi : np.ndarray
        phi[j] is mortality risk between ages j and j+1 (same length as pij, last element 0)
    n : np.ndarray
        n[t] is growth rate of population between dates t-1 and t
    """
    def __init__(self, pij, phi, n, phi_actual=None):
        # added N giving total population, normalized to 1 in SS and 1 in first period
        self.pij = pij
        self.phi = phi

        # if this is not None, then phi is perceived mortality, and use this for ingoing deaths 
        self.phi_actual = phi_actual 

        if np.asarray(n).ndim != 0:
            self.n = n # make sure n[0] is steady-state n, same across all countries
            self.N = np.concatenate(([1], np.cumprod(1+n[1:])))
        else:
            self.n = n
            self.N = 1

    def migrants(self):
        phi = self.phi_actual if self.phi_actual is not None else self.phi
        if self.pij.ndim == 1:
            m = np.zeros_like(self.pij)
            # pij_(j+1) = m_(j+1) + phi_j*pij_j / (1+n)
            m[1:] = self.pij[1:] - phi[:-1]  * self.pij[:-1] / (1 + self.n)
        elif self.pij.ndim == 2:
            m = np.zeros_like(self.pij)
            # assume that first period demographics coincide with steady state
            m[0, 1:] = self.pij[0, 1:] - phi[0, :-1] * self.pij[0, :-1] / (1 + self.n[0])

            # pij_(t+1, j+1) = m_(t+1, j+1) + phi_(t, j) * pij_(t, j) / (1+n_(t+1))
            m[1:,1:] = self.pij[1:, 1:] - phi[:-1, :-1] * self.pij[:-1, :-1] / (1 + self.n[1:])
        return m
    
    def get_ingoing_deaths(self):
        """For each age j (and sometimes time t), return mass dying between j-1 and j (from t-1 to t)"""
        phi = self.phi_actual if self.phi_actual is not None else self.phi

        if self.pij.ndim == 1: # steady state
            # mass_death_j gives deaths between j-1 and j (zero for j=0)
            mass_death_j = np.zeros(len(self.pij) + 1)
            mass_death_j[1:] = (1 - phi) * self.pij / (1 + self.n)
            return mass_death_j
        elif self.pij.ndim == 2: # transition
            # mass_death_tj gives deaths between time t-1 / age j-1 and time t / age j
            mass_death_tj = np.zeros((self.pij.shape[0], self.pij.shape[1] + 1))
            
            # special case for t=0: assume t=0 demographics coincide with previous steady state
            mass_death_tj[0, 1:] = (1 - phi[0]) * self.pij[0] / (1 + self.n[0])

            # in general, use survival prob and population shares from previous period
            mass_death_tj[1:, 1:] = (1 - phi[:-1, :]) * self.pij[:-1, :] / (1 + self.n[1:, np.newaxis])
            return mass_death_tj
        
    def get_ss_terminal(self):
        assert self.pij.ndim == 2
        popss = Population(self.pij[-1], self.phi[-1], self.n[-1],
                           phi_actual=self.phi_actual[-1] if self.phi_actual is not None else None)
        popss.N = (1+self.n[-1])*self.N[-1] # not great
        return popss
    
    def get_ss_t(self, t):
        # just make steady state from t-th period
        assert self.pij.ndim == 2
        popss = Population(self.pij[t], self.phi[t], self.n[t],
                           phi_actual=self.phi_actual[t] if self.phi_actual is not None else None)
        popss.N = self.N[t]
        return popss

    def get_L(self, hh, rho):
        # sum over age dimension (always last, either only in ss case or second in transition)
        return np.sum((1-rho) * self.pij * hh.h_adj, axis=-1) * self.N

    def get_Nret(self, rho):
        return (rho * self.pij).sum(axis=-1) * self.N

    def ss_to_td(self, Ttrans):
        assert self.pij.ndim == 1 # should be ss right now
        pij = np.tile(self.pij, (Ttrans, 1))
        phi = np.tile(self.phi, (Ttrans, 1))
        n = np.full(Ttrans, self.n)
        return Population(pij, phi, n)
    
    def log_shift_share(self, x):
        # calculate log shift-share for some profile in x
        return np.log(self.pij @ x) - np.log(self.pij[0] @ x)


def load_poptrans(fixed_mortality=False, migration=False, closed_economy=None, fixed_demog=False, fertility_scenario="medium", vintage_UNPP=False):
    
    suffix = "_mig" if migration else ""
    suffix += f"_{fertility_scenario}" if fertility_scenario != "medium" else ""
    folder = "vintage_UNPP/" if vintage_UNPP else ""

    df_countries = pd.read_csv(f"@Import/Data/calibration_targets/{folder}targets_trans_countries{suffix}.csv")
    df_countries_age = pd.read_csv(f"@Import/Data/calibration_targets/{folder}targets_trans_countries_age{suffix}.csv")
    
    # these now include previous years
    df_countries = df_countries.loc[df_countries['year'] >= 2016]
    df_countries_age = df_countries_age.loc[df_countries_age['year'] >= 2016]
    
    if closed_economy is not None:
        df_countries = df_countries.loc[df_countries['isocode'] == closed_economy]
        df_countries_age = df_countries_age.loc[df_countries_age['isocode'] == closed_economy]
    
    # If fix age structure, apply the same age structure after 2016    
    if fixed_demog:
        cs = df_countries_age['isocode'].unique()
        df_countries_age = df_countries_age.set_index(['isocode','year','age_bin']).sort_index()
        pi_2016 = df_countries_age.query('year==2016')['pi']
        
        for country in cs:
          for year in range(2017,2401):
            df_countries_age.loc[(country,year),'pi'] = pi_2016.loc[country].values

    pijs = {}
    phis, phis_actual = {}, {}
    for c, df_c in df_countries_age.groupby('isocode'):
        pijs[c] = df_c.pivot_table(index='year', columns='age_bin', values='pi').values
        phis[c] = df_c.pivot_table(index='year', columns='age_bin', values='phi').values
        phis_actual[c] = None

        if fixed_mortality:
            # for constant perceived mortality, set phi to be constant at first period value
            # phi is only used for (1) household optimization (for which perceived mortality matters)
            # and (2) calculating ingoing deaths (which is shut off for the constant mortality case
            # because we are already using constant bequests)
            phis_actual[c] = phis[c]
            phis[c] = np.tile(phis[c][0], (len(phis[c]), 1))

    ns = {c: df_c['n'].values for c, df_c in df_countries.groupby('isocode')}

    return {c: Population(pijs[c], phis[c], ns[c], phis_actual[c]) for c in pijs}


def pop_stationary(n, phi, T=95):
    """Computes the stationary distribution at given growth rate survival probabilities."""
    phi_lag = np.append(1, phi[:-1])
    Phi = np.cumprod(phi_lag)
    n_cum = (1 + n) ** np.arange(0, T + 1)
    pi0 = 1 / np.sum(Phi / n_cum)
    pi = Phi / n_cum * pi0

    return pi

def pij_no_migration(N_j_init, phi, n):
    """Simulates population forward given initial population by age, survival probabiliies
    by age, and growth rate of newborns."""
    N_j = np.empty((phi.shape[0]+1, len(N_j_init)))
    N_j[0] = N_j_init  # initial population
    for t in range(phi.shape[0]):
        N_j[t + 1][0] = N_j[t][0] * (1 + n[t])  # newborns
        N_j[t+1][1:] = N_j[t, :-1] * phi[t, :-2]

    N = np.sum(N_j, axis=1)  # Total population

    return np.einsum('tj,t->tj', N_j[:-1], 1/N[:-1]), np.diff(np.log(N))  # return shares and growth rate

def pij_simul(Nj_init, M_j, phi, births):
    """Simulate population forward according to
        N(j,t) = (N(j-1,t-1) + M(j,t)) * phi_(j,j-1,t)
    where N(j,t) is given with N(0,t) births and net migration M(j,t)."""
    N_bins, T = len(Nj_init), len(births)
    N_j, pi_j = (np.empty((N_bins, T)) for _ in range(2))
    N_j[:, 0] = Nj_init  # initial population
    pi_j[:, 0] = Nj_init / np.sum(Nj_init)  # initial population shares
    for t in range(T-1):
        N_j[0, t + 1] = births[t + 1]
        N_j[1:, t + 1] = (N_j[:-1, t] + M_j[1:, t + 1]) * phi[:-1, t]
        pi_j[:, t + 1] = N_j[:, t + 1] / np.sum(N_j[:, t + 1])  # share

    return N_j, pi_j  # Return population by age and shares

def population_simul(pop_sim, end_year, varname='', n_c=-0.005, Tn_c=100, nomigration_from=2500,
                     migration_fixed_from=2500, mortality_fixed_from=2100, births_fixed_from=2100):
    """ Simulates population forward with option to hold fixed mortality/fertility and
     include or not net migration."""
    year_init = pop_sim['year'].unique()[0]
    countries = pop_sim['isocode'].unique()
    pop_sim['N_j_simul' + varname], pop_sim['pi_j_simul' + varname], pop_sim['phi_j_simul' + varname] = .0, .0, .0
    for c in countries:

        # Initial population by age in year "year_init"
        Nj_init = pop_sim.loc[
            (pop_sim['isocode'] == c) & (pop_sim['year'] == year_init),
            'N_j'].values

        # Survival probabilities (fixed after year "mortality_fixed_from")
        phi = pop_sim.dropna().loc[
            (pop_sim['isocode'] == c) & (pop_sim['year'] <= mortality_fixed_from)][['year', 'age_bin', 'phi']] \
            .set_index(['year', 'age_bin']).unstack(level=0).values
        phi_simul = np.hstack((phi, np.tile(phi[:, -1][:, np.newaxis], end_year - mortality_fixed_from)))

        # Births (cgrowth rate converging to n_c by 2200 then constant)
        births = pop_sim.dropna().loc[
            (pop_sim['isocode'] == c) & (pop_sim['year'] <= births_fixed_from) & (pop_sim['age_bin'] == 0),
            'N_j'].values
        births_upto2100 = np.append(births, np.tile(births[-1], 2100 - births_fixed_from))[:-1]
        # from 2100: growth rate linearly converges to n_c in 2200 and fixed afterwards
        n_c_proj = np.linspace(np.log(births_upto2100[-1] / births_upto2100[-2]), n_c, num=Tn_c+1)
        births_post2100 = births_upto2100[-2] * np.cumprod(1 + np.append(n_c_proj, np.tile(n_c_proj[-1], end_year-(2100+Tn_c))))
        births_simul = np.append(births_upto2100, births_post2100)

        # Net migration (fixed after last year of data)
        M_j = pop_sim.dropna().loc[
            pop_sim['isocode'] == c][['year', 'age_bin', 'M_j']] \
            .set_index(['year', 'age_bin']).unstack(level=0).values[:, :-1]
        T_decay = 50
        M_j_decay = np.hstack((M_j, M_j[:, -1][:, np.newaxis] * np.linspace(1, 0, T_decay)))
        M_j_simul = np.hstack((M_j_decay, np.tile(M_j_decay[:, -1][:, np.newaxis], end_year - max(pop_sim.dropna()['year']) - T_decay+1)))

        # If option to have fixed migration
        if migration_fixed_from <= end_year:
            M_j_simul[:, (migration_fixed_from - year_init + 1):] = M_j_simul[:, migration_fixed_from - year_init + 1][:, np.newaxis]

        # If option to have no migration after a specified year
        if nomigration_from <= end_year:
            M_j_simul[:, (nomigration_from - year_init + 1):] = 0.

        N_j, pi_j = pij_simul(Nj_init, M_j_simul, phi_simul, births_simul)

        pop_sim.loc[(pop_sim['isocode'] == c) & (pop_sim['year'] <= end_year), 'N_j_simul' + varname] = N_j.T.flatten()
        pop_sim.loc[(pop_sim['isocode'] == c) & (pop_sim['year'] <= end_year), 'pi_j_simul' + varname] = pi_j.T.flatten()
        pop_sim.loc[(pop_sim['isocode'] == c) & (pop_sim['year'] <= end_year), 'phi_j_simul' + varname] = phi_simul.T.flatten()

    return pop_sim


def demographic_inputs(year, start_year=1952, end_year=2300, varname='', nomigration_from=2301,
                       mortality_fixed_from=2100, births_fixed_from=2100, T=95):
    """Returns demographic inputs in a given year from UN data."""
    if year > end_year:
        end_year = year  # overwrite if desired year is higher than end_year

    # If some options are not default: return actual population shares and actual mortality rates
    pop = population(start_year=start_year, end_year=end_year, varname=varname, nomigration_from=nomigration_from,
                     mortality_fixed_from=mortality_fixed_from, births_fixed_from=births_fixed_from, T=T)

    return {'phi': np.squeeze(pop['phi'][np.argwhere(pop['years'] == year), :]),
            'pij': np.squeeze(pop['pij'][np.argwhere(pop['years'] == year), :]),
            'children': np.squeeze(pop['children'][np.argwhere(pop['years'] == year), :]),
            'children_avg': pop['children_avg'][np.argwhere(pop['years'] == year).squeeze()],
            'n': pop['n_annual'][np.argwhere(pop['years'] == year).squeeze()],
            'year': year}

def population(start_year=2016, end_year=2300, varname='', nomigration_from=2301,
               mortality_fixed_from=2100, births_fixed_from=2100, T=95):
    """
    Reads in demographic inputs from the UN and simulates population according to
            N(j,t) = ( N(j-1,t-1) + M(j-1,t-1)) * phi_{j,j-1,t}
    under specific assumptions about mortality/births/migration.
    """

    # Import population by age and migration (file extends to 2300, to simulate after 2300 need to extend this file)
    pop_simul = pd.read_csv(path_data_inputs + 'demographics/UN/UN_data_full.csv')
    pop_simul = pop_simul.loc[pop_simul['age_bin'] <= T]

    # Simulate population under specified assumptions
    pop_simul = population_simul(pop_simul, end_year, varname=varname, nomigration_from=nomigration_from,
                                 mortality_fixed_from=mortality_fixed_from, births_fixed_from=births_fixed_from)
    pop = pop_simul.loc[(pop_simul['isocode'] == 'USA') & (pop_simul['year'] >= start_year) & (pop_simul['year'] <= end_year)]

    # Years
    years = np.arange(start_year, end_year + 1, 1)
    # Population share
    pij = pop[['year', 'age_bin', 'pi_j_simul']].set_index(['year', 'age_bin']).unstack(level=0).values.T
    pij_age = pop['age_bin'].unique()
    # Annual population
    N_annual = pop[['year', 'N_j_simul']].groupby('year').sum().values
    # Population growth rate (convention is n[j] is growth from N[j] to N[j+1])
    n_annual = (N_annual[1:] - N_annual[0:-1]) / N_annual[0:-1]
    n_annual = np.append(n_annual, n_annual[-1])
    # Total population by age
    Nj_annual = pop[['year', 'age_bin', 'N_j_simul']].set_index(['year', 'age_bin']).unstack(level=0).values.T
    # Survival probabilities
    phi_data = pop.loc[pop_simul['year'] <= 2100][['year', 'age_bin', 'phi']].dropna().set_index(['year', 'age_bin']).unstack(level=0).values
    phi = np.hstack((phi_data, np.tile(phi_data[:, -1][:, np.newaxis], end_year - 2100)))
    phi = np.vstack((phi, np.zeros(phi.shape[1])[np.newaxis,:])).T

    # Import dependents per adult
    dep_UN_data = pd.read_csv(path_data_inputs + 'demographics/UN/dependents_UN_data.csv')
    dep_USA = dep_UN_data.loc[(dep_UN_data['isocode'] == 'USA') & (dep_UN_data['age_bin'] <= pij_age[-1]) & (dep_UN_data['year'] >= start_year)]
    children = dep_USA[['year', 'age_bin', 'dependents']].set_index(['year', 'age_bin']).unstack(level=0).values
    children = np.hstack((children, np.tile(children[:, -1][:, np.newaxis], end_year - dep_USA['year'].unique()[-1]))).T
    children_avg = np.sum(pij * children, axis=1)

    return {'pij': pij,
            'pij_age': pij_age,
            'N_annual': N_annual,
            'n_annual': n_annual,
            'Nj_annual': Nj_annual,
            'phi': phi,
            'children': children,
            'children_avg': children_avg,
            'years': years}
