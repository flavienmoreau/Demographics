"""
Various utility functions.
"""
import numpy as np
from numba import njit, jit
import pandas as pd

# Export and import paths
path_tables = '@Export/Tables/'
path_graphs = '@Export/Graphs/'
path_results = '@Import/Data/results/'
path_data_inputs = '@Import/Data/input_data/'
path_intermediates = '@Import/Data/intermediate_data/'
path_calibration_targets = '@Import/Data/calibration_targets/'
path_cached_results = '@Import/Data/cached_results/'

round_digits = 2  # Round to 2 digits in exports
tol = 1E-11  # Tolerance level for optimization

def mean(x, pr):
    """"
    Compute the mean.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the values in x

    Returns
    -------
    float
        Mean of x

    """
    if np.ndim(x) == 1:
        pr = pr / np.sum(pr)
        return np.sum(pr * x)
    elif np.ndim(x) == 2:
        return np.sum(pr*x, axis=1)

def cov(x, y, pr):
    """"
    Compute the covariance.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    y: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the values in x, y

    Returns
    -------
    float
        Covariance between x and y

    """
    if np.ndim(x) == 1:
        pr = pr / np.sum(pr)
        return np.sum(pr * (x - mean(x, pr)) * (y - mean(y, pr)))
    elif np.ndim(x) == 2:
        return np.sum(pr * (x - mean(x, pr)[:, np.newaxis]) * (y - mean(y, pr)[:, np.newaxis]), axis=1)

def var(x, pr):
    """"
    Compute the variance.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the values in x

    Returns
    -------
    float
        Variance of x

    """
    return cov(x, x, pr)

def std(x, pr):
    """"
    Compute the standard deviation.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the values in x

    Returns
    -------
    float
        Standard deviation of x

    """
    return np.sqrt(var(x, pr))

def corr(x, y, pr):
    """"
    Compute the correlation.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    y: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the values in x, y

    Returns
    -------
    float
        Correlation between x and y

    """
    return cov(x, y, pr)/(std(x, pr)*std(y, pr))

def power(x, a):
    """Raise x to the power a"""
    return np.exp(np.log(x) * a)

def make_path(x, T):
    """Takes in x as either a number, a vector or a matrix, turning it into a path."""
    x = np.asarray(x)
    if x.ndim <= 1:
        return np.tile(x, (T, 1))

    elif x.ndim == 2:
        return np.tile(x, (T, 1, 1))

def make_full_path(x, T):
    """Takes a path x (vector/matrix), and repeats the last line until x has T lines."""
    if x.ndim == 1:
        x = x[:, np.newaxis]

    if T < x.shape[0]:
        raise ValueError('T must be greater than the number of lines in x')

    return np.vstack((x, make_path(x[-1], T - x.shape[0])))

@njit
def within_tolerance(x1, x2, tol):
    """
    Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2
    """
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True

@njit
def maxabs(x1, x2):
    """Returns max(abs(x1-x2)). """
    return np.max(np.abs(x1-x2))

@njit
def setmin_2D(x, xmin):
    """Set 2-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                x[i, j] = xmin
            else:
                break

@njit
def setmin_3D(x, xmin):
    """Set 3-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj, nk = x.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if x[i, j, k] < xmin:
                    x[i, j, k] = xmin
                else:
                    break

def td_world_export(countries, td, name, start_year=2016, end_year=2300):
    # Export aggregates
    Ttrans_years = np.arange(start_year, end_year + 1, 1)
    N_c, Ttrans = len(countries), len(Ttrans_years)
    df_export = pd.DataFrame(
        [np.repeat(countries, Ttrans),
         np.tile(Ttrans_years, N_c)], index=['isocode', 'year']).T\
        .assign(r=np.nan, NFAY=np.nan, AY=np.nan, AnetmigY=np.nan)
    for country in countries:
        df_export.loc[df_export['isocode'] == country, 'r'] = td['r'][:Ttrans]
        for var in ['NFAY', 'AY', 'WY', 'AnetmigY', 'IY', 'Y', 'KY', 'CY',
                    'tau', 'd_bar', 'BY', 'GY',
                    'Delta_pi', 'Delta_pi_log', 'Delta', 'Delta_log', 'Delta_check_WY',
                    'Beq_xi1', 'Beq_xi2', 'Beq_xi3', 'BeqY', 'Beq_receivedY']:
            df_export.loc[df_export['isocode']==country, var] = td[country][var][:Ttrans]
    for country in countries:
        df_export.loc[df_export['isocode'] == country, 'weight'] = td['weights_td'][country][:Ttrans]

    df_export.to_csv(path_results + name + '.csv', index=False)

    # # Export profiles
    # N_ages = len(td['USA']['A_j'])
    # ages = np.arange(0, N_ages+1, 1)
    # df_profiles_export = pd.DataFrame(
    #     np.vstack((
    #         np.repeat(countries, N_ages * Ttrans),
    #         np.tile([np.repeat(Ttrans_years.astype(int), N_ages),
    #                  np.tile(ages, Ttrans)], len(countries))
    #     )), index=['isocode', 'year', 'age_bin']).T
    # for country in countries:
    #     for var in ['W_j', 'A_j', 'Beq_received_j', 'pij', 'h_gross']:
    #         df_export.loc[df_export['isocode'] == country, var] = td[country][var][:Ttrans].flatten()
    #
    # df_profiles_export.to_csv(path_results + name + '_profiles.csv', index=False)