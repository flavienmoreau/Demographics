"""
Various function to compute inequality statistics.
"""
import numpy as np
from scipy.interpolate import interp1d

def lorenz(x, pr):
    """ Return the lorenz curve given a vector of values and a vector of associated probabilities.

    Parameters
    ----------
    x: np.ndarray
        Vector of values associated with the probabilities in pr
    pr: np.ndarray
        Vector of probabilities associated with the probabilities in x

    Returns
    -------
    tuple
        Returns a tuple containing a vector of percentiles and a vector of associated values percentiles

    """
    if np.ndim(pr) == 1:
        # first do percentiles of the total population
        pctl = np.concatenate(([0], pr.cumsum()-pr/2, [1]))
        # now do percentiles of total wealth
        wealthshare = (x * pr / np.sum(x * pr) if np.sum(x * pr) != 0 else np.zeros_like(x))  # Returns zeros if sum=0
        wealthpctl = np.concatenate(([0], wealthshare.cumsum() - wealthshare / 2, [1]))

    elif np.ndim(pr) == 2:
        # first do percentiles of the total population
        pctl = np.hstack((np.hstack((np.zeros((pr.shape[0], 1)),
                                     pr.cumsum(axis=1) - pr / 2)),
                          np.ones((pr.shape[0], 1))))
        # now do percentiles of total wealth
        xpr_sum = np.sum(x * pr, axis=1)
        discard = xpr_sum == 0
        xpr_sum[discard] = 1  # Ignore entries where sum(x * pr) != 1
        wealthshare = x * pr / (xpr_sum[:, np.newaxis])
        wealthshare[discard] = 0
        # now do percentiles of total wealth
        wealthpctl = np.hstack((np.hstack((np.zeros((wealthshare.shape[0], 1)),
                                           wealthshare.cumsum(axis=1) - wealthshare / 2)),
                                np.ones((wealthshare.shape[0], 1))))

    return pctl, wealthpctl

def topshare(pctl, wealthpctl, top):
    """ Return the top share of desired quantile(s) given a vector of values and a vector of associated probabilities.

    Parameters
    ----------
    pctl: np.ndarray
        Vector of percentiles
    wealthpctl: np.ndarray
        Vector of values associated with the percentiles in pctl
    top: np.ndarray
        Desired quantiles to be returned, can be an integer or an array

    Returns
    -------
    np.ndarray
        Top shares associated with the desired quantile(s)

    """
    if np.ndim(pctl) == 1:
        return 1 - np.interp(1 - top, pctl, wealthpctl)
    if np.ndim(pctl) == 2:
        XX = np.empty((pctl.shape[0], top.shape[0]))
        for i in range(pctl.shape[0]):
            XX[i] = np.interp(1 - top, pctl[i, ], wealthpctl[i, ])
        return 1-XX

def gini(pctl, wealthpctl):
    """ Function used to find the gini_coef coefficient associated with the lorenz curve obtained from
    lorenz(x, pr). Interpolate the given lorenz curve on a grid for percentiles using splines.

    Parameters
    -----
    pctl: np.ndarray
        Vector of percentiles
    wealthpctl: np.ndarray
        Vector of values associated with the percentiles in pctl

    Returns
    -------
    gini_coef: float
        Gini coefficient

    """
    if np.ndim(pctl) == 1:
        # Obtain interpolated relation
        lorenz_x = interp1d(pctl, wealthpctl, kind='slinear')
        # Define percentiles grid
        x = np.linspace(0, 1, 10000)
        # Compute Gini index as area between 45 degree line and lorenz curve
        gini_coef = np.sum(x - lorenz_x(x)) / np.sum(x)

    if np.ndim(pctl) == 2:
        gini_coef = np.empty((pctl.shape[0]))
        for i in range(pctl.shape[0]):
            # Obtain interpolated relation
            lorenz_x = interp1d(pctl[i, ], wealthpctl[i, ], kind='slinear')
            # Define percentiles grid
            x = np.linspace(0, 1, 10000)
            # Compute Gini index as area between 45 degree line and lorenz curve
            gini_coef[i, ] = np.sum(x - lorenz_x(x)) / np.sum(x)

    return gini_coef


def dststat_age(popss, ss, qtls=np.array([0.01, 0.1, 0.2])):
    """
    Distributional statistics overall and by age.

    Parameters
    ----------
    popss: class
        Population class with demographic inputs
    ss: dict
        Dict containing equilibrium objects to get equilibrium policies and distributions
    qtls: list[optional]
        Desired points {x} at which to evaluate top x% of each distribution

    Returns
    -------
     dict
        Dict containing distributional statistics

    """
    # Read inputs
    a = ss['a']
    c = ss['c']
    pij = popss.pij
    Dst = ss['D']
    quantiles = np.array(qtls)

    # reshape multi-dimensional policies
    T, Ntheta, Neps, Na = a.shape
    a_flat = a.reshape(T, 1, Ntheta * Neps * Na).squeeze()
    c_flat = c.reshape(T, 1, Ntheta * Neps * Na).squeeze()

    # flatten out the joint distribution
    Dst_flat = Dst.reshape(T, 1, Ntheta * Neps * Na).squeeze()

    # Statistics overall
    # ------------------

    # Lorenz curves
    a = np.einsum('js,js->s', pij[:, np.newaxis], a_flat)
    c = np.einsum('js,js->s', pij[:, np.newaxis], c_flat)
    p = np.einsum('js,js->s', pij[:, np.newaxis], Dst_flat)
    p = p / np.sum(p)  # Make sure sums to one

    # Sort vectors from lowest to highest
    a_sorted = np.sort(a)
    a_sorted_i = np.argsort(a)
    c_sorted = np.sort(c)
    c_sorted_i = np.argsort(c)
    # Recover associated probabilities
    p_a_sorted = p[a_sorted_i]
    p_c_sorted = p[c_sorted_i]

    # Get Lorenz curves
    lorenz_a_pctl, lorenz_a = lorenz(a_sorted, p_a_sorted)
    lorenz_c_pctl, lorenz_c = lorenz(c_sorted, p_c_sorted)

    # Top shares
    tq_a = topshare(lorenz_a_pctl, lorenz_a, quantiles)
    tq_c = topshare(lorenz_c_pctl, lorenz_c, quantiles)

    # Statistics by age
    # -----------------

    # Sort vectors from lowest to highest
    a_age_sorted = np.sort(a_flat, axis=1)
    a_age_sorted_i = np.argsort(a_flat, axis=1)
    c_age_sorted = np.sort(c_flat, axis=1)
    c_age_sorted_i = np.argsort(c_flat, axis=1)
    # Recover associated probabilities
    p_age = Dst_flat
    p_a_age_sorted = np.array(list(map(lambda x, y: y[x], a_age_sorted_i, p_age)))
    p_c_age_sorted = np.array(list(map(lambda x, y: y[x], c_age_sorted_i, p_age)))

    # Get Lorenz curves by age
    lorenz_a_age_pctl, lorenz_a_age = lorenz(a_age_sorted, p_a_age_sorted)
    lorenz_c_age_pctl, lorenz_c_age = lorenz(c_age_sorted, p_c_age_sorted)

    # Top shares by age
    tq_a_age = topshare(lorenz_a_age_pctl, lorenz_a_age, quantiles)
    tq_c_age = topshare(lorenz_c_age_pctl, lorenz_c_age, quantiles)

    return {
        'a_sorted': a_sorted, 'p_a_sorted': p_a_sorted,
        # Overall Lorenz curves and associates percentiles
        'lorenz_a': lorenz_a, 'lorenz_a_pctl': lorenz_a_pctl,
        'lorenz_c': lorenz_c, 'lorenz_c_pctl': lorenz_c_pctl,
        # Lorenz curves by age and associated percentiles
        'lorenz_a_age': lorenz_a_age, 'lorenz_a_age_pctl': lorenz_a_age_pctl,
        'lorenz_c_age': lorenz_c_age, 'lorenz_c_age_pctl': lorenz_c_age_pctl,
        # Top shares overall
        'quantiles': quantiles,
        'tq_a': tq_a, 'tq_c': tq_c,
        # Top shares by age
        'tq_a_age': tq_a_age, 'tq_c_age': tq_c_age,
    }


def td_within_between(a, pij, Dst, x):
    """Compute within/between decomposition of log wealth."""
    Ttrans, T, Ntheta, Neps, Na = a.shape

    # w_tji :  weights by time, group, observation
    w_tji = pij[:,:,np.newaxis] * Dst.reshape(Ttrans, T, 1, Ntheta * Neps * Na).squeeze()
    w_tji /= np.sum(w_tji, axis=(1,2))[:,np.newaxis,np.newaxis]

    # x_tji :  value by time, group, observation
    x_tji_level = (a.copy().reshape(Ttrans, T, 1, Ntheta * Neps * Na).squeeze())
    x_tji = np.log(x_tji_level)

    # Truncate bottom observations (less than x% of average net worth in level)
    w_tji[x_tji_level < x * np.einsum('tji,tji->t', x_tji_level, w_tji)[:,
                            np.newaxis, np.newaxis]] = 0.  # Weight of zero for those observations
    x_tji[x_tji_level < x * np.einsum('tji,tji->t', x_tji_level, w_tji)[:,
                            np.newaxis, np.newaxis]] = 0.  # zero for those observations

    # w_tj :  weights by time, group
    w_tj = np.einsum('tji->tj', w_tji)

    # x_tj :  average value by time, group
    x_tj = np.einsum('tj,tji,tji->tj', 1 / w_tj, w_tji, x_tji)

    # x_t :  average value by time
    x_t = np.einsum('tj,tj->t', w_tj, x_tj)

    # overall variance
    var_a = np.einsum('tji,tji->t', w_tji,  (x_tji - x_t[:,np.newaxis,np.newaxis]) ** 2)
    std_a = var_a ** (1 / 2)

    # variance by age
    var_a_age = np.einsum('tj, tji,tji->tj', 1 / w_tj, w_tji, (x_tji - x_tj[:, :, np.newaxis]) ** 2)
    std_a_age = var_a_age ** (1 / 2)

    # within variance
    var_a_within = np.einsum('tj,tj->t', w_tj, var_a_age)
    std_a_within = var_a_within ** (1 / 2)

    # between variance
    var_a_between = np.einsum('tj,tj->t', w_tj, (x_tj - x_t[:,np.newaxis]) ** 2)
    std_a_between = var_a_between ** (1 / 2)

    # Make sure correctly constructed
    assert np.allclose(var_a,  var_a_within + var_a_between)

    return {
        # Standard deviation and variance of log-income overall
        'std_a': std_a, 'var_a': var_a,
        # Standard deviation and variance of log-income by age
        'std_a_age': std_a_age, 'var_a_age': var_a_age,
        # Between-within decomposition of variance of log-wealth
        'var_a_between': var_a_between, 'var_a_within': var_a_within,
        'std_a_between': std_a_between, 'std_a_within': std_a_within,
        'var_a_between_pct': var_a_between/var_a, 'var_a_within_pct': var_a_within/var_a,
    }

