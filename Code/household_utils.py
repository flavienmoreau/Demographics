"""
Various useful function to construct grids for state variables, transition matrices of exogenous
Markov state variable and associated stationary distribution, discretize AR(1) income processes
"""
import numpy as np
from numba import njit
from scipy.stats import norm

from Code.utils import var, within_tolerance, path_tables
from Code.interpolation import interpolate_coord, forward_step
from Code.government import Government


@njit
def aggregate(X, Y):
    # aggregate X_ij and Y_ij to get Z_i
    Z = np.empty(X.shape[0])
    for i in range(Z.shape[0]):
        Z[i] = np.vdot(X[i, :], Y[i, :])
    return Z


def agrid(amax, N, amin=0):
    """
    Grid a+pivot evenly log-spaced between amin+pivot and amax+pivot
    """
    pivot = np.abs(amin) + 0.5 # changed to 0.5
    a = np.geomspace(amin + pivot, amax + pivot, N) - pivot
    a[0] = amin  # make sure *exactly* equal to amin

    return a

def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """
    Find invariant distribution of a Markov chain by iteration
    """
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if within_tolerance(pi_new, pi, tol):
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi

def invdist(a, a_pol, Pi, D=None, atol=1E-10, maxit=10000):
    """
    Finds invariant distribution given s*a policy array a_pol for endogenous state a and Markov transition matrix Pi
    for exogenous state s, possibly with starting distribution a*s D as a seed
    """
    pi = stationary(Pi)  # compute separately exogenous inv dist to start there
    if D is None:
        D = pi[:, np.newaxis] * np.ones_like(a) / a.shape[0]  # assume equispaced on grid

    a_pol_i, a_pol_pi = interpolate_coord(a, a_pol)  # obtain policy rule

    # now iterate until convergence according to atol, only checking every 10 it
    for it in range(maxit):
        Dnew = forward_step(D, Pi, a_pol_i, a_pol_pi)
        if it % 10 == 0 and within_tolerance(Dnew, D, atol):
            break
        D = Dnew

    return Dnew

def markov_tauchen(rho, sigma, N=11, m=3):
    """
    Implements Tauchen method to approximate AR(1) with persistence rho and normal innovations with sd sigma, with N
    discrete states, within -m and m times its stationary sd
    """
    sigma_y = sigma * np.sqrt(1 / (1 - rho ** 2))
    s = np.linspace(-m * sigma_y, m * sigma_y, N)
    ds = s[1] - s[0]

    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sigma)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sigma)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sigma)
                    - norm.cdf(s[j] - rho * s - ds / 2, scale=sigma))

    pi = stationary(Pi)
    return s, pi, Pi

def markov_rouwenhorst(rho, sigma, N=7):
    """
    Rouwenhorst method analog to markov_tauchen
    """
    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(var(s, pi)))

    return s, pi, Pi

def markov_incomes(rho, sigma_y, N=11):
    """
    Simple helper method that assumes AR(1) process in logs for incomes and scales aggregate income
    to 1, also that takes in sdy as the *cross-sectional* sd of log incomes
    """
    if N == 1:
        # special case with no risk
        y = np.array([1.])
        pi = np.array([1.])
        Pi = np.array([[1.]])
        return y, pi, Pi
    else:
        sigma = sigma_y * np.sqrt(1 - rho ** 2)
        s, pi, Pi = markov_tauchen(rho, sigma, N)
        y = np.exp(s) / np.sum(pi * np.exp(s))
        return y, pi, Pi

def markov_tauchen_nonstationary(rho_t, sigma_t, N=11, m=3):
    """
    Implements Tauchen method to approximate a non-stationary AR(1) with persistence
    rho(t) and normal innovations with standard deviation sigma(t), with N discrete
    states, within -m and m times the standard deviation
    """
    T = len(rho_t)
    sigma_y = np.empty_like(sigma_t)
    Pi_t = np.empty((T, N, N))

    # Step 1: construct the state space s(t) in each period t.
    # Evenly-spaced N-state space over [-m(t)*sigma_y(t),m(t)*sigma_y(t)]

    # 1.a Compute unconditional variances of y(t)
    sigma_y[0] = sigma_t[0]
    for i in range(1, T):
        sigma_y[i] = np.sqrt(rho_t[i] ** 2 * sigma_y[i - 1] ** 2 + sigma_t[i] ** 2)

    # 1.b Construct state space
    h = 2 * m * sigma_y / (N - 1)  # grid step
    s_t = np.tile(h, (N, 1))
    s_t[0, :] = - m * sigma_y
    s_t = np.cumsum(s_t, 0).T

    # Step 2: construct the transition matricies Pi(:,:,t) from period (t-1) to period t
    # Compute the transition matrix for t=1, ie from y(0)=0 to any gridpoint of s_t(1) in period 1
    # rows are the (unconditional) distribution in period 1
    cdf = np.empty((N, N))
    for i in range(0, N):
        temp1d = (s_t[0, :] - h[0]/2) / sigma_t[0]
        temp1d = np.maximum(temp1d, -37)  # To avoid underflow in next line
        cdf[:, i] = norm.cdf(temp1d)

    Pi_t[0, :, 0] = cdf[1, :]
    Pi_t[0, :, N-1] = 1 - cdf[N-1, :]

    for j in range(1, N-1):
        Pi_t[0, :, j] = cdf[j+1, :] - cdf[j, :]

    # Compute the transition matrices for t>2
    temp3d = np.empty((T, N, N))
    for t in range(1, T):
        for i in range(0, N):
            temp3d[t, :, i] = (s_t[t, :] - rho_t[t]*s_t[t-1, i] - h[t] / 2) / sigma_t[t]
            temp3d[t, :, i] = np.maximum(temp3d[t, :, i], -37)  # To avoid underflow in next line
            cdf[:, i] = norm.cdf(temp3d[t, :, i])

        Pi_t[t, :, 0] = cdf[1, :]
        Pi_t[t, :, N - 1] = 1 - cdf[N - 1, :]

        for j in range(1, N - 1):
            Pi_t[t, :, j] = cdf[j + 1, :] - cdf[j, :]

    return s_t, Pi_t

def markov_rouwenhorst_nonstationary(rho_t, sigma_t, N=11):
    """
    Implements Rouwenhorst method to approximate non-stationary AR(1) with persistence
    rho(t) and innovations with standard deviations sigma(t), with N discrete states
    """
    T = len(rho_t)
    sigma_y = np.empty_like(sigma_t)
    Pi_t = np.empty((T, N, N))

    # Step 1: construct the state space s(t) in each period t.
    # Evenly-spaced N-state space over [-sqrt(N-1)*sigma(t),sqrt(N-1)*sigma(t)]

    # 1.a Compute unconditional variances of y(t)
    sigma_y[0] = sigma_t[0]
    for i in range(1, T):
        sigma_y[i] = np.sqrt(rho_t[i] ** 2 * sigma_y[i - 1] ** 2 + sigma_t[i] ** 2)

    # 1.b Construct state space
    h = 2 * np.sqrt(N - 1) * sigma_y / (N - 1)  # grid step
    s_t = np.tile(h, (N, 1))
    s_t[0, :] = - np.sqrt(N - 1) * sigma_y
    s_t = np.cumsum(s_t, 0).T

    # Step 2: construct the transition matricies Pi(:,:,t) from period (t-1) to period t
    # The transition amtrix for period t is defined by the parameter p(t).
    # p(t) = 0.2*(1+rho*sigma(t-1)/sigma(t))

    # Note: Pi(:,:,0) is the transition matrix form y(0)=0 to any grid point of the grid
    # s(1) in period 1. Any of its rows i the unconditional distribution in period 1.

    p = 1 / 2  # first period: p=0.5 as y(1) is white noise
    Pi_t[0, :, :] = rhmat(p, N)

    for j in range(1, T):
        p = (sigma_y[j] + rho_t[j] * sigma_y[j - 1]) / (2 * sigma_y[j])
        Pi_t[j, :, :] = rhmat(p, N)

    return s_t, Pi_t

def rhmat(p, N):
    """
    Computes Rouwenhorst matrix as a function of p and N
    """
    Pmat = np.zeros((N, N))

    # Get the transition matrix P1 for the N=2 case and apply recursion
    if N == 2:
        Pmat = np.array([[p, 1 - p], [1 - p, p]])
    else:
        P1 = np.array([[p, 1 - p], [1 - p, p]])
        for i in range(1, N - 1):
            P2 = p * np.vstack((np.hstack((P1, np.zeros((P1.shape[0], 1)))),
                                np.append(np.zeros((1, P1.shape[1])), 0))) + \
                 (1 - p) * np.vstack((np.hstack((np.zeros((P1.shape[0], 1)), P1)),
                                      np.append(np.zeros((1, P1.shape[1])), 0))) + \
                 (1 - p) * np.hstack((np.vstack((np.zeros((1, P1.shape[0])), P1)),
                                      np.append(np.zeros((P1.shape[1], 1)), [[0]], axis=0))) + \
                 p * np.hstack((np.append(np.zeros((P1.shape[1], 1)), [[0]], axis=0),
                                np.vstack((np.zeros((1, P1.shape[0])), P1))))
            P2[1:i + 1, :] = 0.5 * P2[1:i + 1, :]

            if i == N - 2:
                Pmat = P2
            else:
                P1 = P2

    return Pmat

def markov_incomes_rouwenhorst(rho, sigma_y, N=11):
    """
    Analog to markov_incomes using Rouwenhorst method
    """
    sigma = sigma_y * np.sqrt(1 - rho ** 2)
    s, pi, Pi = markov_tauchen(rho, sigma, N)
    y = np.exp(s) / np.sum(pi * np.exp(s))
    return y, pi, Pi
