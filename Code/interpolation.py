"""
Fast linear interpolation for two ordered vectors of similar length; also iterates forward on
distribution using linearized rule. uses decorator @jit from numba to speed things up.
"""
import numpy as np
from numba import njit, guvectorize

# Numba's guvectorize decorator compiles functions and allows them to be broadcast by NumPy when dimensions differ.
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """
    Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x: array
        ascending data points
    xq: array
        ascending query points
    y: array
        data points
    yq: array
        empty to be filled with interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """
    Efficient linear interpolation exploiting monotonicity. xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x: array
        ascending data points
    xq: array
        ascending query points
    xq: array
        empty to be filled with indices of lower bracketing gridpoints
    xqpi: array
        empty to be filled with weights on lower bracketing gridpoints

    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi

@njit
def forward_a(D, a_pol_i, a_pol_pi):
    """Same as forward_step, but just the asset part"""
    Dnew = np.zeros((D.shape[0], D.shape[1]))
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            d = D[s, i]
            Dnew[s, apol] += d * api
            Dnew[s, apol + 1] += d * (1 - api)

    return Dnew

@njit
def expectation_a(Xend, a_i, a_pi):
    X = np.zeros_like(Xend)
    for e in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):
            # expectation is pi(e,a)*Xend(e,i(e,a)) + (1-pi(e,a))*Xend(e,i(e,a)+1)
            X[e, a] = a_pi[e, a]*Xend[e, a_i[e, a]] + (1-a_pi[e, a])*Xend[e, a_i[e, a]+1]
    return X

@njit(fastmath=True)
def forward_step(D, Pi_T, a_pol_i, a_pol_pi):
    """
    Single forward step to update distribution using an arbitrary asset policy.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of Lam_{t-1}.

    Parameters
    ----------
    D: np.ndarray
        Beginning-of-period distribution over s_t, a_(t-1)
    Pi_T: np.ndarray
        Transpose Markov matrix that maps s_t to s_(t+1)
    a_pol_i: np.ndarray
        Left gridpoint of asset policy
    a_pol_pi: np.ndarray
        Weight on left gridpoint of asset policy

    Returns
    ----------
    Dnew : np.ndarray
        Beginning-of-next-period dist s_(t+1), a_t

    """
    # first create Dnew from updating asset state
    Dnew = np.zeros((D.shape[0], D.shape[1]))
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            d = D[s, i]
            Dnew[s, apol] += d * api
            Dnew[s, apol + 1] += d * (1 - api)

    # then use transpose Markov matrix to update income state
    Dnew = Pi_T @ Dnew

    return Dnew

@njit
def forward_step_shock_1d(Dss, Pi_T, x_i_ss, x_pi_shock):  # old forward_step_policy_shock()
    """forward_step_1d linearized wrt x_pi"""
    # first find effect of shock to endogenous policy
    nZ, nX = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i_ss[iz, ix]
            dshock = x_pi_shock[iz, ix] * Dss[iz, ix]
            Dshock[iz, i] += dshock
            Dshock[iz, i + 1] -= dshock

    # then apply exogenous transition matrix to update
    return Pi_T @ Dshock

def D_init(a_grid):
    """Asset distribution that is degenerate at zero, only nontrivial if zero not on grid"""
    D = np.zeros(len(a_grid))
    if a_grid[0] == 0:
        D[0] = 1
    else:
        izero = max(np.argmax(a_grid > 0) - 1, 0)   # first index less than zero
        D[izero] = a_grid[izero+1] / (a_grid[izero+1] - a_grid[izero])
        D[izero+1] = 1 - D[izero]
    return D

@njit
def fast_aggregate(X, Y):
    """If X has dims (T, ...) and Y has dims (T, ...), do dot product for each T to get length-T vector.

    Identical to np.sum(X*Y, axis=(1,...,X.ndim-1)) but avoids costly creation of intermediates, useful
    for speeding up aggregation in td by factor of 4 to 5."""
    T = X.shape[0]
    Xnew = X.reshape(T, -1)
    Ynew = Y.reshape(T, -1)
    Z = np.empty(T)
    for t in range(T):
        Z[t] = Xnew[t, :] @ Ynew[t, :]
    return Z
