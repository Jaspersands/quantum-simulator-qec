# Ported from qsample 0.0.2 (Copyright Don Winter, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: this package's tableau simulator, and no networkx, anytree, simpleeval, scipy or dill.
from __future__ import annotations

__all__ = ['comb', 'binom', 'joint_binom', 'Wilson_var', 'Wald_var', 'subset_cards', 'cartesian_product', 'subset_probs']

import math as _math

import numpy as np
import itertools as it


def comb(n, k):
    """Vectorized combination
    
    .. math:: comb(n,k) = n! / ((n-k)!k!)
    
    Parameters
    ----------
    n : int or np.array of int
        First parameter of combination
    k : int or np.array of int
        Second parameter of combination
        
    Returns
    -------
    np.array
        Combination (choose k out of n); inf where it is beyond a float (from comb(1030, 515))
    """
    # Exact binomial coefficients (qsample divides scipy's floating factorials, which are inf from
    # 171!, so its comb is nan from n = 171).
    out = np.vectorize(_comb, otypes=[float])(np.asarray(n), np.asarray(k))
    return out if out.ndim else out[()]

def _comb(n, k):
    """One exact binomial coefficient as a float, inf where it is beyond a float."""
    if not 0 <= k <= n:
        return 0.0
    n, k = int(n), int(k)
    if _math.lgamma(n + 1) - _math.lgamma(k + 1) - _math.lgamma(n - k + 1) > 712:  # log of the largest float: 709.78
        return _math.inf
    try:
        return float(_math.comb(n, k))
    except OverflowError:
        return _math.inf

def _stirlerr(n):
    """log(n!) - log(sqrt(2 pi n) (n/e)^n), the error of Stirling's formula, for an integer n >= 1
    (Loader's series beyond 15)."""
    if n <= 15:
        return _math.lgamma(n + 1) - (n + 0.5) * _math.log(n) + n - 0.5 * _math.log(2 * _math.pi)
    nn = n * n
    s0, s1, s2, s3, s4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188
    if n > 500:
        return (s0 - s1 / nn) / n
    if n > 80:
        return (s0 - (s1 - s2 / nn) / nn) / n
    if n > 35:
        return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / n
    return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / n

def _bd0(x, m):
    """x log(x/m) + m - x, without its cancellation where x is near m (Loader's series)."""
    if abs(x - m) < 0.1 * (x + m):
        v = (x - m) / (x + m)
        s = (x - m) * v
        ej = 2 * x * v
        v *= v
        j = 1
        while True:
            ej *= v
            s1 = s + ej / (2 * j + 1)
            if s1 == s:
                return s1
            s = s1
            j += 1
    return x * _math.log(x / m) + m - x

def _log_binom(k, n, p):
    """One binomial probability by Loader's saddle-point method (as R's dbinom), accurate to a few
    ulps for any n: for the terms whose coefficient is beyond a float, or whose product falls below
    the normal floats."""
    if not 0 < p < 1:  # then the probability is 0: here k > 0 if p is 0, and k < n if p is 1
        return 0.0
    k, n, p = int(k), int(n), float(p)
    if k == 0:
        return _math.exp(n * _math.log1p(-p))
    if k == n:
        return _math.exp(n * _math.log(p))
    q = 1 - p
    lc = _stirlerr(n) - _stirlerr(k) - _stirlerr(n - k) - _bd0(k, n * p) - _bd0(n - k, n * q)
    lf = _math.log(2 * _math.pi) + _math.log(k) + _math.log1p(-k / n)
    return _math.exp(lc - 0.5 * lf)

def binom(k, n, p):
    """Vectorized binomial distribution
    
    Example
    -------
    >> binom(k=[1,2], n=[3,4], p=0.1)
    [0.243 , 0.0486]
    
    Parameters
    ----------
    n : int or list of int
        First parameter of combination
    k : int or list of int
        Second parameter of combination
    p : float or list of float
        Probability 
    
    Returns
    -------
    np.array
        Value(s) of binomial distribution evaluated at k,n,p.
    """
    k, n, p = np.array(k), np.array(n), np.array(p)
    c, pk, qk = comb(n,k), p**k, (1-p)**(n-k)
    with np.errstate(invalid='ignore'):
        out = c * pk * qk
    # A coefficient beyond a float (n of 1030 and more: a subset sampler's group of 1030 or more
    # fault locations) makes the product inf * 0, and a power below the normal floats loses digits
    # or is 0 (binom(200, 1029, 0.02) is 3.3e-129, not 0). Those terms come from Loader's
    # saddle-point method instead (qsample's are nan from n = 171).
    tiny = np.finfo(float).tiny
    redo = np.isinf(c) | ((c > 0) & ((np.abs(pk) < tiny) | (np.abs(qk) < tiny)))
    if np.any(redo):
        shape = np.shape(out)
        k_, n_, p_, redo = (np.broadcast_to(x, shape).ravel() for x in (k, n, p, redo))
        out = np.array(out, dtype=float).ravel()
        for i in np.flatnonzero(redo):
            out[i] = _log_binom(k_[i], n_[i], p_[i])
        out = out.reshape(shape)
        return out if out.ndim else out[()]
    return out

def joint_binom(k, n, p):
    """Product of independent binomial distributions with
    parameters `k`, `n` and `p` (can be list of lists)
    
    Example
    -------
    >> joint_binom(k=[1,1], n=[2,2], p=[0.5,0.5])
    0.25
    
    Parameters
    ----------
    n : list of int
        List of first parameters of combination
    k : list of int
        List of second parameters of combination
    p : list of list, list of float, or float
        Probability
        
    Returns
    -------
    np.array
        Joint probability
    """
    return np.prod(binom(k,n,p), axis=-1) # In case p is list of list: vector, else scalar

def Wilson_var(p, N):
    r"""Wilson estimator of binomial variance
    
    The formula for the Wilson interval is:
    
    .. math:: CI = p+z^2/(2n) \pm z\sqrt{pq/n + z^2/(4n^2)}/(1 + z^2/n)
    
    we can extract the var (z=1) as:
    
    .. math: Var[p] = (CI/2)^2 = (npq + 0.25) / (1 + n)^2
    
    Parameters
    ----------
    p : float
        Estimator of probability
    N : int
        Sample size
        
    Returns
    -------
    float
        Estimated variance of Wilson CI
    """
    return (N*p*(1-p) + 0.25)  / (N**2 + 2*N + 1)

def Wald_var(p, N):
    """Wald estimation of binomial variance
    
    Parameters
    ----------
    p : float
        Estimator of probability
    N : int
        Sample size
        
    Returns
    -------
    float
        Estimated variance of Wald CI
    """
    return p * (1-p) / N

def subset_cards(superset):
    """Calculate cardinalities of all possible subsets of `superset`
    
    Example
    -------
    >> subset_cards({1,2,3})
    {0,1,2,3}
    
    Parameters
    ----------
    superset : set
        Input set
    
    Returns
    -------
    list of int
        All possible cardinalities of subsets in superset
    """
    return set(range(len(superset) + 1))

def cartesian_product(list_of_sets):
    """Calculate cartesian product between all members of sets
    
    Example
    -------
    >> cartesian_product([{1,2}, {3,4}])
    [(1,3), (1,4), (2,3), (2,4)]
    
    Parameters
    ----------
    list_of_sets : list
        List of sets between which to calculate Cartesian product
        
    Returns
    -------
    list of tuple
        Cartesian products
    """
    return list(it.product(*list_of_sets))

def subset_probs(circuit, error_model, prob):
    """Calculate occurence probability of subsets in `circuit` with physical
    error rate `prob`. `error_model` defines how the circuit is to be 
    partitioned before occurence probabilities are calculated.
    
    Example
    -------
    >> subset_probs(qsample.examples.por, qsample.noise.E1, 0.1)
    {(0,): 0.6561,
     (1,): 0.2916,
     (2,): 0.04860000000000001,
     (3,): 0.0036000000000000008,
     (4,): 0.00010000000000000002}
    
    Parameters
    ----------
    circuit : Circuit
        Circuit wrt. which subset probabilities are calculated
    error_model : ErrorModel
        Error model by which to partition `circuit`
    prob : float or list of float
        Physical error probabilities 
    Returns
    -------
    dict
        keys: subset, values: corresponding probability
    """
    list_of_sets = error_model.group(circuit).values()
    cards = list(map(len, list_of_sets))
    cp_subset_cards = cartesian_product([subset_cards(s) for s in list_of_sets])
    return {cp : joint_binom(cp, cards, prob) for cp in cp_subset_cards}
