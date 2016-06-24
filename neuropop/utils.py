"""
A few miscellaneous helper functions for neuropop.py
"""
import numpy as np
from scipy import stats

def slow_exp_python(z, eta):
    """
    A slowly rising exponential
    that linearizes above threshold parameter eta

    Parameters
    ----------
    z: array
    eta: float, threshold parameter

    Returns
    -------
    qu: array, the resulting slow exponential
    """
    qu = np.zeros(z.shape)
    slope = np.exp(eta)
    intercept = (1 - eta) * slope
    qu[z > eta] = z[z > eta] * slope + intercept
    qu[z <= eta] = np.exp(z[z <= eta])
    return qu

def grad_slow_exp_python(z, eta):
    """
    Gradient of a slowly rising exponential
    that linearizes above threshold parameter eta

    Parameters
    ----------
    z: array
    eta: float, threshold parameter

    Returns
    -------
    dqu_dz: array, the resulting gradient of the slow exponential
    """

    dqu_dz = np.zeros(z.shape)
    slope = np.exp(eta)
    dqu_dz[z > eta] = slope
    dqu_dz[z <= eta] = np.exp(z[z <= eta])
    return dqu_dz

def log_likelihood(y, yhat):
    """Helper to compute the log likelihood."""
    eps = np.spacing(1)
    return np.sum(y * np.log(eps + yhat) - yhat)

def circ_corr(alpha1, alpha2):
    """Helper to compute the circular correlation."""
    alpha1_bar = stats.circmean(alpha1)
    alpha2_bar = stats.circmean(alpha2)
    num = np.sum(np.sin(alpha1 - alpha1_bar) * np.sin(alpha2 - alpha2_bar))
    den = np.sqrt(np.sum(np.sin(alpha1 - alpha1_bar) ** 2) * np.sum(np.sin(alpha2 - alpha2_bar) ** 2))
    rho = num / den
    return rho
