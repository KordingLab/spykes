"""
A few miscellaneous helper functions for neuropop.py
"""
import numpy as np

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
