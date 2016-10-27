"""
A few miscellaneous helper functions for spykes
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
    return np.nansum(y * np.log(eps + yhat) - yhat)


def circ_corr(alpha1, alpha2):
    """Helper to compute the circular correlation."""
    alpha1_bar = stats.circmean(alpha1)
    alpha2_bar = stats.circmean(alpha2)
    num = np.sum(np.sin(alpha1 - alpha1_bar) * np.sin(alpha2 - alpha2_bar))
    den = np.sqrt(np.sum(np.sin(alpha1 - alpha1_bar) ** 2) *
                  np.sum(np.sin(alpha2 - alpha2_bar) ** 2))
    rho = num / den
    return rho


def get_sort_indices(data, sortby=None, sortorder='descend'):
    """
    Helper function to calculate sorting indices given sorting condition

    Parameters
    ----------
    data : 2-D numpy array
        n_neurons x n_bins

    sortby: str or list
        None:
        'rate': sort by firing rate
        'latency': sort by peak latency
        list: list of integer indices to be used as sorting indicces


    sortorder: direction to sort in
        'descend'
        'ascend'

    Returns
    -------
    sort_idx : numpy array of sorting indices

    """

    if isinstance(sortby, list):

        if np.array_equal(np.sort(sortby), range(data.shape[0])):
            # make sure it's a permutation
            return sortby

        else:
            raise ValueError(
                "Specified sorting indices not a proper permutation")

    else:

        if sortby == 'rate':
            sort_idx = np.sum(data, axis=1).argsort()

        elif sortby == 'latency':
            sort_idx = np.argmax(data, axis=1).argsort()

        else:
            sort_idx = np.arange(data.shape[0])

        if sortorder == 'ascend':
            return sort_idx
        else:
            return sort_idx[::-1]
