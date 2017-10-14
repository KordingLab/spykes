"""
A few miscellaneous helper functions for spykes
"""
import numpy as np
from scipy import stats
import matplotlib # noqa


def train_test_split(*datasets, **split):
    '''Splits test data into training and testing data.
    This is a replacement for the Scikit Learn version of the function.
    Args:
        datasets: list of Numpy arrays, where the first dimension is the batch
            dimension.
        n: int, number of test samples to split off (only split_num
            or split_prct may be specified).
        percent: int, percentange of test samples to split off.
    Returns:
        list of pairs of Numpy arrays, the split test data.
    '''
    if not datasets:
        return []  # Guarentee there's at least one dataset.
    num_batches = int(datasets[0].shape[0])

    # Checks the input shapes.
    if not all(d.shape[0] == num_batches for d in datasets):
        raise ValueError('Not all of the datasets have the same batch size. '
                         'Received batch sizes: {batch_sizes}'
                         .format(batch_sizes=[d.shape[0] for d in datasets]))

    # Gets the split num or split percent.
    split_num = split.get('n', None)
    split_prct = split.get('percent', None)

    # Checks the splits
    if (split_num and split_prct) or not (split_num or split_prct):
        raise ValueError('Must specify either `split_num` or `split_prct`')

    # Splits all of the datasets.
    if split_prct is None:
        num_test = split_num
    else:
        num_test = int(num_batches * split_prct)

    # Checks that the test number is less than the number of batches.
    if num_test >= num_batches:
        raise ValueError('Invalid split number: {num_test} There are only '
                         '{num_batches} samples.'
                         .format(num_test=num_test, num_batches=num_batches))

    # Splits each of the datasets.
    idxs = np.arange(num_batches)
    np.random.shuffle(idxs)
    train_idxs, test_idxs = idxs[num_test:], idxs[:num_test]
    datasets = [(d[train_idxs], d[test_idxs]) for d in datasets]
    return datasets if len(datasets) > 1 else datasets[0]


def slow_exp(z, eta):
    """
    A slowly rising exponential
    that linearizes above threshold parameter eta

    Parameters
    ----------
    z: array
        ????
    eta: float, threshold parameter
        ???

    Returns
    -------
    qu: array
        The resulting slow exponential
    """
    qu = np.zeros(z.shape)
    slope = np.exp(eta)
    intercept = (1 - eta) * slope
    qu[z > eta] = z[z > eta] * slope + intercept
    qu[z <= eta] = np.exp(z[z <= eta])
    return qu


def grad_slow_exp(z, eta):
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

        if np.array_equal(np.sort(sortby), list(range(data.shape[0]))):
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


def set_matplotlib_defaults(plt):

    """
    Set publication quality defaults for matplotlib.
    Parameters
    ----------
    plt : instance of matplotlib.pyplot
        The plt instance.
    """

    params = {'font.family': 'sans-serif',
              'font.sans-serif': 'Bitsream Vera Sans',
              'font.size': 13,
              'axes.titlesize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'xtick.major.size': 6,
              'ytick.major.size': 6,
              'legend.fontsize': 11}

    plt.rcParams.update(params)
