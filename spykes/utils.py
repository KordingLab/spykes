from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import matplotlib.pyplot


def train_test_split(*datasets, **split):
    '''Splits test data into training and testing data.

    This is a replacement for the Scikit Learn version of the function (which
    is being deprecated).

    Args:
        datasets (list of Numpy arrays): The datasets as Numpy arrays, where
            the first dimension is the batch dimension.
        n (int): Number of test samples to split off (only `n` or `percent`
            may be specified).
        percent (int): Percentange of test samples to split off.

    Returns:
        tuple of train / test data, or list of tuples: If only one dataset is
        provided, this method returns a tuple of training and testing data;
        otherwise, it returns a list of such tuples.
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
    '''Applies a slowly rising exponential function to some data.

    This function defines a slowly rising exponential that linearizes above
    the threshold parameter :data:`eta`. Mathematically, this is defined as:

    .. math::

        q = \\begin{cases}
            (z + 1 - eta) * \\exp(eta)  & \\text{if } z > eta \\\\
            \\exp(eta)                  & \\text{if } z \\leq eta
        \\end{cases}

    The gradient of this function is defined in :meth:`grad_slow_exp`.

    Args:
        z (array): The data to apply the :func:`slow_exp` function to.
        eta (float): The threshold parameter.

    Returns:
        array: The resulting slow exponential, with the same shape as
        :data:`z`.
    '''
    qu = np.zeros(z.shape)
    slope = np.exp(eta)
    intercept = (1 - eta) * slope
    qu[z > eta] = z[z > eta] * slope + intercept
    qu[z <= eta] = np.exp(z[z <= eta])
    return qu


def grad_slow_exp(z, eta):
    '''Computes the gradient of a slowly rising exponential function.

    This is defined as:

    .. math::

        \\nabla q = \\begin{cases}
            \\exp(eta)  & \\text{if } z > eta \\\\
            \\exp(z)    & \\text{if } z \\leq eta
        \\end{cases}

    Args:
        z (array): The dependent variable, before calling the :func:`slow_exp`
            function.
        eta (float): The threshold parameter used in the original
            :func:`slow_exp` call.

    Returns:
        array: The gradient with respect to :data:`z` of the output of
        :func:`slow_exp`.
    '''
    dqu_dz = np.zeros(z.shape)
    slope = np.exp(eta)
    dqu_dz[z > eta] = slope
    dqu_dz[z <= eta] = np.exp(z[z <= eta])
    return dqu_dz


def log_likelihood(y, yhat):
    '''Helper function to compute the log likelihood.'''
    eps = np.spacing(1)
    return np.nansum(y * np.log(eps + yhat) - yhat)


def circ_corr(alpha1, alpha2):
    '''Helper function to compute the circular correlation.'''
    alpha1_bar = stats.circmean(alpha1)
    alpha2_bar = stats.circmean(alpha2)
    num = np.sum(np.sin(alpha1 - alpha1_bar) * np.sin(alpha2 - alpha2_bar))
    den = np.sqrt(np.sum(np.sin(alpha1 - alpha1_bar) ** 2) *
                  np.sum(np.sin(alpha2 - alpha2_bar) ** 2))
    rho = num / den
    return rho


def get_sort_indices(data, by=None, order='descend'):
    '''Helper function to calculate sorting indices given sorting condition.

    Args:
        data (2-D numpy array): Array with shape :data:`(n_neurons, n_bins)`.
        by (str or list): If :data:`rate`, sort by firing rate. If
            :data:`latency`, sort by peak latency. If a list or array is
            provided, it must correspond to integer indices to be used as
            sorting indices. If no sort order is provided, the data is
            returned as-is.
        order (str): Direction to sort in (either :data:`descend` or
            :data:`ascend`).

    Returns:
        list: The sort indices as a Numpy array, with one index per element in
        :data:`data` (i.e. :data:`data[sort_idxs]` gives the sorted data).
    '''
    # Checks if the by indices are a list or array.
    if isinstance(by, list):
        by = np.array(by)
    if isinstance(by, np.ndarray):
        if np.array_equal(np.sort(by), list(range(data.shape[0]))):
            return by  # Returns if it is a proper permutation.
        else:
            raise ValueError('The sorting indices not a proper permutation: {}'
                             .format(by))

    # Converts the by array to
    if by == 'rate':
        sort_idx = np.sum(data, axis=1).argsort()
    elif by == 'latency':
        sort_idx = np.argmax(data, axis=1).argsort()
    elif by is None:
        sort_idx = np.arange(data.shape[0])
    else:
        raise ValueError('Invalid sort preference: "{}". Must be "rate", '
                         '"latency" or None.'.format(by))

    # Checks the sorting order.
    if order == 'ascend':
        return sort_idx
    elif order == 'descend':
        return sort_idx[::-1]
    else:
        raise ValueError('Invalid sort order: {}'.format(order))


def set_matplotlib_defaults(plt=None):
    '''Sets publication quality defaults for matplotlib.

    Args:
        plt (matplotlib.pyplot instance): The plt instance.
    '''
    if plt is None:
        plt = matplotlib.pyplot
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Bitsream Vera Sans',
        'font.size': 13,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'legend.fontsize': 11,
    })
