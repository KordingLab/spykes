"""
Functions that allow examples to fetch data from online resources
"""

import os
import urllib
import scipy.io
import numpy as np
import deepdish as dd


def load_reward_data(dpath='spykes_data/reward/'):

    """
    Downloads and returns data for Neural Coding Reward Example as well as
    PopVis Example. Dataset comes from Ramkumar et al's "Premotor and Motor
    Cortices Encode Reward" paper located at
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0160851

    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded

    Returns
    -------
    sess_one_mat: .mat file
        Monkey M, Session 1
    sess_four_mat: .mat file
        Monkey M, Session 4
    """

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    fname = os.path.join(dpath, 'Mihili_07112013.mat')

    if not os.path.exists(fname):

        urllib.urlretrieve('https://ndownloader.figshare.com/files/5652051',
                           fname)

    sess_one_mat = _load_file(fname)

    fname = os.path.join(dpath, 'Mihili_08062013.mat')

    if not os.path.exists(fname):

        urllib.urlretrieve('https://ndownloader.figshare.com/files/5652060',
                           fname)

    sess_four_mat = _load_file(fname)

    return sess_one_mat, sess_four_mat


def load_neuropixels_data(dpath='spykes_data/neuropixels/'):

    """
    Downloads and returns data for Neuropixels Example. Dataset comes from
    UCL's Cortex Lab, which is located at
    http://data.cortexlab.net/dualPhase3/data/

    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded

    Returns
    -------
    data_dict
        dictionary, where every key corresponds to a needed file
    """

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    base_url = 'http://data.cortexlab.net/dualPhase3/data/'

    file_dict = dict()

    parent_fnames = ['experiment1stimInfo.mat', 'experiment2stimInfo.mat',
                     'experiment3stimInfo.mat', 'timeCorrection.mat',
                     'timeCorrection.npy']

    parent_dir = ['frontal/', 'posterior/']

    subdir_fnames = ['spike_clusters.npy', 'spike_templates.npy',
                     'spike_times.npy', 'templates.npy',
                     'whitening_mat_inv.npy', 'cluster_groups.csv',
                     'channel_positions.npy']

    for name in parent_fnames:
        fname = os.path.join(dpath, name)
        url = os.path.join(base_url, name)
        if not os.path.exists(fname):
            urllib.urlretrieve(url, fname)
        file_dict[name] = _load_file(fname)

    for directory in parent_dir:

        if not os.path.exists(os.path.join(dpath, directory)):
            os.makedirs(os.path.join(dpath, directory))

        for subdir in subdir_fnames:
            fname = os.path.join(dpath, directory, subdir)
            url = os.path.join(base_url, directory, subdir)

            if not os.path.exists(fname):
                urllib.urlretrieve(url, fname)

            key = os.path.join(directory, subdir)

            if subdir == 'cluster_groups.csv':
                file_dict[key] = np.recfromcsv(fname, delimiter='\t')
            else:
                file_dict[key] = _load_file(fname)

    return file_dict


def load_reaching_data(dpath='spykes_data/reaching/'):

    """
    Downloads and returns data for Reaching Dataset Example. Dataset is
    publicly available at
    https://northwestern.app.box.com/s/xbe3xpnv6gpx0c1mrfhb1bal4cyei5n8

    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded

    Returns
    -------
    deep dish loaded dataset
    """

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    url = 'https://northwestern.app.box.com/index.php?rm=box_download_shared_file&shared_name=xbe3xpnv6gpx0c1mrfhb1bal4cyei5n8&file_id=f_71457089609' # noqa

    fname = os.path.join(dpath, 'reaching_dataset.h5')

    if not os.path.exists(fname):
        urllib.urlretrieve(url, fname)

    return dd.io.load(fname)


def _load_file(fname):

    """
    Helper function to check whether a file is a .mat or .npy file and then
    load it

    Parameters
    ----------
    fname: str
        specifies exact path of where data will be downloaded to

    Returns
    -------
    mat or numpy loaded dataset
    """

    if fname[-4:] == '.mat':
        return scipy.io.loadmat(fname)

    elif fname[-4:] == '.npy':
        return np.load(fname)
