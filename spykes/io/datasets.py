'''Functions that allow examples to fetch data from online resources.'''

import os
import urllib
import scipy.io
import numpy as np

from .. import config


def _load_file(fpath):
    '''Checks whether a file is a .mat or .npy file and loads it.
    Args:
        fpath: str, the exact path of where data will be downloaded to
    Returns:
        mat or numpy loaded dataset
    '''
    if fpath[-4:] == '.mat':
        return scipy.io.loadmat(fpath)
    elif fpath[-4:] == '.npy':
        return np.load(fpath)
    else:
        raise ValueError('Invalid file type: {}'.format(fpath))


def load_reward_data(dir_name='reward'):
    '''Downloads and returns the data for the PopVis example.
    Downloads and returns data for Neural Coding Reward Example as well as
    PopVis Example. Dataset comes from Ramkumar et al's "Premotor and Motor
    Cortices Encode Reward" paper located at
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0160851
    Args:
        dpath: str, specifies path to which the data files should be
            downloaded. This is concatenated with the user-set data directory.
    Returns:
        sess_one_mat: .mat file, Monkey M, Session 1
        sess_four_mat: .mat file, Monkey M, Session 4
    '''
    dpath = os.path.join(config.get_data_directory(), dir_name)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    def download_mat(fname, url):
        '''Helper function for downloading the existing MAT files.'''
        fpath = os.path.join(dpath, fname)
        if not os.path.exists(fname):
            urllib.urlretrieve(url, fpath)
        return _load_file(fpath)

    # Downloads sess_one_mat.
    sess_one_mat = download_mat(
        fname='Mihili_07112013.mat',
        url='https://ndownloader.figshare.com/files/5652051',
    )

    # Downloads sess_four_mat.
    sess_four_mat = download_mat(
        fname='Mihili_08062013.mat',
        url='https://ndownloader.figshare.com/files/5652060',
    )

    return sess_one_mat, sess_four_mat


def load_neuropixels_data(dir_name='neuropixels'):
    '''Downloads and returns data for the Neuropixels example.
    The dataset comes from UCL's Cortex Lab, which is located at:
        http://data.cortexlab.net/dualPhase3/data/
    Args:
        dir_name: str, the name of the directory where the data is located.
            This is concatenated with the user-set data directory.
    Returns:
        a dict, where every key corresponds to a needed file
    '''
    dpath = os.path.join(config.get_data_directory(), dir_name)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    base_url = 'http://data.cortexlab.net/dualPhase3/data/'
    file_dict = dict()

    parent_fnames = [
        'experiment1stimInfo.mat',
        'experiment2stimInfo.mat',
        'experiment3stimInfo.mat',
        'timeCorrection.mat',
        'timeCorrection.npy',
    ]
    parent_dir = [
        'frontal/',
        'posterior/',
    ]
    subdir_fnames = [
        'spike_clusters.npy',
        'spike_templates.npy',
        'spike_times.npy',
        'templates.npy',
        'whitening_mat_inv.npy',
        'cluster_groups.csv',
        'channel_positions.npy',
    ]

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


def load_reaching_data(dir_name='reaching'):
    '''Downloads and returns data for the Reaching Dataset example.
    The dataset is publicly available at:
        https://northwestern.app.box.com/s/xbe3xpnv6gpx0c1mrfhb1bal4cyei5n8
    Args:
        dir_name: str, the name of the directory where the data is located.
            This is concatenated with the user-set data directory.
    Returns:
        deep dish loaded dataset
    '''
    # Import is performed here so that deepdish is not required for all of
    # the "datasets" functions.
    import deepdish

    dpath = os.path.join(config.get_data_directory(), dir_name)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    # Downloads the file if it doesn't exist already.
    fpath = os.path.join(dpath, 'reaching_dataset.h5')
    if not os.path.exists(fpath):
        # Hosted on Dropbox, so it can't be downloaded propertly with urllib.
        url = 'http://goo.gl/eXeUz8'
        raise RuntimeError('Reaching dataset not found: You need to download '
                           'it to {fpath} from {url}'
                           .format(fpath=fpath, url=url))

    return deepdish.io.load(fpath)
