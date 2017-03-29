import os
import urllib
import scipy.io
import numpy as np
import deepdish as dd


def load_reward_data(dpath='spykes_data/reward/'):

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

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    url = 'https://northwestern.app.box.com/index.php?rm=box_download_shared_file&shared_name=xbe3xpnv6gpx0c1mrfhb1bal4cyei5n8&file_id=f_71457089609' # noqa

    fname = os.path.join(dpath, 'reaching_dataset.h5')

    if not os.path.exists(fname):
        urllib.urlretrieve(url, fname)

    return dd.io.load(fname)


def _load_file(fname):

    if fname[-4:] == '.mat':
        return scipy.io.loadmat(fname)

    elif fname[-4:] == '.npy':
        return np.load(fname)
