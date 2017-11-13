from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.io
import numpy as np
import requests
import zipfile

from .. import config


def _urlretrieve(url, filename):
    '''Defines a convenience method for downloading files with requests.

    Args:
        url (str): The URL of the file to download.
        filename (str): The path to save the file.
    '''
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def _load_file(fpath):
    '''Checks whether a file is a .mat or .npy file and loads it.

    This is a convenience method for the other loading functions.

    Args:
        fpath (str): The exact path of where data is located.

    Returns:
        mat or numpy array: The loaded dataset.
    '''
    if fpath[-4:] == '.mat':
        data = scipy.io.loadmat(fpath)
    elif fpath[-4:] == '.npy':
        data = np.load(fpath)
    else:
        raise ValueError('Invalid file type: {}'.format(fpath))
    return data


def load_spikefinder_data(dir_name='spikefinder'):
    '''Downloads and returns a dataset of paired calcium recordings.

    This dataset was used for the Spikefinder competition
    (DOI: 10.1101/177956), and consists of datasets of paired calcium traces
    and spike trains collected from multiple sources.

    Args:
        dir_name (str): Specifies the directory to which the data files should
            be downloaded. This is concatenated with the user-set data
            directory.

    Returns:
        tuple: Paths to the downloaded training and testing datasets. Each
        dataset is a CSV which can be loaded using Pandas,
        :data:`pd.read_csv(path)`.

        * :data:`train_data`: List of pairs of strings, where each pair
          consists of the path to the calcium data (inputs) and the path to
          the spike data (ground truth) for that dataset pair.
        * :data:`test_data`: List of strings, where each string is the path
          to a testing dataset.
    '''
    dpath = os.path.join(config.get_data_directory(), dir_name)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    url_template = (
        'https://s3.amazonaws.com/neuro.datasets/'
        'challenges/spikefinder/spikefinder.{version}.zip'
    )

    # Downloads the two datasets.
    def _download(version):
        zipname = os.path.join(dpath, '{}.zip'.format(version))
        if not os.path.exists(zipname):
            url = url_template.format(version=version)
            _urlretrieve(url, zipname)

        # Unzips the associated files.
        unzip_path = os.path.join(dpath, 'spikefinder.{}'.format(version))
        if not os.path.exists(unzip_path):
            zipref = zipfile.ZipFile(zipname, 'r')
            zipref.extractall(dpath)
            zipref.close()
        return unzip_path

    # Downloads the two datasets.
    train_path, test_path = _download('train'), _download('test')
    train_template = os.path.join(train_path, '{index}.train.{mode}.csv')
    test_template = os.path.join(test_path, '{index}.test.calcium.csv')

    # Converts each dataset to a file path.
    train_paths = [(
        train_template.format(index=i, mode='calcium'),
        train_template.format(index=i, mode='spikes'),
    ) for i in range(1, 11)]
    test_paths = [test_template.format(index=i) for i in range(1, 6)]

    # Checks that all of the files exist.
    assert all(os.path.exists(i) and os.path.exists(j) for i, j in train_paths)
    assert all(os.path.exists(i) for i in test_paths)

    return train_paths, test_paths


def load_reward_data(dir_name='reward'):
    '''Downloads and returns the data for the PopVis example.

    Downloads and returns data for Neural Coding Reward Example as well as
    PopVis Example. Dataset comes from `Ramkumar et al's` "Premotor and Motor
    Cortices Encode Reward" paper.

    Args:
        dir_name (str): Specifies the directory to which the data files should
            be downloaded. This is concatenated with the user-set data
            directory.

    Returns:
        tuple: The two downloaded files.

        * :data:`sess_one_mat`: :data:`.mat` file for Monkey M, Session 1.
        * :data:`sess_four_mat`: :data:`.mat` file for Monkey M, Session 4.
    '''
    dpath = os.path.join(config.get_data_directory(), dir_name)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    def download_mat(fname, url):
        '''Helper function for downloading the existing MAT files.'''
        fpath = os.path.join(dpath, fname)
        if not os.path.exists(fname):
            _urlretrieve(url, fpath)
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

    The dataset comes from `UCL's Cortex Lab
    <http://data.cortexlab.net/dualPhase3/data/>`_.

    Args:
        dir_name (str): Specifies the directory to which the data files
            should be downloaded. This is concatenated with the user-set
            data directory.

    Returns:
        dict: A dictionary where each key corresponds to a needed file.
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
            _urlretrieve(url, fname)
        file_dict[name] = _load_file(fname)

    for directory in parent_dir:
        if not os.path.exists(os.path.join(dpath, directory)):
            os.makedirs(os.path.join(dpath, directory))
        for subdir in subdir_fnames:
            fname = os.path.join(dpath, directory, subdir)
            url = os.path.join(base_url, directory, subdir)
            if not os.path.exists(fname):
                _urlretrieve(url, fname)
            key = os.path.join(directory, subdir)
            if subdir == 'cluster_groups.csv':
                file_dict[key] = np.recfromcsv(fname, delimiter='\t')
            else:
                file_dict[key] = _load_file(fname)

    return file_dict


def load_reaching_data(dir_name='reaching'):
    '''Downloads and returns data for the Reaching Dataset example.

    The dataset is publicly available `here <http://goo.gl/eXeUz8>`_. Because
    this is hosted on DropBox, you have to manually visit the link, then
    download it to the appropriate location (usually
    :data:`~/.spykes/reaching/reaching_dataset.h5`).

    Args:
        dir_name (str): Specifies the directory to which the data files
            should be downloaded. This is concatenated with the user-set
            data directory.

    Returns:
        deep dish dataset: The dataset, loaded using :meth:`deepdish.io.load`.
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
        url = 'http://goo.gl/eXeUz8'
        _urlretrieve(url, fpath)

    data = deepdish.io.load(fpath)
    return data


def load_reaching_xy(event='goCueTime', feature='endpointOfReach', neuron='M1',
                     window_min=0., window_max=500., threshold=10.,
                     dir_name='reaching'):
    '''Extracts the reach direction and M1 spikes from the reaching dataset.

    Args:
        event (str): Event to which to align each trial; :data:`goCueTime`,
            :data:`targetOnTime` or :data:`rewardTime`.
        feature (str): The feature to get; :data:`endpointOfReach` or
            :data:`reward`.
        neuron (str): The neuron response to use, either :data:`M1` or
            :data:`PMd`.
        window_min (double): The lower window value around the align queue to
            get spike counts, in milliseconds.
        window_max (double): The upper window value around the align queue to
            get spike counts, in milliseconds.
        threshold (double): The threshold for selecting high-firing neurons,
            representing the minimum firing rate in Hz.
        dir_name (str): Specifies the directory to which the data files
            should be downloaded. This is concatenated with the user-set
            data directory.

    Returns:
        tuple: The :data:`x` and :data:`y` features of the dataset.

        * :data:`x`: Array with shape :data:`(num_samples, num_features)`
        * :data:`y`: Array with shape :data:`(num_samples, num_neurons)`
    '''

    # Loads the formatted data, if it has already been processed.
    fname = '{}.npz'.format('_'.join('{}'.format(i) for i in [
        event, feature, neuron, window_min, window_max, threshold
    ]))
    fpath = os.path.join(config.get_data_directory(), dir_name, fname)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            data = np.load(f)
            return data['x'], data['y']

    # Loads the reaching data normally.
    reaching_data = load_reaching_data(dir_name)

    events = list(reaching_data['events'].keys())
    features = list(reaching_data['features'].keys())

    # Checks the input arguments, throwing helpful error messages if needed.
    if event not in events:
        raise ValueError('Invalid align event: "{}". Must be one of {}.'
                         .format(event, events))
    if feature not in features:
        raise ValueError('Invalid feature: "{}". Must be one of {}.'
                         .format(feature, features))
    if neuron not in ('M1', 'PMd'):
        raise ValueError('Invalid neuron type: "{}". Must be either "M1" or '
                         '"PMd".'.format(neuron))

    neuron_key = 'neurons_{}'.format(neuron)
    spike_times = np.asarray([
        np.squeeze(np.sort(s)) for s in reaching_data[neuron_key]
    ])
    spike_freqs = np.asarray([len(t) / (t[-1] - t[0]) for t in spike_times])

    # Applies the cutoff threshold.
    thresh_idxs = np.where(spike_freqs > threshold)[0]
    spike_times = spike_times[thresh_idxs]
    spike_freqs = spike_freqs[thresh_idxs]

    # Gets the reach angle, in radians.
    x = reaching_data['features'][feature] * np.pi / 180.0
    x = np.arctan2(np.sin(x), np.cos(x))

    # Gets the spike responses.
    event_data = reaching_data['events'][event]

    def _get_spikecounts(n):
        return np.asarray([
            np.sum(np.all((
                n >= e + 1e-3 * window_min,
                n <= e + 1e-3 * window_max,
            ), axis=0))
            for e in event_data
        ])
    y = np.stack([_get_spikecounts(n) for n in spike_times]).transpose(1, 0)

    # Saves the dataset after processing it.
    with open(fpath, 'wb') as f:
        np.savez(f, x=x, y=y)

    return x, y
