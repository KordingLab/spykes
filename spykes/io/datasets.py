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


def _arg_check(name, arg, valid_args):
    '''Convenience function for doing argument cleaning and checking.'''

    # Makes sure that the argument is valid.
    if arg not in valid_args:
        valid_args = list(valid_args)
        formatted_args = ', '.join('"{}"'.format(i) for i in valid_args[:-1])
        formatted_args += ' or "{}"'.format(valid_args[-1])
        raise ValueError('Invalid {}: "{}". Expected {}.'
                         .format(name, arg, formatted_args))

    return arg


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


def load_neuropixels_times(location, cutoff=0.3, dir_name='neuropixels'):
    '''Extracts the neuropixel spike deltas.

    This code is adopted from the Cortex Lab's Matlab implementation. This
    method provides a simpler interface for loading that data by location.

    Args:
        location (str): One of :data:`striatum`, :data:`motor_ctx`,
            :data:`thalamus`, :data:`hippocampus`, or :data:`visual_ctx`.
        cutoff (double): The cutoff threshold for spike templates.
        dir_name (str): Specifies the directory to which the data files
            should be downloaded. This is concatenated with the user-set
            data directory.

    Returns:
        list of arrays: each element contains the spike times for one cluster.
    '''

    # Cleans and validates arguments.
    location = location.lower().replace('cortex', 'ctx')
    _arg_check('location', location, ('striatum', 'motor_ctx', 'thalamus',
                                      'hippocampus', 'visual_ctx'))

    fname = 'processed_{}_{}.npy'.format(location, cutoff)
    fpath = os.path.join(config.get_data_directory(), dir_name, fname)

    # Loads a cached version, if one exists.
    if os.path.exists(fpath):
        return np.load(fpath)

    # Parses the mode from the location.
    mode = 'frontal' if location in ('striatum', 'motor_ctx') else 'posterior'

    # Initializes the recording frequency.
    frequency = 30000.0

    # Loads the data normally.
    data_dict = load_neuropixels_data(dir_name=dir_name)

    def _load_key(name, ext='npy', squeeze=True):
        key = '{}/{}.{}'.format(mode, name, ext)
        return np.squeeze(data_dict[key]) if squeeze else data_dict[key]

    # Loads data that is common to any of the analysis.
    clusters = _load_key('spike_clusters')  # Number of clusters
    spike_times = _load_key('spike_times') / frequency
    spike_templates = _load_key('spike_templates')
    templates = _load_key('templates')
    winv = _load_key('whitening_mat_inv')
    y_coords = _load_key('channel_positions')[:, 1]

    # Performs time correction on the spike times if needed.
    if mode == 'frontal':
        time_correction = data_dict['timeCorrection.npy']
        spike_times = spike_times * time_correction[0] + time_correction[1]

    data = _load_key('cluster_groups', ext='csv', squeeze=False)

    # Gets indices.
    cids = np.array([x[0] for x in data])
    cfg = np.array([x[1] for x in data])
    cids, cfgs = (np.asarray(i) for i in zip(*data))
    good_indices = np.in1d(clusters, cids[cfg == b'good'])

    # Orders spikes by how many clusters they are in.
    real_clusters = clusters[good_indices]
    sort_idx = np.argsort(real_clusters)
    sorted_spikes = spike_times[good_indices][sort_idx]
    sorted_spike_templates = spike_templates[good_indices][sort_idx]

    # Gets the counts per cluster.
    counts_per_cluster = np.bincount(real_clusters)

    # Computes the depth for each spike.
    templates_unw = np.array([np.dot(t, winv) for t in templates])
    template_amps = np.ptp(templates_unw, axis=1)
    template_thresholds = np.max(template_amps, axis=1, keepdims=True) * cutoff
    template_amps[template_amps < template_thresholds] = 0
    amp_sums = np.sum(template_amps, axis=1, keepdims=True)
    template_depths = ((y_coords * template_amps) / amp_sums).sum(axis=1)
    sorted_spike_depths = template_depths[sorted_spike_templates]

    # Splits by cluster and computes the average cluster depth.
    split_idxs = np.cumsum(counts_per_cluster[counts_per_cluster != 0])[:-1]
    times = np.split(sorted_spikes, split_idxs)
    depths = [np.mean(i) for i in np.split(sorted_spike_depths, split_idxs)]

    def _get_range(lo, hi):
        return np.array([np.sort(t) for t, d in
                         zip(times, depths) if lo < d <= hi])

    if location == 'striatum':
        data = _get_range(0, 1550)
    elif location == 'motor_ctx':
        data = _get_range(1550, 3840)
    elif location == 'thalamus':
        data = _get_range(0, 1634)
    elif location == 'hippocampus':
        data = _get_range(1634, 2797)
    else:  # visual_ctx
        data = _get_range(2797, 3840)

    # Caches the data to avoid recomputation.
    np.save(fpath, data)

    return data


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


def _load_reaching_helper(transformer, identifier, event, feature, neuron,
                          window_min, window_max, threshold, dir_name):
    # Loads the formatted data, if it has already been processed.
    fname = '{}.npz'.format('_'.join('{}'.format(i) for i in [
        event, feature, neuron, window_min, window_max, threshold, identifier
    ]))
    fpath = os.path.join(config.get_data_directory(), dir_name, fname)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            data = np.load(f)
            return data['x'], data['y']

    # Converts to seconds.
    window_max, window_min = window_max * 1e-3, window_min * 1e-3

    # Loads the reaching data normally.
    reaching_data = load_reaching_data(dir_name)

    events = list(reaching_data['events'].keys())
    features = list(reaching_data['features'].keys())

    # Checks the input arguments, throwing helpful error messages if needed.
    _arg_check('align event', event, events)
    _arg_check('feature', feature, features)
    _arg_check('neuron type', neuron, ('M1', 'PMd'))

    neuron_key = 'neurons_{}'.format(neuron)
    spike_times = np.asarray([
        np.squeeze(np.sort(s)) for s in reaching_data[neuron_key]
    ])

    # Applies the cutoff threshold.
    spike_freqs = np.asarray([len(t) / (t[-1] - t[0]) for t in spike_times])
    thresh_idxs = np.where(spike_freqs > threshold)[0]
    spike_times = spike_times[thresh_idxs]

    # Gets the reach angle, in radians.
    if feature == 'endpointOfReach':
        x = reaching_data['features'][feature] * np.pi / 180.0
        x = np.arctan2(np.sin(x), np.cos(x))
    else:
        x = reaching_data['features'][feature]

    # Gets the spike responses.
    event_data = reaching_data['events'][event].reshape(-1)

    # Gets the on and off times.
    on_off = np.sort(np.concatenate([
        event_data + window_min, event_data + window_max
    ]))

    # Checks that we haven't violated the order.
    window_diff = window_max - window_min
    for i in range(1, len(on_off), 2):
        d = on_off[i] - on_off[i-1]
        if abs(d - window_diff) > 1e-9:
            raise ValueError('Samples were found that overlap! Make sure that '
                             '`window_max` - `window_min` is small enough. '
                             'Time difference is {:.3f}s, average is {:.3f}s.'
                             .format(d, window_diff))

    # Applies the transformation.
    y = np.stack([
        transformer(n, on_off) for n in spike_times
    ]).transpose(1, 0)

    # Saves the dataset after processing it.
    with open(fpath, 'wb') as f:
        np.savez(f, x=x, y=y)

    return x, y


def load_reaching_rates(event='goCueTime', feature='endpointOfReach',
                        neuron='M1', window_min=0., window_max=500.,
                        threshold=10., dir_name='reaching'):
    '''Extracts the reach direction and spike rates from the reaching dataset.

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

        * :data:`x`: Array with shape :data:`(samples, features)`
        * :data:`y`: Array with shape :data:`(samples, neurons)`
    '''

    def _get_spike_rates(n, on_off):
        arrs = np.split(n, n.searchsorted(on_off))
        windows = [arrs[i] for i in range(1, len(arrs), 2)]
        return np.asarray([len(w) for w in windows])

    return _load_reaching_helper(
        transformer=_get_spike_rates,
        identifier='rates',
        event=event,
        feature=feature,
        neuron=neuron,
        window_min=window_min,
        window_max=window_max,
        threshold=threshold,
        dir_name=dir_name,
    )


def load_reaching_deltas(event='goCueTime', feature='endpointOfReach',
                         neuron='M1', window_min=0., window_max=500.,
                         threshold=10., dir_name='reaching'):
    '''Extracts the reach direction and spike deltas from the reaching dataset.

    The first spike delta is the difference between the event onset minus the
    min time and the first spike. The remaining spike deltas are the difference
    between the time of the current spike and the time of the previous spike.
    The last dimension of the :data:`y` data is a list of variable length,
    since there are a variable number of spikes.

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

        * :data:`x`: Array with shape :data:`(samples, features)`
        * :data:`y`: Array with shape :data:`(samples, neurons, deltas)`
    '''

    def _get_spike_deltas(n, on_off):
        arrs = np.split(n, n.searchsorted(on_off))
        windows = [[on_off[i-1]] + arrs[i] for i in range(1, len(arrs), 2)]
        deltas = [w[1:] - w[:-1] for w in windows]
        return deltas

    return _load_reaching_helper(
        transformer=_get_spike_deltas,
        identifier='deltas',
        event=event,
        feature=feature,
        neuron=neuron,
        window_min=window_min,
        window_max=window_max,
        threshold=threshold,
        dir_name=dir_name,
    )
