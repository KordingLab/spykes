from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

from fractions import gcd

from .neurovis import NeuroVis
from .. import utils
from ..config import DEFAULT_POPULATION_COLORS

# Defines the default colors for a PSTH plot.
DEFAULT_PSTH_COLORS = ['Blues', 'Reds', 'Greens']


class PopVis(object):
    '''Facilitates visualization of neuron population firing activity.

    Args:
        neuron_list (list of NeuroVis objects): The list of neurons to
            visualize (see the NeuroVis class in
            :class:`spykes.plot.neurovis`).

    Attributes:
        n_neurons (int): The number of neurons in the visualization.
        name (str): The name of this visualization.
    '''

    def __init__(self, neuron_list, name='PopVis'):
        self.neuron_list = neuron_list
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def n_neurons(self):
        return len(self.neuron_list)

    def get_all_psth(self, event=None, df=None, conditions=None,
                     window=[-100, 500], binsize=10, conditions_names=None,
                     plot=True, colors=DEFAULT_PSTH_COLORS):
        '''Iterates through all neurons and computes their PSTH's.

        Args:
            event (str): Column/key name of DataFrame/dictionary "data" which
                contains event times in milliseconds (e.g.
                stimulus/trial/ fixation onset, etc.).
            df (DataFrame or dictionary): The data to use.
            conditions (str): Column/key name of DataFrame/dictionary
                :data:`df` which contains the conditions by which the trials
                must be grouped
            window (list of 2 elements): Time interval to consider, in
                milliseconds.
            binsize (int): Bin size, in milliseconds.
            conditions_names (list of str): Legend names for conditions.
                Default are the unique values in :data:`df['conditions']`.
            plot (bool): If set, automatically plot; otherwise, don't.
            colors (list): List of colors for heatmap (only if plot is True).

        Returns:
            dict: With keys :data:`event`, :data:`conditions`, :data:`binsize`,
            :data:`window`, and :data:`data`. Each entry in
            :data:`psth['data']` is itself a dictionary with keys of
            each :data:`cond_id` that correspond to the means for that
            condition.
        '''
        all_psth = {
            'window': window,
            'binsize': binsize,
            'event': event,
            'conditions': conditions,
            'data': defaultdict(list),
        }

        for i, neuron in enumerate(self.neuron_list):
            psth = neuron.get_psth(
                event=event,
                df=df,
                conditions=conditions,
                window=window,
                binsize=binsize,
                plot=False,
            )
            for cond_id in np.sort(list(psth['data'].keys())):
                all_psth['data'][cond_id].append(psth['data'][cond_id]['mean'])

        for cond_id in np.sort(list(all_psth['data'].keys())):
            all_psth['data'][cond_id] = np.stack(all_psth['data'][cond_id])

        if plot is True:
            self.plot_heat_map(
                all_psth,
                conditions_names=conditions_names,
                colors=colors,
            )

        return all_psth

    def plot_heat_map(self, psth_dict, cond_id=None, conditions_names=None,
                      sortby=None, sortorder='descend', normalize=None,
                      neuron_names=True, colors=None, show=False):
        '''Plots heat map for neuron population

        Args:
            psth_dict (dict): With keys :data:`event`, :data:`conditions`,
                :data:`binsize`, :data:`window`, and :data:`data`. Each entry
                in :data:`psth['data']` is itself a dictionary with keys of
                each :data:`cond_id` that correspond to the means for that
                condition.
            cond_id (str): Which psth to plot indicated by the key in
                :data:`all_psth['data']`. If None then all are plotted.
            conditions_names (str or list of str): Name(s) to appear in the
                title.
            sortby (str or list): If :data:`rate`, sort by firing rate. If
                :data:`latency`, sort by peak latency. If a list or array is
                provided, it must correspond to integer indices to be used as
                sorting indices. If no sort order is provided, the data
                is returned as-is.
            sortorder (str): The direction to sort in, either :data:`descend`
                or :data:`ascend`.
            normalize (str): If :data:`all`, divide all PSTHs by highest peak
                firing rate in all neurons. If :data:`each`, divide each PSTH
                by its own peak firing rate. If None, do not normalize.
            neuron_names (bool): Whether or not to list the names of neurons on
                the side.
            colors (list of str): List of colors for the heatmap (as strings).
            show (bool): If set, show the plot once finished.
        '''
        if colors is None:
            colors = ['Blues', 'Reds', 'Greens']

        window = psth_dict['window']
        binsize = psth_dict['binsize']
        conditions = psth_dict['conditions']

        if cond_id is None:
            keys = np.sort(list(psth_dict['data'].keys()))
        else:
            keys = cond_id

        if conditions_names is None:
            conditions_names = keys

        for i, cond_id in enumerate(keys):
            # Sorts and norms the data.
            orig_data = psth_dict['data'][cond_id]
            normed_data = self._get_normed_data(orig_data, normalize=normalize)
            sort_idx = utils.get_sort_indices(
                normed_data,
                by=sortby,
                order=sortorder,
            )

            data = normed_data[sort_idx, :]

            plt.subplot(len(keys), 1, i + 1)
            plt.pcolormesh(data, cmap=colors[i % len(colors)])

            # Makes it visually appealing.
            xtic_len = gcd(abs(window[0]), window[1])
            xtic_labels = range(window[0], window[1] + xtic_len, xtic_len)
            xtic_locs = [(j - window[0]) / binsize for j in xtic_labels]

            if 0 not in xtic_labels:
                xtic_labels.append(0)
                xtic_locs.append(-window[0] / binsize)

            plt.xticks(xtic_locs, xtic_labels)
            plt.axvline((-window[0]) / binsize, color='r',
                        linestyle='--')

            if neuron_names:
                unsorted_ylabels = [neuron.name for neuron in self.neuron_list]
                ylabels = [unsorted_ylabels[j] for j in sort_idx]
            else:
                ylabels = ["" for neuron in self.neuron_list]

            plt.yticks(np.arange(data.shape[0]) + 0.5, ylabels)

            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_frame_on(False)

            plt.tick_params(axis='x', which='both', top='off')
            plt.tick_params(axis='y', which='both', left='off', right='off')

            plt.xlabel('time [ms]')
            plt.ylabel('Neuron')
            plt.title("%s: %s" %
                      (conditions, conditions_names[i]))
            plt.colorbar()

        if show:
            plt.show()

    def plot_population_psth(self, all_psth=None, event=None, df=None,
                             conditions=None, cond_id=None, window=[-100, 500],
                             binsize=10, conditions_names=None,
                             event_name='event_onset', ylim=None,
                             colors=DEFAULT_POPULATION_COLORS, show=False):
        '''Plots population PSTH's.

        This involves two steps. First, it normalizes each neuron's PSTH across
        the conditions. Second, it averages out and plots population PSTH.

        Args:
            all_psth (dict): With keys :data:`event`, :data:`conditions`,
                :data:`binsize`, :data:`window`, and :data:`data`. Each entry
                in :data:`psth['data']` is itself a dictionary with keys of
                each :data:`cond_id` that correspond to the means for that
                condition.
            event (str): Column/key name of the :data:`df` DataFrame/dictionary
                which contains event times in milliseconds
                (stimulus/trial/fixation onset, etc.).
            df (DataFrame or dictionary): DataFrame containing the data to
                plot, or a dictionary corresponding to such a DataFrame.
            conditions (str): Column/key name of DataFrame/dictionary
                :data:`df` which contains the conditions by which the trials
                must be grouped.
            cond_id (list): Which psth to plot indicated by the key in
                :data:`all_psth['data']``. If :data:`None` then all are
                plotted.
            window (list of 2 elements): Time interval to consider, in
                milliseconds.
            binsize (int): Bin size, in milliseconds.
            conditions_names (list): Legend names for conditions. Default are
                the unique values in :data:`df['conditions']`.
            event_name (string): Legend name for event. Default is the actual
                event name
            ylim (list): The minimum and maximum values for Y.
            colors (list): The colors to use for the plot (as strings).
            show (bool): If set, show the plot once it has been created.
        '''

        # placeholder in order to use NeuroVis functionality
        base_neuron = NeuroVis(spiketimes=range(10), name=self.name)

        if all_psth is None:
            psth = self.get_all_psth(
                event=event,
                df=df,
                conditions=conditions,
                window=window,
                binsize=binsize,
                plot=False,
            )
        else:
            psth = copy.deepcopy(all_psth)

        keys = np.sort(list(psth['data'].keys()))

        # Normalizes each neuron across all conditions.
        for i in range(self.n_neurons):
            max_rates = list()
            for key in keys:
                max_rates.append(np.amax(psth['data'][key][i, :]))
            norm_factor = max(max_rates)
            for key in keys:
                psth['data'][key][i, :] /= norm_factor

        # Averages out all the neurons and plots.
        for i, key in enumerate(keys):
            normed_data = psth['data'][key]
            psth['data'][key] = dict()
            psth['data'][key]['mean'] = np.nanmean(normed_data, axis=0)
            psth['data'][key]['sem'] = \
                np.nanvar(normed_data, axis=0) / (len(self.neuron_list)**.5)

        # Plots the PSTH.
        base_neuron.plot_psth(
            psth=psth,
            event_name=event_name,
            cond_id=cond_id,
            conditions_names=conditions_names,
            ylim=ylim,
            colors=colors,
        )

        # Adds the appropriate title.
        plt.title("%s Population PSTH: %s" % (self.name, psth['conditions']))

        if show:
            plt.show()

    def _get_normed_data(self, data, normalize):
        '''Normalizes all PSTH data

        Args:
            data (2-D numpy array): Array with shape
                :data:`(n_neurons, n_bins)`
            normalize (str): If :data:`all`, divide all PSTHs by highest peak
                firing rate in all neurons. If :data:`each`, divide each PSTH
                by its own peak firing rate. If None, do not normalize.

        Returns:
            array: The original data array, divided such
            that all values fall between 0 and 1.
        '''
        max_rates = np.amax(data, axis=1)

        # Computes the normalization factors.
        if normalize == 'all':
            norm_factors = np.ones([data.shape[0], 1]) * np.amax(max_rates)
        elif normalize == 'each':
            norm_factors = (
                np.reshape(max_rates, (max_rates.shape[0], 1)) *
                np.ones((1, data.shape[1]))
            )
        elif normalize is None:
            norm_factors = np.ones([data.shape[0], 1])
        else:
            raise ValueError('Invalid norm factors: {}'.format(norm_factors))

        return data / norm_factors
