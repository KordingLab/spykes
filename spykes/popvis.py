import numpy as np
import matplotlib.pyplot as plt
import copy

from fractions import gcd

from spykes.neurovis import NeuroVis
from . import utils


class PopVis(object):

    """
    This class implements several conveniences for
    visualizing firing activity of neuron populations
    Parameters
    ----------
    neuron_list: list of NeuroVis objects
    (see NeuroVis class in spykes.neurovis)
    Internal variables
    ------------------
    neuron_list: list of NeuroVis objects
    n_neurons: int
    Callable methods
    ----------------
    get_all_psth
    plot_heat_map
    plot_population_psth
    Class methods
    -------------
    _get_normed_data
    """

    def __init__(self, neuron_list, name=""):
        """
        Initialize the object
        """
        self.neuron_list = neuron_list
        self.n_neurons = len(neuron_list)
        self.name = name

    def get_all_psth(self, event=None, df=None, conditions=None,
                     window=[-100, 500], binsize=10, conditions_names=None,
                     plot=True, colors=['Blues', 'Reds', 'Greens']):
        """
        Iterates through all neurons and computes their PSTH's
        Parameters
        ----------
        event: str
            Column/key name of DataFrame/dictionary "data" which contains
            event times in milliseconds (e.g. stimulus/ trial/ fixation onset,
            etc.)
        df: DataFrame (or dictionary)
        conditions: str
            Column/key name of DataFrame/dictionary "data" which contains the
            conditions by which the trials must be grouped
        window: list of 2 elements
            Time interval to consider (milliseconds)
        binsize: int
            Bin size in milliseconds
        conditions_names:
            Legend names for conditions. Default are the unique values in
            df['conditions']
        plot: bool
            Whether to automatically plot or not
        colors:
            list of colors for heatmap (only if plot is True)
        Returns
        -------
        all_psth : dictionary
            With keys: 'event', 'conditions', 'binsize', 'window', and 'data'.
            Each entry in psth['data'] is itself a dictionary with keys of
            each cond_id that correspond to the means for that condition
        """

        all_psth = dict()

        all_psth['window'] = window
        all_psth['binsize'] = binsize
        all_psth['event'] = event
        all_psth['conditions'] = conditions
        all_psth['data'] = dict()

        for i, neuron in enumerate(self.neuron_list):

            psth = neuron.get_psth(event=event, df=df,
                                   conditions=conditions, window=window,
                                   binsize=binsize, plot=False)

            for cond_id in np.sort(list(psth['data'].keys())):

                if cond_id not in all_psth['data']:
                    all_psth['data'][cond_id] = list()

                all_psth['data'][cond_id].append(
                    psth['data'][cond_id]['mean'])

        for cond_id in np.sort(list(all_psth['data'].keys())):
            all_psth['data'][cond_id] = np.stack(all_psth['data'][cond_id])

        if plot is True:

            self.plot_heat_map(all_psth, conditions_names=conditions_names,
                               colors=colors)

        return all_psth

    def plot_heat_map(self, psth_dict, cond_id=None,
                      conditions_names=None, sortby=None, sortorder='descend',
                      normalize=None, neuron_names=True,
                      colors=['Blues', 'Reds', 'Greens']):
        """
        Plots heat map for neuron population
        Parameters
        ----------
        psth_dict : dictionary
            With keys: 'event', 'conditions', 'binsize', 'window', and 'data'.
            Each entry in psth['data'] is itself a dictionary with keys of
            each cond_id that correspond to the means for that condition
        cond_id
            Which psth to plot indicated by the key in all_psth['data'].
            If None then all are plotted.
        conditions_names: str (or list)
            Name(s) to appear in the title
        sortby: str or list
            None:
            'rate': sort by firing rate
            'latency': sort by peak latency
            list: list of integer indices to be used as sorting indicces
        sortorder: direction to sort in
            'descend'
            'ascend'
        normalize: str
            None
            'all' : divide all PSTHs by highest peak firing rate in all neurons
            'each' : divide each PSTH by its own peak firing rate
        neuron_names: bool
            Whether or not to list the names of neurons on the side
        colors: list of colors for heatmap
        """

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

            # sort and norm the data

            orig_data = psth_dict['data'][cond_id]

            normed_data = self._get_normed_data(orig_data, normalize=normalize)

            sort_idx = utils.get_sort_indices(normed_data, sortby=sortby,
                                              sortorder=sortorder)

            data = normed_data[sort_idx, :]

            plt.subplot(len(keys), 1, i + 1)
            plt.pcolormesh(data, cmap=colors[i % len(colors)])

            # making it visually appealing

            xtic_len = gcd(abs(window[0]), window[1])
            xtic_labels = range(window[0], window[1] + xtic_len, xtic_len)
            xtic_locs = [(j - window[0]) / binsize for j in xtic_labels]

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

        plt.show()

    def plot_population_psth(self, all_psth=None, event=None, df=None,
                             conditions=None, cond_id=None, window=[-100, 500],
                             binsize=10, conditions_names=None,
                             event_name='event_onset', ylim=None,
                             colors=['#F5A21E', '#134B64', '#EF3E34',
                                     '#02A68E', '#FF07CD']):
        """
        1. Normalizes each neuron's PSTH across condition
        2. Averages out and plots population PSTH
        Parameters
        ----------
        all_psth : dictionary
            With keys: 'event', 'conditions', 'binsize', 'window', and 'data'.
            Each entry in psth['data'] is itself a dictionary with keys of
            each cond_id that correspond to the means for that condition
        event: str
            Column/key name of DataFrame/dictionary "data" which contains
            event times in milliseconds (e.g. stimulus/ trial/ fixation onset,
            etc.)
        df: DataFrame (or dictionary)
        conditions: str
            Column/key name of DataFrame/dictionary "data" which contains the
            conditions by which the trials must be grouped
        cond_id: list
            Which psth to plot indicated by the key in all_psth['data'].
            If None then all are plotted.
        window: list of 2 elements
            Time interval to consider (milliseconds)
        binsize: int
            Bin size in milliseconds
        conditions_names:
            Legend names for conditions. Default are the unique values in
            df['conditions']
        event_name: string
            Legend name for event. Default is the actual 'event' name
        ylim: list
        colors: list
        """

        # placeholder in order to use NeuroVis functionality
        base_neuron = NeuroVis(spiketimes=range(10), name=self.name)

        if all_psth is None:
            psth = self.get_all_psth(event=event, df=df,
                                     conditions=conditions, window=window,
                                     binsize=binsize, plot=False)
        else:
            psth = copy.deepcopy(all_psth)

        keys = np.sort(list(psth['data'].keys()))

        # normalize each neuron across all conditions

        for i in range(self.n_neurons):

            max_rates = list()

            for key in keys:
                max_rates.append(np.amax(psth['data'][key][i, :]))

            norm_factor = max(max_rates)

            for key in keys:
                psth['data'][key][i, :] /= norm_factor

        # average out and plot

        for i, key in enumerate(keys):

            normed_data = psth['data'][key]

            psth['data'][key] = dict()
            psth['data'][key]['mean'] = np.nanmean(normed_data, axis=0)
            psth['data'][key]['sem'] = \
                np.nanvar(normed_data, axis=0) / (len(self.neuron_list)**.5)

        base_neuron.plot_psth(psth=psth, event_name=event_name,
                              cond_id=cond_id,
                              conditions_names=conditions_names, ylim=ylim,
                              colors=colors)

        plt.title("%s Population PSTH: %s" % (self.name, psth['conditions']))

    def _get_normed_data(self, data, normalize):
        """
        Normalizes all PSTH data
        Parameters
        ----------
        data : 2-D numpy array
            n_neurons x n_bins
        normalize: str
            None
            'all' : divide all PSTHs by highest peak firing rate in all neurons
            'each' : divide each PSTH by its own peak firing rate
        Returns
        ----------
        normed_data : original data array that has been divided s.t. all values
            fall between [0,1]
        """
        norm_factors = np.ones([data.shape[0], 1])
        max_rates = np.amax(data, axis=1)

        if normalize == 'all':
            norm_factors *= np.amax(max_rates)

        elif normalize == 'each':
            norm_factors = np.reshape(max_rates, (max_rates.shape[0], 1)) * \
                np.ones([1, data.shape[1]])

        return (data / norm_factors)
