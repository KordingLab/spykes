import os
import numpy as np
import matplotlib.pyplot as plt
from . import utils

plt.style.use(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../mpl_styles/spykes.mplstyle')
)


class NeuroVis(object):

    """
    This class implements several conveniences for
    visualizing firing activity of single neurons

    Parameters
    ----------
    spiketimes: float, array of spike times


    Methods
    -------
    get_raster
    plot_raster
    get_psth
    plot_psth
    get_spikecounts

    """

    def __init__(self, spiketimes, name='neuron'):
        """
        Initialize the object
        """
        self.name = name
        self.spiketimes = np.squeeze(np.sort(spiketimes))
        n_seconds = (self.spiketimes[-1] - self.spiketimes[0])
        n_spikes = np.size(spiketimes)
        self.firingrate = (n_spikes / n_seconds)

    # -----------------------------------------------------------------------
    def get_raster(self, event=None, conditions=None, df=None,
                   window=[-100, 500], binsize=10, plot=True,
                   sortby=None, sortorder='descend'):
        """
        Compute the raster and plot it

        Parameters
        ----------
        event: str
            Column/key name of DataFrame/dictionary "data" which contains
            event times in milliseconds (e.g. stimulus/ trial/ fixation onset,
            etc.)

        conditions: str
            Column/key name of DataFrame/dictionary "data" which contains the
            conditions by which the trials must be grouped

        df: DataFrame (or dictionary)

        window: list of 2 elements
            Time interval to consider (milliseconds)

        binsize: int
            Bin size in milliseconds

        plot: bool
            If True then plot

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
        rasters : dictionary
            With keys: 'event', 'conditions', 'binsize','window', and 'data'.
            rasters['data'] is a dictionary where each value is a raster for
            each unique entry of df['conditions']

        """

        if not type(df) is dict:
            df = df.reset_index()

        window = [np.floor(window[0] / binsize) * binsize,
                  np.ceil(window[1] / binsize) * binsize]

        # Get a set of binary indicators for trials of interest
        if conditions:
            trials = dict()
            for cond_id in np.sort(df[conditions].unique()):
                trials[cond_id] = \
                    np.where((df[conditions] == cond_id).apply(
                        lambda x: (0, 1)[x]).values)[0]
        else:
            trials = dict()
            trials[0] = np.where(np.ones(np.size(df[event])))[0]

        # Initialize rasters
        rasters = dict()
        rasters['event'] = event
        rasters['conditions'] = conditions
        rasters['window'] = window
        rasters['binsize'] = binsize
        rasters['data'] = dict()

        # Loop over each raster
        for cond_id in trials:
            # Select events relevant to this raster
            selected_events = df[event][trials[cond_id]]

            raster = []

            bin_template = 1e-3 * \
                np.arange(window[0], window[1] + binsize, binsize)
            for event_time in selected_events:
                bins = event_time + bin_template

                # consider only spikes within window
                searchsorted_idx = np.searchsorted(self.spiketimes,
                                                   [event_time + 1e-3 *
                                                    window[0],
                                                    event_time + 1e-3 *
                                                    window[1]])

                # bin the spikes into time bins
                spike_counts = np.histogram(
                    self.spiketimes[searchsorted_idx[0]:searchsorted_idx[1]],
                    bins)[0]

                raster.append(spike_counts)

            rasters['data'][cond_id] = np.array(raster)

        # Show the raster
        if plot is True:
            self.plot_raster(rasters, cond_id=None, sortby=sortby,
                             sortorder=sortorder)

        # Return all the rasters
        return rasters

    # -----------------------------------------------------------------------
    def plot_raster(self, rasters, cond_id=None, cond_name=None, sortby=None,
                    sortorder='descend', cmap='Greys', has_title=True):
        """
        Plot one raster

        Parameters
        ----------
        rasters: dict
            Output of get_raster method

        cond_id
            Which raster to plot indicated by the key in rasters['data'].
            If None then all are plotted.

        cond_name: str
            Name to appear in the title

        sortby: str or list
            None:
            'rate': sort by firing rate
            'latency': sort by peak latency
            list: list of integer indices to be used as sorting indicces

        sortorder: direction to sort in
            'descend'
            'ascend'

        cmap: str
            Colormap for raster

        has_title: bool
            If True then adds title

        """
        window = rasters['window']
        binsize = rasters['binsize']

        xtics = [window[0], 0, window[1]]
        xtics = [str(i) for i in xtics]
        xtics_loc = [-0.5, (-window[0]) / binsize - 0.5,
                     (window[1] - window[0]) / binsize - 0.5]

        if cond_id is None:
            for cond in rasters['data'].keys():
                self.plot_raster(rasters, cond_id=cond, cond_name=cond_name,
                                 sortby=sortby, sortorder=sortorder, cmap=cmap,
                                 has_title=has_title)
                plt.show()
        else:
            raster = rasters['data'][cond_id]

            if len(raster) > 0:

                sort_idx = utils.get_sort_indices(data=raster, sortby=sortby,
                                                  sortorder=sortorder)
                raster_sorted = raster[sort_idx]

                plt.imshow(raster_sorted, aspect='auto',
                           interpolation='none', cmap=plt.get_cmap(cmap))

                plt.axvline(
                    (-window[0]) / binsize - 0.5, color='r', linestyle='--')
                plt.ylabel('trials')
                plt.xlabel('time [ms]')
                plt.xticks(xtics_loc, xtics)

                if has_title:
                    if cond_id:
                        if cond_name:
                            plt.title('neuron %s. %s' %
                                      (self.name, cond_name))
                        else:
                            plt.title('neuron %s. %s: %s' %
                                      (self.name, rasters['conditions'],
                                       cond_id))
                    else:
                        plt.title('neuron %s' % self.name)

                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                plt.tick_params(axis='x', which='both', top='off')
                plt.tick_params(axis='y', which='both', right='off')

            else:
                print 'No trials for this condition!'

    # -----------------------------------------------------------------------
    def get_psth(self, event=None, df=None, conditions=None, cond_id=None,
                 window=[-100, 500], binsize=10, plot=True, event_name=None,
                 conditions_names=None, ylim=None,
                 colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E',
                         '#FF07CD']):
        """
        Compute the psth and plot it

        Parameters
        ----------
        event: str
            Column/key name of DataFrame/dictionary "data" which contains
            event times in milliseconds (e.g. stimulus/ trial/ fixation onset,
            etc.)

        conditions: str
            Column/key name of DataFrame/dictionary "data" which contains the
            conditions by which the trials must be grouped

        cond_id: list
            Which psth to plot indicated by the key in all_psth['data'].
            If None then all are plotted.

        df: DataFrame (or dictionary)

        window: list of 2 elements
            Time interval to consider (milliseconds)

        binsize: int
            Bin size in milliseconds

        plot: bool
            If True then plot

        event_name: string
            Legend name for event. Default is the actual 'event' name

        conditions_names:
            Legend names for conditions. Default are the unique values in
            df['conditions']

        ylim: list with 2 elements

        colors: list

        Returns
        -------
        psth : dictionary
            With keys: 'event', 'conditions', 'binsize', 'window', and 'data'.
            Each entry in psth['data'] is itself a dictionary with keys 'mean'
            and 'sem' that correspond to the mean and sem of the psth for that
            condition
        """

        window = [np.floor(window[0] / binsize) * binsize,
                  np.ceil(window[1] / binsize) * binsize]
        # Get all the rasters first
        rasters = self.get_raster(event=event, df=df,
                                  conditions=conditions,
                                  window=window, binsize=binsize, plot=False)

        # Initialize PSTH
        psth = dict()

        psth['window'] = window
        psth['binsize'] = binsize
        psth['event'] = event
        psth['conditions'] = conditions
        psth['data'] = dict()

        # Compute the PSTH
        for cond_id in np.sort(rasters['data'].keys()):

            psth['data'][cond_id] = dict()
            raster = rasters['data'][cond_id]
            mean_psth = np.mean(raster, axis=0) / (1e-3 * binsize)
            std_psth = np.sqrt(np.var(raster, axis=0)) / (1e-3 * binsize)

            sem_psth = std_psth / np.sqrt(float(np.shape(raster)[0]))

            psth['data'][cond_id]['mean'] = mean_psth
            psth['data'][cond_id]['sem'] = sem_psth

        if plot is True:
            if not event_name:
                event_name = event
                conditions_names = psth['data'].keys()
            self.plot_psth(psth, ylim=ylim, event_name=event_name,
                           conditions_names=conditions_names,
                           colors=colors)

        return psth

    # -----------------------------------------------------------------------
    def plot_psth(self, psth, event_name='event_onset', conditions_names=None,
                  cond_id=None, ylim=None, colors=['#F5A21E', '#134B64',
                                                   '#EF3E34', '#02A68E',
                                                   '#FF07CD']):
        """
        Plot psth

        Parameters
        ----------
        psth: dict, output of get_psth method

        event_name: string
            Legend name for event. Default is the actual 'event' name

        conditions_names:
            Legend names for conditions. Default are the keys in psth['data']

        cond_id: list
            Which psth to plot indicated by the key in all_psth['data'].
            If None then all are plotted.

        ylim: list

        colors: list

        """

        window = psth['window']
        binsize = psth['binsize']
        conditions = psth['conditions']

        if cond_id is None:
            keys = np.sort(psth['data'].keys())
        else:
            keys = cond_id

        if conditions_names is None:
            conditions_names = keys

        scale = 0.1
        y_min = (1.0 - scale) * np.nanmin([np.min(
            psth['data'][psth_idx]['mean'])
            for psth_idx in psth['data']])
        y_max = (1.0 + scale) * np.nanmax([np.max(
            psth['data'][psth_idx]['mean'])
            for psth_idx in psth['data']])

        legend = [event_name]

        time_bins = np.arange(window[0], window[1], binsize) + binsize / 2.0

        if ylim:
            plt.plot([0, 0], ylim, color='k', ls='--')
        else:
            plt.plot([0, 0], [y_min, y_max], color='k', ls='--')

        for i, cond_id in enumerate(keys):

            if np.all(np.isnan(psth['data'][cond_id]['mean'])):
                plt.plot(0, 0, alpha=1.0, color=colors[i % len(colors)])
            else:
                plt.plot(time_bins, psth['data'][cond_id]['mean'],
                         color=colors[i % len(colors)], lw=1.5)

        for i, cond_id in enumerate(keys):
            if conditions > 0:
                legend.append('%s' % str(conditions_names[i]))
            else:
                legend.append('all')

            if not np.all(np.isnan(psth['data'][cond_id]['mean'])):
                plt.fill_between(time_bins, psth['data'][cond_id]['mean'] -
                                 psth['data'][cond_id]['sem'],
                                 psth['data'][cond_id]['mean'] +
                                 psth['data'][cond_id]['sem'],
                                 color=colors[i % len(colors)],
                                 alpha=0.2)

        if conditions:
            plt.title('neuron %s: %s' % (self.name, conditions))
        else:
            plt.title('neuron %s' % self.name)

        plt.xlabel('time [ms]')
        plt.ylabel('spikes per second [spks/s]')

        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim([y_min, y_max])

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tick_params(axis='y', right='off')
        plt.tick_params(axis='x', top='off')

        plt.legend(legend, frameon=False)

    # -----------------------------------------------------------------------
    def get_spikecounts(self, event=None, df=None,
                        window=np.array([50.0, 100.0])):
        """
        Parameters
        ----------
        event: str
            Column/key name of DataFrame/dictionary "data" which contains
            event times in milliseconds (e.g. stimulus/ trial/ fixation onset,
            etc.)

        window: list of 2 elements
            Time interval to consider (milliseconds)

        Returns
        -------
        spikecounts: float, n x 1 array of spike counts

        """
        events = df[event].values
        spiketimes = self.spiketimes
        spikecounts = np.zeros(events.shape)

        for i, eve in enumerate(events):
            spikecounts[i] = np.sum(np.all((spiketimes >= eve +
                                            1e-3 * window[0],
                                            spiketimes <= eve +
                                            1e-3 * window[1]),
                                           axis=0))
        return spikecounts
