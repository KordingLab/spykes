import os
import numpy as np
import matplotlib.pyplot as plt

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
    get_trialtype
    get_spikecounts

    """

    def __init__(self, spiketimes, name='neuron'):
        """
        Initialize the object
        """
        self.name = name
        self.spiketimes = np.squeeze(np.sort(spiketimes))
        n_seconds = (self.spiketimes[-1]-self.spiketimes[0])
        n_spikes = np.size(spiketimes)
        self.firingrate = (n_spikes/n_seconds)

    #-----------------------------------------------------------------------
    def get_raster(self, events, features=None, conditions=None,
                   window=[-100, 500], binsize=10, plot=True,
                   sort=False):
        """
        Compute the raster and plot it

        Parameters
        ----------
        events : float, n x 1 array of event times in milliseconds
                (e.g. stimulus/ trial/ fixation onset, etc.)

        features : dictionary, each value is a n x 1 array of trial features
                (e.g. bolean for good trial, reaching angle, etc.)

        conditions : dictionary, each value is either a list of 1 element
                (e.g. [1] for good trials, [0] for bad trials) or of 2 elements
                for an interval (e.g reaching angles between [0,90] degrees)

        window :
            list of 2 elements, the time interval to consider (milliseconds)

        binsize : integer, bin size in milliseconds

        plot : boolean,

        Returns
        -------
        rasters : dictionary, with keys: 'binsize', 'conditions', 'window',
            and 'data'. 'data' is a dictionary where each value is a raster
            for each condition, or only one raster if conditions is not given
            or empty


        """

        window = [np.floor(window[0]/binsize)*binsize, np.ceil(window[1]/binsize)*binsize]
        # Get a set of binary indicators for trials of interest
        if features is None:
            features = list()
        if conditions is None:
            conditions = list()
        if len(conditions) > 0:
            trials = self.get_trialtype(features, conditions)
        else:
            trials = np.transpose(np.atleast_2d(np.ones(np.size(events)) == 1))

        # Initialize rasters
        rasters = dict()

        rasters['window'] = window
        rasters['binsize'] = binsize
        rasters['conditions'] = conditions
        rasters['data'] = dict()

        # Loop over each raster
        for r_idx in np.arange(trials.shape[1]):

            # Select events relevant to this raster
            selected_events = events[trials[:, r_idx]]

            raster = []

            bin_template = 1e-3 * np.arange(window[0],window[1]+binsize,binsize)
            for event_time in selected_events:
                bins = event_time + bin_template

                # consider only spikes within window
                searchsorted_idx = np.searchsorted(self.spiketimes,
                [event_time+1e-3 * window[0],event_time+1e-3 * window[1]])

                # bin the spikes into time bins
                spike_counts = np.histogram( \
                    self.spiketimes[searchsorted_idx[0]:searchsorted_idx[1]], bins)[0]

                raster.append(spike_counts)

            rasters['data'][r_idx] = np.array(raster)

        # Show the raster
        if plot is True:
            for i in np.arange(len(rasters['data'])):
                self.plot_raster(rasters, condition=i+1, sort=sort)
                plt.show()


        # Return all the rasters
        return rasters

    #-----------------------------------------------------------------------
    def plot_raster(self, rasters, condition = 1, condition_names=None,
        sort=False, cmap=['Greys'], has_title=True):
        """
        Plot rasters

        Parameters
        ----------
        rasters:
            dict, output of get_raster method

        condition_names: list

        sort:
            boolean, default is False. True for sorting rasters according
            to number of spikes.

        cmap:
            list, colormaps for rasters

        """
        r_idx = condition-1
        window = rasters['window']
        binsize = rasters['binsize']
        conditions = rasters['conditions']
        xtics = [window[0], 0, window[1]]
        xtics = [str(i) for i in xtics]
        xtics_loc = [0-0.5, (-window[0])/binsize-0.5, (window[1]-window[0])/binsize-0.5]

        raster = rasters['data'][r_idx]

        if len(raster)>0:
            if sort is True:
                # Sorting by total spike count in the duration
                raster_sorted = raster[np.sum(raster, axis=1).argsort()]
            else:
                raster_sorted = raster

            if len(cmap) > 1:
                plt.imshow(raster_sorted, aspect='auto',
                    interpolation='none', cmap=plt.get_cmap(cmap[r_idx]))
            else:
                plt.imshow(raster_sorted, aspect='auto',
                    interpolation='none', cmap=plt.get_cmap(cmap[0]))
            plt.axvline((-window[0])/binsize-0.5, color='r', linestyle='--')
            plt.ylabel('trials')
            plt.xlabel('time [ms]')
            plt.xticks(xtics_loc, xtics)

            if has_title:
                if len(conditions) > 0:
                    if condition_names:
                        plt.title('neuron %s: %s' % \
                        (self.name, condition_names[r_idx]))
                    else:
                        plt.title('neuron %s: Condition %d' % (self.name, r_idx+1))
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
            print 'No trials for condition %d!' % (r_idx+1)


    #-----------------------------------------------------------------------
    def get_psth(self, events, features=None, conditions=None, \
                 window=[-100, 500], binsize=10, plot=True):
        """
        Compute the psth and plot it

        Parameters
        ----------
        events: float, n x 1 array of event times in milliseconds
                (e.g. stimulus/ trial/ fixation onset, etc.)

        features: dictionary, each value is a n x 1 array of trial features
                (e.g. bolean for good trial, reaching angle, etc.)

        conditions: dictionary, each value is either a list of 1 element
                (e.g. [1] for good trials, [0] for bad trials) or of 2 elements
                for an interval (e.g reaching angles between [0,90] degrees)

        window:
            list of 2 elements, the time interval to consider in milliseconds

        binsize:
            integer, bin size in milliseconds

        plot: boolean

        Returns
        -------
        psth: dictionary, with keys: 'binsize', 'conditions', 'window',
            and 'data'. 'data' is a dictionary with as many values as
            conditions (or only one if conditions is an empty list or not
            defined). Each value in psth['data'] is itself a dictionary
            with keys 'mean' and 'sem' that correspond to the mean and sem
            of the psth for that condition


        """

        if features is None:
            features = []

        if conditions is None:
            conditions = []

        window = [np.floor(window[0]/binsize)*binsize, np.ceil(window[1]/binsize)*binsize]
        # Get all the rasters first
        rasters = self.get_raster(events, features, conditions, window, binsize, plot=False)

        # Open the figure
        # Initialize PSTH
        psth = dict()

        psth['window'] = window
        psth['binsize'] = binsize
        psth['conditions'] = conditions
        psth['data'] = dict()

        # Compute the PSTH
        for r_idx in rasters['data']:

            psth['data'][r_idx] = dict()
            raster = rasters['data'][r_idx]
            mean_psth = np.mean(raster, axis=0)/(1e-3*binsize)
            std_psth = np.sqrt(np.var(raster, axis=0))/(1e-3*binsize)

            sem_psth = std_psth/np.sqrt(float(np.shape(raster)[0]))

            psth['data'][r_idx]['mean'] = mean_psth
            psth['data'][r_idx]['sem'] = sem_psth

        # Visualize the PSTH
        if plot is True:
            self.plot_psth(psth)

            #for i, cond in enumerate(conditions):
            #    print 'Condition %d: %s; %d trials' % \
            #        (cond+1, str(conditions[cond]), np.shape(rasters['data'][i])[0])

        return psth

    #-----------------------------------------------------------------------
    def plot_psth(self, psth, event_name='event_onset',
            condition_names=None, ylim=None,
            colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E', '#FF07CD']):
        """
        Plot psth

        Parameters
        ----------
        psth: dict, output of get_psth method

        event_name: string, legend name for event

        condition_names: list, legend names for the conditions

        ylim: list

        colors: list

        """

        window = psth['window']
        binsize = psth['binsize']
        conditions = psth['conditions']

        scale = 0.1
        y_min = (1.0-scale)*np.nanmin([np.min( \
            psth['data'][psth_idx]['mean']) \
            for psth_idx in psth['data']])
        y_max = (1.0+scale)*np.nanmax([np.max( \
            psth['data'][psth_idx]['mean']) \
            for psth_idx in psth['data']])

        legend = [event_name]

        time_bins = np.arange(window[0],window[1],binsize) + binsize/2.0

        if ylim:
            plt.plot([0, 0], ylim, color='k', ls='--')
        else:
            plt.plot([0, 0], [y_min, y_max], color='k', ls='--')

        for i in psth['data']:
            if np.all(np.isnan(psth['data'][i]['mean'])):
                plt.plot(0,0,alpha=1.0, color=colors[i])
            else:
                plt.plot(time_bins, psth['data'][i]['mean'],
                color=colors[i], lw=1.5)

        for i in psth['data']:
            if len(conditions) > 0:
                if condition_names:
                    legend.append(condition_names[i])
                else:
                    legend.append('Condition %d' % (i+1))
            else:
                legend.append('all')

            if not np.all(np.isnan(psth['data'][i]['mean'])):
                plt.fill_between(time_bins, psth['data'][i]['mean'] - \
                psth['data'][i]['sem'], psth['data'][i]['mean'] + \
                psth['data'][i]['sem'], color=colors[i], alpha=0.2)

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

    #-----------------------------------------------------------------------
    def get_trialtype(self, features, conditions):
        """
        For an arbitrary query on features
        get a subset of trials

        Parameters
        ----------
        features: float, n x p array of features,
                  n trials, p features
                  (e.g. stimulus/ behavioral features)

        conditions: list of intervals on arbitrary features

        Returns
        -------
        trials: bool, n x 1 array of indicators

        """
        trials = []
        for rast in conditions:
            condition = conditions[rast]
            trials.append([np.all([np.all((features[rast] >= condition[rast][0], \
                                 features[rast] <= condition[rast][-1]), axis=0) \
                                 for rast in condition], axis=0)])
        return np.transpose(np.atleast_2d(np.squeeze(trials)))

    #-----------------------------------------------------------------------
    def get_spikecounts(self, events, window=np.array([50.0, 100.0])):
        """
        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus onset, trial onset, fixation onset, etc.)

        window: denoting the intervals (milliseconds)

        Returns
        -------
        spikecounts: float, n x 1 array of spike counts

        """
        spiketimes = self.spiketimes
        spikecounts = np.zeros(events.shape)
        for eve in range(events.shape[0]):
            spikecounts[eve] = np.sum(np.all((spiketimes >= events[eve] + 1e-3*window[0],\
                                            spiketimes <= events[eve] + 1e-3*window[1]),\
                                            axis=0))
        return spikecounts
