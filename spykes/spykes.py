import numpy as np
import matplotlib.pyplot as plt


class Spyke(object):
    """
    This class implements several conveniences for
    visualizing firing activity of single neurons

    Parameters
    ----------
    spiketimes: float, array of spike times


    Methods
    -------
    plot_raster
    plot_psth
    get_trialtype
    get_spikecounts
    plot_tuning_curve
    fit_tuning_curve
    """

    def __init__(self, spiketimes, name='neuron'):
        """
        Initialize the object
        """
        self.name = name
        self.spiketimes = spiketimes
        n_seconds = (self.spiketimes[-1]-self.spiketimes[0])
        n_spikes = np.size(spiketimes)
        self.firingrate = (n_spikes/n_seconds)

    #-----------------------------------------------------------------------
    def get_raster(self, events, features=None, conditions=None, \
                   window=[-100, 500], binsize=10, plot=True, sort=True):
        """
        Compute the raster and plot it

        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus/ trial/ fixation onset, etc.)

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

        # Assign time bins
        firstspike = self.spiketimes[0]
        lastspike = self.spiketimes[-1]
        bins = np.arange(np.floor(firstspike), np.ceil(lastspike), 1e-3*binsize)

        # Loop over each raster
        for rast in np.arange(trials.shape[1]):

            # Select events relevant to this raster
            selected_events = events[trials[:, rast]]

            # Eliminate events before the first spike after last spike
            selected_events = selected_events[selected_events > firstspike]
            selected_events = selected_events[selected_events < lastspike]

            # bin the spikes into time bins
            spike_counts = np.histogram(self.spiketimes, bins)[0]

            # bin the events into time bins
            event_counts = np.histogram(selected_events, bins)[0]
            event_bins = np.where(event_counts > 0)[0]

            raster = np.array([(spike_counts[(i+window[0]/binsize): \
                                             (i+window[1]/binsize+1)]) \
                               for i in event_bins])
            rasters[rast] = raster

        # Show the raster
        if plot is True:

            xtics = [window[0], 0, window[1]]
            xtics = [str(i) for i in xtics]
            xtics_loc = [0, (-window[0])/binsize, (window[1]-window[0])/binsize]

            for i, rast in enumerate(rasters):

                raster = rasters[rast]

                if sort is True:
                    # Sorting by total spike count in the duration
                    raster_sorted = raster[np.sum(raster, axis=1).argsort()]
                else:
                    raster_sorted = raster

                plt.imshow(raster_sorted, aspect='auto', \
                    interpolation='none', cmap=plt.get_cmap('Greys'))
                plt.axvline((-window[0])/binsize, color='r', linestyle='--')
                plt.ylabel('trials')
                plt.xlabel('time [ms]')
                plt.xticks(xtics_loc, xtics)
                if len(conditions) > 0:
                    plt.title('neuron %s: Condition %d' % (self.name, i+1))
                else:
                    plt.title('neuron %s' % self.name)

                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_yaxis().set_tick_params(direction='out')
                ax.get_xaxis().set_tick_params(direction='out')

                plt.tick_params(axis='x', which='both', top='off')
                plt.tick_params(axis='y', which='both', right='off')
                plt.show()

                if len(conditions) > 0:
                    print 'Condition %d: %s' % (i+1, str(conditions[i]))

        # Return all the rasters
        return rasters

    #-----------------------------------------------------------------------
    def get_psth(self, events, features=None, conditions=None, \
                 window=[-100, 500], binsize=10, plot=True, \
                 colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E', '#FF07CD']):
        """
        Compute the psth and plot it

        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus/ trial/ fixation onset, etc.)
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

        # Compute the PSTH
        for i, rast in enumerate(rasters):

            psth[i] = dict()
            raster = rasters[rast]
            mean_psth = np.mean(raster, axis=0)/(1e-3*binsize)
            std_psth = np.sqrt(np.var(raster, axis=0))/(1e-3*binsize)

            sem_psth = std_psth/np.sqrt(float(np.shape(raster)[0]))

            psth[i]['mean'] = mean_psth
            psth[i]['sem'] = sem_psth

        # Visualize the PSTH
        if plot is True:

            plt.figure()

            scale = 0.1
            y_min = (1.0-scale)*np.min([np.min( \
                np.mean(rasters[raster_idx], axis=0)/(1e-3*binsize)) \
                for raster_idx in rasters])
            y_max = (1.0+scale)*np.max([np.max( \
                np.mean(rasters[raster_idx], axis=0)/(1e-3*binsize)) \
                for raster_idx in rasters])

            legend = ['event onset']

            time_bins = np.append( \
                np.linspace(window[0], 0, num=np.abs(window[0])/binsize+1), \
                np.linspace(0, window[1], num=np.abs(window[1])/binsize+1)[1:])

            plt.plot([0, 0], [y_min, y_max], color='k', ls='--')

            for i, rast in enumerate(rasters):
                plt.plot(time_bins, psth[i]['mean'], color=colors[i], lw=1.5)

            for i, rast in enumerate(rasters):
                plt.fill_between(time_bins, psth[i]['mean']-psth[i]['sem'], \
                    psth[i]['mean']+psth[i]['sem'], color=colors[i], alpha=0.2)

                if len(conditions) > 0:
                    legend.append('Condition %d' % (i+1))
                else:
                    legend.append('all')

            plt.title('neuron %s' % self.name)
            plt.xlabel('time [ms]')
            plt.ylabel('spikes per second [spks/s]')
            plt.ylim([y_min, y_max])

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_yaxis().set_tick_params(direction='out')
            ax.get_xaxis().set_tick_params(direction='out')

            plt.tick_params(axis='y', right='off')
            plt.tick_params(axis='x', top='off')

            plt.legend(legend, frameon=False)

            plt.show()

        for i, cond in enumerate(conditions):
            print 'Condition %d: %s; %d trials' % \
                (cond+1, str(conditions[cond]), np.shape(rasters[i])[0])

        return psth

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

        Outputs
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
    def get_spikecounts(self, events, window=1e-3*np.array([50.0, 100.0])):
        """
        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus onset, trial onset, fixation onset, etc.)
        win: denoting the intervals

        Outputs
        -------
        spikecounts: float, n x 1 array of spike counts
        """
        spiketimes = self.spiketimes
        spikecounts = np.zeros(events.shape)
        for eve in range(events.shape[0]):
            spikecounts[eve] = np.sum(np.all((spiketimes >= events[eve] + window[0],\
                                            spiketimes <= events[eve] + window[1]),\
                                            axis=0))
        return spikecounts
