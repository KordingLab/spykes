import numpy as np
import matplotlib.pyplot as plt


class Spyke:
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
    def get_raster(self, events, features=[], conditions=[], \
                   window=[-100, 500], binsize=10, plot=True, sort=True):
        """
        Compute the raster and plot it

        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus/ trial/ fixation onset, etc.)

        """

        # Get a set of binary indicators for trials of interest
        if len(conditions) > 0:
            trials = self.get_trialtype(features, conditions)
        else:
            trials = np.transpose(np.atleast_2d(np.ones(np.size(events)) == 1))

        # Initialize rasters
        Rasters = dict()

        # Assign time bins
        firstspike = self.spiketimes[0]
        lastspike = self.spiketimes[-1]
        bins = np.arange(np.floor(firstspike),np.ceil(lastspike), 1e-3*binsize)

        # Loop over each raster
        for r in np.arange(trials.shape[1]):

            # Select events relevant to this raster
            selected_events = events[trials[:,r]]

            # Eliminate events before the first spike after last spike
            selected_events = selected_events[selected_events > firstspike]
            selected_events = selected_events[selected_events < lastspike]

            # bin the spikes into time bins
            spike_counts = np.histogram(self.spiketimes, bins)[0]

            # bin the events into time bins
            event_counts = np.histogram(selected_events, bins)[0]
            event_bins =  np.where(event_counts > 0)[0]

            raster = np.array([(spike_counts[(i+window[0]/binsize): \
                                             (i+window[1]/binsize)]) \
                               for i in event_bins])
            Rasters[r] = raster

        # Show the raster
        if plot == True:

            xtics = np.arange(window[0], window[1], binsize*10)
            xtics = [str(i) for i in xtics]

            for i,r in enumerate(Rasters):

                raster = Rasters[r]

                if sort == True:
                    # Sorting by total spike count in the duration
                    raster_sorted = raster[np.sum(raster, axis=1).argsort()]
                else:
                    raster_sorted = raster

                plt.imshow(raster_sorted, aspect='auto', interpolation='none', cmap=plt.get_cmap('Greys'))
                plt.axvline((-window[0])/binsize, color='r', linestyle='--')


                plt.ylabel('trials')
                plt.xlabel('time [ms]')

                plt.xticks(np.arange(0,(window[1]-window[0])/binsize,10), xtics)
                if len(conditions)>0:
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

                #ax.tick_params(axis='x', colors='red')
                #ax.tick_params(axis='y', colors='red')
                
                
                plt.show()

                if len(conditions)>0:
                    print 'Condition %d: %s' % (i+1, str(conditions[i]))

        # Return all the rasters
        return Rasters

    #-----------------------------------------------------------------------
    def get_psth(self, events, features=[], conditions=[], \
                 window=[-100, 500], binsize=10, plot=True, \
                 colors=['#F5A21E','#134B64','#EF3E34','#02A68E','#FF07CD']):
        """
        Compute the psth and plot it

        Parameters
        ----------
        events: float, n x 1 array of event times
                (e.g. stimulus/ trial/ fixation onset, etc.)
        """

        # Get all the rasters first
        Rasters = self.get_raster(events, features, conditions, window, binsize, plot=False)

        # Open the figure
        # Initialize PSTH
        PSTH = dict()

        # Compute the PSTH
        for i, r in enumerate(Rasters):

            PSTH[i] = dict()
            raster = Rasters[r]
            mean_psth = np.mean(raster,axis=0)/(1e-3*binsize)
            std_psth = np.sqrt(np.var(raster,axis=0))/(1e-3*binsize)

            sem_psth = std_psth/np.sqrt(float(np.shape(raster)[0]))

            PSTH[i]['mean'] = mean_psth
            PSTH[i]['sem'] = sem_psth

        # Visualize the PSTH
        if plot == True:

            fig = plt.figure()

            scale = 0.1
            y_min = (1.0-scale)*np.min([np.min( \
                np.mean(Rasters[raster_idx],axis=0)/(1e-3*binsize)) \
                for raster_idx in Rasters])
            y_max = (1.0+scale)*np.max([np.max( \
                np.mean(Rasters[raster_idx],axis=0)/(1e-3*binsize)) \
                for raster_idx in Rasters])

            legend = ['event onset']

            xx = np.linspace(window[0], window[1], num=np.diff(window)/binsize)

            plt.plot([0,0],[y_min,y_max], color='k', ls = '--')

            for i, r in enumerate(Rasters):

                plt.plot(xx, PSTH[i]['mean'], color=colors[i], lw=1.5)

            for i, r in enumerate(Rasters):
                #plt.plot(xx, PSTH[i]['mean']+PSTH[i]['sem'], color=colors[i], ls =':')
                #plt.plot(xx, PSTH[i]['mean']-PSTH[i]['sem'], color=colors[i], ls =':')



                plt.fill_between(xx, PSTH[i]['mean']-PSTH[i]['sem'], PSTH[i]['mean']+PSTH[i]['sem'], color=colors[i], alpha=0.2)

                if len(conditions)>0:
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
            print 'Condition %d: %s; %d trials' % (cond+1,str(conditions[cond]),np.shape(Rasters[i])[0])

        return PSTH

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
        for r in conditions:
            condition = conditions[r]
            trials.append([np.all([np.all((features[r] >= condition[r][0], \
                                 features[r] <= condition[r][-1]), axis=0) \
                                 for r in condition], axis=0)])
        return np.transpose(np.atleast_2d(np.squeeze(trials)))

    #-----------------------------------------------------------------------
    def get_spikecounts(self, events, window = 1e-3*np.array([50.0, 100.0])):
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
        for e in range(events.shape[0]):
            spikecounts[e] = np.sum(np.all((spiketimes >= events[e] + window[0],\
                                            spiketimes <= events[e] + window[1]),\
                                            axis=0))
        return spikecounts
