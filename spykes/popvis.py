import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../mpl_styles/spykes.mplstyle')
    )

class PopVis(object):

    """
    This class implements several conveniences for
    visualizing firing activity of neuron populations

    Parameters
    ----------
    neuron_list: list of NeuroVis objects

    Internal variables
    ------------------
    neuron_list: list of NeuroVis objects
    n_neurons: int

    Callable methods
    ----------------
    get_psths
    plot_heat_map

    Class methods
    -------------
    _sort

    """

    def __init__(self, neuron_list):
        """
        Initialize the object
        """
        self.neuron_list = neuron_list
        self.n_neurons = len(neuron_list)
            
    def get_psths(self, event=None, df=None, conditions=None,
                 window=[-100, 500], binsize=10, event_name=None,
                 conditions_names=None, plot=True, cmap=plt.cm.coolwarm):
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

        event_name: string
            Legend name for event. Default is the actual 'event' name

        conditions_names:
            Legend names for conditions. Default are the unique values in
            df['conditions']

        plot: bool
            Whether to automatically plot or not

        cmap: str
            Colormap for heatmap


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
            
            psth = neuron.get_psth(event=event, df=df, conditions=conditions,
                                    window=window, binsize=binsize, plot=False)

            for cond_id in np.sort(psth['data'].keys()):

                if cond_id not in all_psth['data']:
                    all_psth['data'][cond_id] = list()
 
                all_psth['data'][cond_id].append(psth['data'][cond_id]['mean'])
                
                # TODO: what if cond_id empty for given neuron
        
        for cond_id in np.sort(all_psth['data'].keys()):
            all_psth['data'][cond_id] = np.stack(all_psth['data'][cond_id])

        if plot is True:

            self.plot_heat_map(all_psth, event_name=event, conditions_names=
                                conditions_names, cmap=cmap)

        return all_psth


    def plot_heat_map(self, psth_dict, event_name='event_onset', cond_id=None, 
                        conditions_names=None, sortby=None, 
                        cmap=plt.cm.coolwarm):

        """
        Plots heat map for neuron population

        Parameters
        ----------
        psth_dict : dictionary
            
            With keys: 'event', 'conditions', 'binsize', 'window', and 'data'.
            
            Each entry in psth['data'] is itself a dictionary with keys of
            each cond_id that correspond to the means for that condition

        event_name: str
            Name to appear in the title

        cond_id
            Which psth to plot indicated by the key in all_psth['data'].
            If None then all are plotted.

        conditions_names: str (or list)
            Name(s) to appear in the title

        sortby: str
            None: neuron number
            'rate': sort by firing rate
            'stimulus': sort by preferred stimulus
            'latency': sort by peak latency

        cmap: str
            Colormap for heatmap

        """

        window = psth_dict['window']
        binsize = psth_dict['binsize']
        conditions = psth_dict['conditions']
        event = psth_dict['event']


        if conditions_names is None:
            conditions_names = np.sort(psth_dict['data'].keys()).tolist()

        for i, cond_id in enumerate(np.sort(psth_dict['data'].keys())):

            orig_data = psth_dict['data'][cond_id]

            sort_idx = self._sort(orig_data, sortby)
            data = orig_data[sort_idx,:]

            plt.pcolormesh(data, cmap=cmap)

            plt.xticks([0, -window[0]/binsize, data.shape[1]], \
                [window[0], 0, window[1]])

            unsorted_ylabels = [neuron.name for neuron in self.neuron_list]
            ylabels = [unsorted_ylabels[j] for j in sort_idx]

            plt.yticks(np.arange(data.shape[0])+0.5, ylabels)

            ax = plt.gca()
            ax.invert_yaxis()

            for t in ax.xaxis.get_major_ticks(): 
                t.tick1On = True
                t.tick2On = False 
            for t in ax.yaxis.get_major_ticks(): 
                t.tick1On = False 
                t.tick2On = False 

            plt.xlabel('time [ms]')
            plt.ylabel('Neuron')

            plt.title("Event: %s | Condition: %s" % \
                            (event_name, conditions_names[i]))

            plt.colorbar()
        
            plt.show()

    def _sort(self, data, sortby=None):

        """
        Calculates sort indices for PSTH data given sorting condition

        Parameters
        ----------
        data : 2-D numpy array
            n_neurons x n_bins

        sortby: str
            None: neuron number
            'rate': sort by firing rate
            'stimulus': sort by preferred stimulus
            'latency': sort by peak latency

        Returns
        -------
        sort_idx : numpy array of sorting indices

        """

        if sortby == 'rate':
            avg_rates = np.sum(data, axis=1)
            return np.argsort(avg_rates)

        elif sortby == 'stimulus':
            return np.arange(data.shape[0]) # TODO: change

        elif sortby == 'latency':
            peaks = np.argmax(data, axis=1)
            return np.argsort(peaks)

        else:
            return np.arange(data.shape[0])





