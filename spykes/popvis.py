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
    (see NeuroVis class in spykes.neurovis)

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
    _get_sort_indices

    """

    def __init__(self, neuron_list):
        """
        Initialize the object
        """
        self.neuron_list = neuron_list
        self.n_neurons = len(neuron_list)
            
    def get_psths(self, event=None, df=None, conditions=None,
                window=[-100, 500], binsize=10, conditions_names=None,
                plot=True, colors=[plt.cm.Blues, plt.cm.Reds, 
                plt.cm.Greens]):
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

        colors: list of Colormap objects for heatmap

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
                        

        for cond_id in np.sort(all_psth['data'].keys()):
            all_psth['data'][cond_id] = np.stack(all_psth['data'][cond_id])

        if plot is True:

            self.plot_heat_map(all_psth, conditions_names=conditions_names, 
                colors=colors)

        return all_psth


    def plot_heat_map(self, psth_dict, cond_id=None, 
                    conditions_names=None, sortby=None, sortorder='descend', 
                    normalize=None, colors=[plt.cm.Blues, plt.cm.Reds, 
                    plt.cm.Greens]):

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

        sortby: str
            None: neuron number
            'rate': sort by firing rate
            'stimulus': sort by preferred stimulus
            'latency': sort by peak latency

        sortorder: direction to sort in
            'descend'
            'ascend'

        normalize: str
            'all' : divide all PSTHs by highest peak firing rate in all neurons
            'each' : divide each PSTH by its own peak firing rate

        colors: list of Colormap objects for heatmap

        """

        window = psth_dict['window']
        binsize = psth_dict['binsize']
        conditions = psth_dict['conditions']
        event = psth_dict['event']


        if conditions_names is None:
            conditions_names = np.sort(psth_dict['data'].keys()).tolist()

        if cond_id is None:
            keys = np.sort(psth_dict['data'].keys())
        else:
            keys = cond_id

        for i, cond_id in enumerate(keys):

            orig_data = psth_dict['data'][cond_id]

            sort_idx = self._get_sort_indices(orig_data, sortby=sortby, sortorder=sortorder)

            data = orig_data[sort_idx,:]

            plt.pcolormesh(data, cmap=colors[i%len(colors)])

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

            plt.title("%s: %s" % \
                            (conditions, conditions_names[i]))

            plt.colorbar()
        
            plt.show()

    def _get_sort_indices(self, data, sortby=None, sortorder='descend'):

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

        sortorder: direction to sort in
            'descend'
            'ascend'

        Returns
        -------
        sort_idx : numpy array of sorting indices

        """

        if sortby == 'rate':
            avg_rates = np.sum(data, axis=1)
            sort_idx = np.argsort(avg_rates)

        elif sortby == 'stimulus':
            sort_idx = np.arange(data.shape[0]) # TODO: change

        elif sortby == 'latency':
            peaks = np.argmax(data, axis=1)
            sort_idx = np.argsort(peaks)

        else:
            sort_idx = np.arange(data.shape[0])


        if sortorder == 'ascend':
            return sort_idx
        else:
            return sort_idx[::-1]



