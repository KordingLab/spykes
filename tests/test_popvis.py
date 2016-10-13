from spykes.popvis import PopVis
from spykes.neurovis import NeuroVis
from nose.tools import assert_true, assert_equal, assert_raises
import numpy as np
import pandas as pd


def test_popvis():

    num_neurons = 10
    num_spikes = 500
    num_trials = 10
    neuron_list = list()

    for i in range(num_neurons):
        rand_spiketimes = num_trials*np.random.rand(num_spikes)
        neuron_list.append(NeuroVis(rand_spiketimes))

    pop = PopVis(neuron_list)

    df = pd.DataFrame()

    event = 'realCueTime'
    condition_num = 'responseNum'
    condition_bool = 'responseBool'

    df['trialStart'] = num_trials*np.sort(np.random.rand(num_trials))
    df[event] = df['trialStart']+ np.random.rand(num_trials)
    df[condition_num] = np.random.rand(num_trials)
    df[condition_bool] = df[condition_num] < 0.5


    all_psth = pop.get_all_psth(event=event, conditions=condition_bool, df=df, 
        plot=False)

    assert_raises(ValueError, pop.plot_heat_map, all_psth, 
        sortby=range(num_trials-1))

