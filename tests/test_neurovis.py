from spykes.neurovis import NeuroVis
from nose.tools import assert_true, assert_equal, assert_raises
import numpy as np
import pandas as pd



def test_neurovis():

    num_spikes = 500
    num_trials = 10

    rand_spiketimes = num_trials*np.random.rand(num_spikes)
    neuron = NeuroVis(spiketimes=rand_spiketimes)

    df = pd.DataFrame()

    event = 'realCueTime'
    condition_num = 'responseNum'
    condition_bool = 'responseBool'

    df['trialStart'] = num_trials*np.sort(np.random.rand(num_trials))
    df[event] = df['trialStart']+ np.random.rand(num_trials)
    df[condition_num] = np.random.rand(num_trials)
    df[condition_bool] = df[condition_num] < 0.5

    raster = neuron.get_raster(event=event, conditions=condition_bool, df=df, 
        plot=False)

    assert_equal(raster['event'], event)
    assert_equal(raster['conditions'], condition_bool)

    psth = neuron.get_psth(event=event, conditions=condition_bool, df=df, 
        plot=False)

    assert_equal(psth['event'], event)
    assert_equal(psth['conditions'], condition_bool)

    spikecounts = neuron.get_spikecounts(event=event, df=df, 
        window=[0,num_trials])






