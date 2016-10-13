from spykes.neurovis import neurovis
from nose.tools import assert_true, assert_equal, assert_raises
import numpy as np



def test():

    num_spikes = 500
    num_trials = 10

    rand_spiketimes = num_trials*np.random.rand(num_spikes)
    neuron = NeuroVis(spiketimes=rand_spiketimes)

    df = dict()

    event = 'realCueTime'
    condition = 'responseBool'

    df['trialStart'] = num_trials*np.sort(np.random.rand(num_trials))
    df[event] = df['trialStart']+ np.random.rand(num_trials)
    df[condition] = np.random.rand(num_trials) < .5

    raster = neuron.get_raster(event=event, condition=condition, df=df, 
        plot=False)

    assert_equal(raster['event'], event)
    assert_equal(raster['condition'], condition)

    psth = neuron.get_psth(event=event, condition=condition, df=df, 
        plot=False)

    assert_equal(psth['event'], event)
    assert_equal(psth['condition'], condition)





