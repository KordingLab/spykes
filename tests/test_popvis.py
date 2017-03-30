from nose.tools import assert_true, assert_equal, assert_raises
import numpy as np
import pandas as pd
import matplotlib.pyplot as p
p.switch_backend('Agg')
from spykes.popvis import PopVis  # noqa
from spykes.neurovis import NeuroVis  # noqa


def test_popvis():

    num_spikes = 500
    num_trials = 10

    binsize = 100
    window = [-500, 1500]

    num_neurons = 10
    neuron_list = list()

    for i in range(num_neurons):
        rand_spiketimes = num_trials * np.random.rand(num_spikes)
        neuron_list.append(NeuroVis(rand_spiketimes))

    pop = PopVis(neuron_list)

    df = pd.DataFrame()

    event = 'realCueTime'
    condition_num = 'responseNum'
    condition_bool = 'responseBool'

    start_times = rand_spiketimes[0::num_spikes / num_trials]
    df['trialStart'] = start_times

    df[event] = df['trialStart'] + np.random.rand(num_trials)

    event_times = ((start_times[:-1] + start_times[1:]) / 2).tolist()
    event_times.append(start_times[-1] + np.random.rand())

    df[event] = event_times

    df[condition_num] = np.random.rand(num_trials)
    df[condition_bool] = df[condition_num] < 0.5

    all_psth = pop.get_all_psth(event=event, conditions=condition_bool, df=df,
                                plot=True, binsize=binsize, window=window)

    assert_equal(all_psth['window'], window)
    assert_equal(all_psth['binsize'], binsize)
    assert_equal(all_psth['event'], event)
    assert_equal(all_psth['conditions'], condition_bool)

    for cond_id in all_psth['data'].keys():

        assert_true(cond_id in df[condition_bool])
        assert_equal(all_psth['data'][cond_id].shape[0],
                     num_neurons)
        assert_equal(all_psth['data'][cond_id].shape[1],
                     (window[1] - window[0]) / binsize)

    assert_raises(ValueError, pop.plot_heat_map, all_psth,
                  sortby=range(num_trials - 1))

    pop.plot_population_psth(all_psth=all_psth)
