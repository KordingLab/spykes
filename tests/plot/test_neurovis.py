from __future__ import absolute_import

import numpy as np
import pandas as pd
import matplotlib.pyplot as p
from nose.tools import (
    assert_true,
    assert_equal,
)

from spykes.plot.neurovis import NeuroVis
p.switch_backend('Agg')


def test_neurovis():

    np.random.seed(1738)

    num_spikes = 500
    num_trials = 10

    binsize = 100
    window = [-500, 1500]

    rand_spiketimes = np.sort(num_trials * np.random.rand(num_spikes))

    neuron = NeuroVis(spiketimes=rand_spiketimes)

    df = pd.DataFrame()

    event = 'realCueTime'
    condition_num = 'responseNum'
    condition_bool = 'responseBool'

    start_times = rand_spiketimes[0::int(num_spikes/num_trials)]
    df['trialStart'] = start_times

    df[event] = df['trialStart'] + np.random.rand(num_trials)

    event_times = ((start_times[:-1] + start_times[1:]) / 2).tolist()
    event_times.append(start_times[-1] + np.random.rand())

    df[event] = event_times

    df[condition_num] = np.random.rand(num_trials)
    df[condition_bool] = df[condition_num] < 0.5

    for cond in [None, condition_bool]:

        raster = neuron.get_raster(event=event, conditions=cond,
                                   df=df, plot=True, binsize=binsize,
                                   window=window)

        neuron.plot_raster(raster, cond_name=raster['conditions'])

        assert_equal(raster['event'], event)
        assert_equal(raster['conditions'], cond)
        assert_equal(raster['binsize'], binsize)
        assert_equal(raster['window'], window)

    total_trials = 0

    for cond_id in raster['data'].keys():

        assert_true(cond_id in df[condition_bool])
        assert_equal(raster['data'][cond_id].shape[1],
                     (window[1] - window[0]) / binsize)
        total_trials += raster['data'][cond_id].shape[0]

    assert_equal(total_trials, num_trials)

    psth = neuron.get_psth(event=event, conditions=condition_bool, df=df,
                           plot=True, binsize=binsize, window=window)

    neuron.plot_psth(psth=psth, ylim=np.random.randn(2).tolist())

    assert_equal(psth['window'], window)
    assert_equal(psth['binsize'], binsize)
    assert_equal(psth['event'], event)
    assert_equal(psth['conditions'], condition_bool)

    for cond_id in psth['data'].keys():

        assert_true(cond_id in df[condition_bool])
        assert_equal(psth['data'][cond_id]['mean'].shape[0],
                     (window[1] - window[0]) / binsize)
        assert_equal(psth['data'][cond_id]['sem'].shape[0],
                     (window[1] - window[0]) / binsize)

    neuron.get_spikecounts(event=event, df=df, window=[0, num_trials])
