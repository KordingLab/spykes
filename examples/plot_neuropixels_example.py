"""
===================
Neuropixels Example
===================

Use spykes to analyze data from UCL's Neuropixels

"""
# Authors: Mayank Agrawal <mayankagrawal96@gmail.com>
#
# License: MIT

########################################################

import numpy as np
import pandas as pd
import scipy.io
from spykes.neurovis import NeuroVis
from spykes.popvis import PopVis
import matplotlib.pyplot as plt

plt.style.use('seaborn-ticks')

########################################################
# Neuropixels
# -----------------------------
# Neuropixels is a new recording technique by UCL's `Cortex Lab
# <http://www.ucl.ac.uk/neuropixels>`__ that is able to measure data from
# hundreds of neurons. Below we show how this data can be worked with in Spykes
#
# 0 Download Data
# -----------------------------
#
# Download all data `here <http://data.cortexlab.net/dualPhase3/data/>`__.
#
# 1 Read In Data
# -----------------------------

folder_names = ['posterior', 'frontal']
Fs = 30000.0

striatum = list()
motor_ctx = list()
thalamus = list()
hippocampus = list()
visual_ctx = list()

# a lot of this code is adapted from Cortex Lab's MATLAB script
# see here: http://data.cortexlab.net/dualPhase3/data/script_dualPhase3.m

for name in folder_names:

    clusters = np.squeeze(np.load(name + '/spike_clusters.npy'))
    spike_times = (np.squeeze(np.load(name + '/spike_times.npy'))) / Fs
    spike_templates = (np.squeeze(np.load(name + '/spike_templates.npy')))

    temps = (np.squeeze(np.load(name + '/templates.npy')))
    winv = (np.squeeze(np.load(name + '/whitening_mat_inv.npy')))
    y_coords = (np.squeeze(np.load(name + '/channel_positions.npy')))[:, 1]

    # frontal times need to align with posterior
    if (name == 'frontal'):
        time_correction = np.load('time_correction.npy')
        spike_times *= time_correction[0]
        spike_times += time_correction[1]

    data = np.recfromcsv(name + '/cluster_groups.csv', delimiter='\t')
    cids = np.array([x[0] for x in data])
    cfg = np.array([x[1] for x in data])

    # find good clusters and only use those spikes
    good_clusters = cids[cfg == 'good']
    good_indices = (np.in1d(clusters, good_clusters))

    real_spikes = spike_times[good_indices]
    real_clusters = clusters[good_indices]
    real_spike_templates = spike_templates[good_indices]

    # find how many spikes per cluster and then order spikes by which cluster
    # they are in

    counts_per_cluster = np.bincount(real_clusters)

    sort_idx = np.argsort(real_clusters)
    sorted_clusters = real_clusters[sort_idx]
    sorted_spikes = real_spikes[sort_idx]
    sorted_spike_templates = real_spike_templates[sort_idx]

    # find depth for each spike
    # this is translated from Cortex Lab's MATLAB code
    # for more details, check out the original code here:
    # https://github.com/cortex-lab/spikes/blob/master/analysis/templatePositionsAmplitudes.m

    temps_unw = np.zeros(temps.shape)
    for t in range(temps.shape[0]):
        temps_unw[t, :, :] = np.dot(temps[t, :, :], winv)

    temp_chan_amps = np.ptp(temps_unw, axis=1)
    temps_amps = np.max(temp_chan_amps, axis=1)
    thresh_vals = temps_amps * 0.3

    thresh_vals = [thresh_vals for i in range(temp_chan_amps.shape[1])]
    thresh_vals = np.stack(thresh_vals, axis=1)

    temp_chan_amps[temp_chan_amps < thresh_vals] = 0

    y_coords = np.reshape(y_coords, (y_coords.shape[0], 1))
    temp_depths = np.sum(
        np.dot(temp_chan_amps, y_coords), axis=1) / (np.sum(temp_chan_amps,
                                                     axis=1))

    sorted_spike_depths = temp_depths[sorted_spike_templates]

    # create neurons and find region

    accumulator = 0

    for idx, count in enumerate(counts_per_cluster):

        if count > 0:

            spike_times = sorted_spikes[accumulator:accumulator + count]
            neuron = NeuroVis(spike_times=spike_times, name='%d' % (idx))
            cluster_depth = np.mean(
                sorted_spike_depths[accumulator:accumulator + count])

            if name == 'frontal':

                if (cluster_depth > 0 and cluster_depth < 1550):
                    striatum.append(neuron)
                elif (cluster_depth > 1550 and cluster_depth < 3840):
                    motor_ctx.append(neuron)

            elif name == 'posterior':

                if (cluster_depth > 0 and cluster_depth < 1634):
                    thalamus.append(neuron)
                elif (cluster_depth > 1634 and cluster_depth < 2797):
                    hippocampus.append(neuron)
                elif (cluster_depth > 2797 and cluster_depth < 3840):
                    visual_ctx.append(neuron)

            accumulator += count


print("Striatum (n = %d)" % len(striatum))
print("Motor Cortex (n = %d)" % len(motor_ctx))
print("Thalamus (n = %d)" % len(thalamus))
print("Hippocampus (n = %d)" % len(hippocampus))
print("Visual Cortex (n = %d)" % len(visual_ctx))

########################################################
# 2 Create Data Frame
# -----------------------------

df = pd.DataFrame()

raw_data = scipy.io.loadmat('experiment1stimInfo.mat')

df['start'] = np.squeeze(raw_data['stimStarts'])
df['stop'] = np.squeeze(raw_data['stimStops'])
df['stimulus'] = np.squeeze(raw_data['stimIDs'])

print(df.head())

########################################################
# 3 Start Plotting
# -----------------------------
# 3.1 Striatum
# ~~~~~~~~~~~~

pop = PopVis(striatum, name='Striatum')

fig = plt.figure(figsize=(30, 20))

all_psth = pop.get_all_psth(
    event='start', df=df, conditions='stimulus', plot=False, binsize=100,
    window=[-500, 2000])

pop.plot_heat_map(all_psth, cond_id=[
                  2, 7, 13], sortorder='descend', neuron_names=False)

########################################################

pop.plot_population_psth(all_psth=all_psth, cond_id=[1, 7, 12])

########################################################
# 3.2 Frontal
# ~~~~~~~~~~~~

pop = PopVis(striatum + motor_ctx, name='Frontal')

fig = plt.figure(figsize=(30, 20))

all_psth = pop.get_all_psth(
    event='start', df=df, conditions='stimulus', plot=False, binsize=100,
    window=[-500, 2000])

pop.plot_heat_map(
    all_psth, cond_id=[2, 7, 13], sortorder='descend', neuron_names=False)

########################################################

pop.plot_population_psth(all_psth=all_psth, cond_id=[1, 7, 12])

########################################################
# 3.3 All Neurons
# ~~~~~~~~~~~~

pop = PopVis(striatum + motor_ctx + thalamus + hippocampus + visual_ctx)

fig = plt.figure(figsize=(30, 20))

all_psth = pop.get_all_psth(
    event='start', df=df, conditions='stimulus', plot=False, binsize=100,
    window=[-500, 2000])

pop.plot_heat_map(
    all_psth, cond_id=[2, 7, 13], sortorder='descend', neuron_names=False)

########################################################

pop.plot_population_psth(all_psth=all_psth, cond_id=[1, 7, 12])

########################################################
# 3.4 Striatum vs. Motor Cortex
# ~~~~~~~~~~~~

striatum_pop = PopVis(striatum, name='Striatum')
motor_ctx_pop = PopVis(motor_ctx, name='Motor Cortex')

striatum_psth = striatum_pop.get_all_psth(
    event='start', df=df, conditions='stimulus', plot=False, binsize=100,
    window=[-500, 2000])
motor_ctx_psth = motor_ctx_pop.get_all_psth(
    event='start', df=df, conditions='stimulus', plot=False, binsize=100,
    window=[-500, 2000])

########################################################

striatum_pop.plot_population_psth(all_psth=striatum_psth, cond_id=[1, 7, 12])

########################################################

motor_ctx_pop.plot_population_psth(all_psth=motor_ctx_psth, cond_id=[1, 7, 12])
