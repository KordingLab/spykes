.. spykes documentation master file, created by
   sphinx-quickstart on Tue Nov  1 20:30:09 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spykes
==================================

Almost any electrophysiology study of neural spiking data relies on a battery of standard analyses.
Raster plots and peri-stimulus time histograms aligned to stimuli and behavior provide a snapshot visual description of neural activity. Similarly, tuning curves are the most standard way to characterize how neurons encode stimuli or behavioral preferences. With increasing popularity of population recordings, maximum-likelihood decoders based on tuning models are becoming part of this standard.

Yet, virtually every lab relies on a set of in-house analysis scripts to go from raw data to summaries. We want to improve this status quo in order to enable easier sharing, better reproducibility and fewer bugs.

Spykes is a collection of Python tools to make the visualization and analysis of neural data easy and reproducible.

At present, spykes comes with three classes:

- ``NeuroVis`` helps you plot beautiful spike rasters and peri-stimulus time histograms (PSTHs).
- ``PopVis`` helps you plot population summaries of PSTHs as normalized averages or heat maps.
- ``NeuroPop`` helps you estimate tuning curves of neural populations and decode stimuli from population vectors with maximum-likelihood decoding.

``Spykes`` deliberately does not aim to provide tools for spike sorting or file i/o with popular electrophysiology formats, but only aims to fill the missing niche for neural data analysis and easy visualization. For file i/o, see `Neo <http://neuralensemble.org/neo/>`__ and `OpenElectrophy <http://neuralensemble.org/OpenElectrophy/>`__. For spike sorting, see `Klusta <http://klusta.readthedocs.io/en/latest/>`__.


`[Repository] <https://github.com/KordingLab/spykes/>`__

Contents
==================================

.. toctree::
   :maxdepth: 1

   installation
   tutorial
   auto_examples/index
   contributing
   api

Bugs
=========================

If you find any errors or bugs, please `report it here <https://github.com/KordingLab/spykes/issues>`__. 