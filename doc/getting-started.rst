===============
Getting Started
===============

What is Spykes?
---------------

Almost any electrophysiology study of neural spiking data relies on a battery of standard analyses. Raster plots and peri-stimulus time histograms aligned to stimuli and behavior provide a snapshot visual description of neural activity. Similarly, tuning curves are the most standard way to characterize how neurons encode stimuli or behavioral preferences. With increasing popularity of population recordings, maximum-likelihood decoders based on tuning models are becoming part of this standard.

Yet, virtually every lab relies on a set of in-house analysis scripts to go from raw data to summaries. We want to improve this status quo in order to enable easier sharing, better reproducibility and fewer bugs.

Spykes is a collection of Python tools to make the visualization and analysis of neural data easy and reproducible.

At present, spykes comes with four classes:

* :class:`NeuroVis` helps you plot beautiful spike rasters and peri-stimulus time histograms (PSTHs).
* :class:`PopVis` helps you plot population summaries of PSTHs as normalized averages or heat maps.
* :class:`NeuroPop` helps you estimate tuning curves of neural populations and decode stimuli from population vectors with maximum-likelihood decoding.
* :class:`STRF` helps you estimate spatiotemporal receptive fields.

Spykes deliberately does not aim to provide tools for spike sorting or file I/O with popular electrophysiology formats, but only aims to fill the missing niche for neural data analysis and easy visualization. For file I/O, see `Neo`_ and `OpenElectrophy`_. For spike sorting, see `Klusta`_.

Installing
----------

For most cases (including following along with the examples) it is sufficient to just install the vanilla version.

Vanilla
~~~~~~~

This installs the current version from PyPi.

.. code-block:: bash

    pip install spykes

Bleeding-Edge
~~~~~~~~~~~~~

This installs the most recent version from Github.

.. code-block:: bash

    pip install git+git://github.com/KordingLab/spykes

Local Version
~~~~~~~~~~~~~

This creates a local copy of the repo, where you can make changes to Spykes that get propagated to your project.

.. code-block:: bash

    git clone http://github.com/KordingLab/spykes  # Clone this somewhere useful
    python spykes/setup.py develop

Datasets
--------

The examples use real datasets. Instructions for downloading these datasets are included in the notebooks. We recommend `deepdish`_ for reading the HDF5 datafile.

.. _OpenElectrophy: http://neuralensemble.org/OpenElectrophy/
.. _Neo: http://neuralensemble.org/neo/
.. _Klusta: http://klusta.readthedocs.io/en/latest/
.. _deepdish: https://github.com/uchicago-cs/deepdish
