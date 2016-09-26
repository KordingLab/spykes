# spykes

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/KordingLab/spykes/blob/master/LICENSE) [![Join the chat at https://gitter.im/KordingLab/spykes](https://badges.gitter.im/KordingLab/spykes.svg)](https://gitter.im/KordingLab/spykes?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Almost any electrophysiology study of awake behaving animals relies on a battery of standard analyses.
Raster plots and peri-stimulus time histograms aligned to stimuli and behavior provide a snapshot visual description of neural activity. Similarly, tuning curves are the most standard way to characterize how neurons encode stimuli or behavioral preferences. With increasing popularity of population recordings, maximum-likelihood decoders based on tuning models are becoming part of this standard.

Yet, virtually every lab relies on a set of in-house analysis scripts to go from raw data to summaries. We want to change this status quo in order to enable easier sharing, better reproducibility and fewer bugs.

Spykes is a collection of Python tools to make the visualization and analysis of neural data easy and reproducible.

At present, spykes comes with two classes:
- ```NeuroVis``` helps you plot beautiful spike rasters and peri-stimulus time histograms (PSTHs).
- ```PopVis``` helps you plot population summaries of PSTHs as normalized averages or heat maps.
- ```NeuroPop``` helps you estimate tuning curves of neural populations and decode stimuli from population vectors with maximum-likelihood decoding.

```Spykes``` deliberately does not aim to provide tools for spike sorting or file i/o with popular electrophysiology formats, but only aims to fill the missing niche for neural data analysis and easy visualization. For file i/o, see [Neo](http://neuralensemble.org/neo/) and [OpenElectrophy](http://neuralensemble.org/OpenElectrophy/). For spike sorting, see [Klusta](http://klusta.readthedocs.io/en/latest/).

Documentation, tutorials and examples are coming soon! Check out the notebooks for now.

![](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/figures/psth_PMd_n91.png)

### Installation

Clone the repository.

```bash
$ git clone http://github.com/KordingLab/spykes
```

Install `spykes` using `pip` as follows

```bash
$ cd spykes
$ pip install -e ./
```

### How to use ```NeuroVis```?

See:
- [crcns_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/crcns_dataset_example.ipynb)
- [reaching_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/reaching_dataset_example.ipynb)

### How to use ```PopVis```?
See:
- [popvis_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/popvis_example.ipynb)

### How to use ```NeuroPop```?

See:
- [neuropop_simul_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/neuropop_simul_example.ipynb)
- [reaching_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/reaching_dataset_example.ipynb)

### Dependencies

Already distributed with [Anaconda](https://www.continuum.io/downloads) and [Canopy](https://www.enthought.com/products/canopy/).
- ```NumPy``` >= 1.6.1
- ```SciPy``` >= 0.14
- ```Matplotlib``` >= 1.5

We also use ```Numba``` to optimize certain functions. We recommend the latest stable version (>= 0.26.0).

```bash
$ pip install numba
```

### Datasets

The example notebooks use two real datasets. Instructions for downloading these datasets are included in the notebooks. We recommend [deepdish](https://github.com/uchicago-cs/deepdish) for reading the HDF5 datafile.

### Authors

* [Pavan Ramkumar](http:/github.com/pavanramkumar)
* [Hugo Fernandes](http:/github.com/hugoguh)

### Acknowledgments

* [Konrad Kording](http://kordinglab.com) for funding and support
