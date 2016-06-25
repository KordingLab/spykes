# spykes

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/KordingLab/spykes/blob/master/LICENSE)

Spykes is a collection of Python tools to make the visualization and analysis of neural data easy and reproducible.

At present, this project comes with two packages:
- ```neurovis``` helps you plot beautiful spike rasters and peri-stimulus time histograms
- ```neuropop``` helps you estimate tuning curves of neural populations and decode stimuli from population vectors

Documentation, tutorials and examples are coming soon! Check out the notebooks for now.

![](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/figures/psth_PMd_n91.png)

### Installation

Clone the repository.

```bash
$ git clone http://github.com/KordingLab/spykes
```

Install `spykes` using `setup.py` as follows

```bash
$ cd spykes
$ python setup.py develop install
```

### How to use ```neurovis```?

See:
- [crcns_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/crcns_dataset_example.ipynb)
- [reaching_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/reaching_dataset_example.ipynb)

### How to use ```neuropop```?

See:
- [neuropop_simul_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/neuropop_simul_example.ipynb)
- [reaching_dataset_example.ipynb](https://github.com/KordingLab/spykes/blob/master/notebooks_examples/reaching_dataset_example.ipynb)

### Dependencies

So far, you only need ```NumPy``` >= 1.6.1 and ```SciPy``` >= 0.14, which are already distributed with [Anaconda](https://www.continuum.io/downloads) and [Canopy](https://www.enthought.com/products/canopy/).

Some functions are optimized using ```Numba```. We recommend the latest stable version (0.26.0 or later).

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
