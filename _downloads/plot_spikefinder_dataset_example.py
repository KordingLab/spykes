"""
===========================
Spikefinder DataSet Example
===========================

A demonstration of Spyke's functionality on the Spikefinder dataset.
"""
# Authors: Ben Bolte <ben@bolte.cc>
#
# License: BSD (3-clause)

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from spykes.io.datasets import load_spikefinder_data
