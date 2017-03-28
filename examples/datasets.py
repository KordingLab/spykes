"""
Functions to fetch datasets
"""

import os
import shutil


def fetch_reward_data(dpath='~/spykes_data/'):
    """
    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded
    


    """

    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)