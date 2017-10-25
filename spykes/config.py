from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Environment variables are formulated using SPYKES_KEY_%s.
SPYKES_KEY = 'SPYKES'

# The default data path.
DEFAULT_DATA_DIR = '.spykes'

# Defines the default colors for the population plot.
DEFAULT_POPULATION_COLORS = [
    '#F5A21E',
    '#134B64',
    '#EF3E34',
    '#02A68E',
    '#FF07CD',
]


def get_home_directory():
    '''Returns the home directory, as a string.

    The home directory is either the :data:`HOME` environment variable, or the
    directory pointed to by :data:`~`, or the :data:`/tmp` directory if neither
    of those have write access.

    Returns:
        str: The path to the home directory.
    '''
    if 'HOME' in os.environ and os.access(os.environ['HOME'], os.W_OK):
        return os.environ['HOME']
    elif os.access(os.path.expanduser('~'), os.W_OK):
        return os.path.expanduser('~')
    else:
        return '/tmp'  # Default is to return a temp directory.


def get_data_directory():
    '''Returns the home directory for Spykes data.

    By default, this points to :data:`~/.spykes`. This can be overridden by
    setting the :data:`SPYKES_DATA` environment variable to point to the
    directory of your choice.

    Returns:
        str: The path to the data directory.
    '''
    data_key = '{prefix}_DATA'.format(prefix=SPYKES_KEY)

    if data_key not in os.environ:
        home = get_home_directory()
        dir_path = os.path.join(home, DEFAULT_DATA_DIR)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path
    else:
        return os.environ[data_key]
