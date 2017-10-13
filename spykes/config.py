'''Handles system configurations (e.g. data loading).'''

import os

# Environment variables are formulated using SPYKES_KEY_%s.
SPYKES_KEY = 'SPYKES_'

# The default data path.
DEFAULT_DATA_DIR = '.spykes'


def get_home_directory():
    '''Returns the home directory, as a string.'''
    if 'HOME' in os.environ and os.access(os.environ['HOME'], os.W_OK):
        return os.environ['HOME']
    elif os.access(os.path.expanduser('~'), os.W_OK):
        return os.path.expanduser('~')
    else:
        return '/tmp'  # Default is to return a temp directory.


def get_data_directory():
    '''Returns the home directory for Spykes data.
    Returns:
        str, the path to the data directory.
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
