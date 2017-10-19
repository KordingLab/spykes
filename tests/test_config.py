from __future__ import absolute_import

import os
from nose.tools import assert_equal

from spykes import config


def test_get_home_directory():
    # Tests using os.environ.
    assert_equal(config.get_home_directory(), os.environ['HOME'])

    # Tests using os.path.expanduser.
    home_tmp = os.environ['HOME']
    os.environ.pop('HOME')
    assert_equal(config.get_home_directory(), os.path.expanduser('~'))
    os.environ['HOME'] = home_tmp


def test_get_data_directory():
    # Tests the data directory as inferred from the home directory.
    assert_equal(
        config.get_data_directory(),
        os.path.join(config.get_home_directory(), config.DEFAULT_DATA_DIR),
    )

    # Tests the data directory after adding
    os.environ[config.SPYKES_KEY + '_DATA'] = 'data'
    assert_equal(config.get_data_directory(), 'data')
