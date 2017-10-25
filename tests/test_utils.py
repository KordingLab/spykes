from __future__ import absolute_import

import os
from nose.tools import assert_equal
import numpy as np

from spykes import utils


def test_train_test_split():
    x, y, z = np.zeros((10,)), np.zeros((10, 10)), np.zeros((10, 10, 10))

    def _check(xyz, xyzs):
        for s, (s_train, s_test) in zip(xyz, xyzs):
            assert_equal(s_train.shape[0], 7)
            assert_equal(s_test.shape[0], 3)
            assert_equal(np.ndim(s), np.ndim(s_train))
            assert_equal(np.ndim(s), np.ndim(s_test))

    # Checks number-wise and percent-wise.
    _check([x, y, z], utils.train_test_split(x, y, z, n=3))
    _check([x, y, z], utils.train_test_split(x, y, z, percent=0.3))
