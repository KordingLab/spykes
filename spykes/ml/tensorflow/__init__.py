from __future__ import absolute_import

# Checks that the correct version of TensorFlow is installed.
MIN_TF_VERSION = '1.3.0'
try:
    from distutils.version import LooseVersion
    import tensorflow as tf
    assert LooseVersion(tf.__version__) >= LooseVersion(MIN_TF_VERSION)
except ImportError, AssertionError:
    raise RuntimeError('To use the `tensorflow` submodule, your Tensorflow '
                       'distribution must be at least version {version}.'
                       .format(version=MIN_TF_VERSION))

from .sparse_filtering import SparseFiltering

__all__ = ['SparseFiltering']
