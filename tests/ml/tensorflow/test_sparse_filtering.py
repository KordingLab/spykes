from __future__ import absolute_import

import os

import numpy as np
from nose.tools import (
    assert_raises,
    assert_true,
    assert_false,
    assert_equal,
)

import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.examples.tutorials.mnist import input_data

from spykes.ml.tensorflow.sparse_filtering import SparseFiltering
from spykes.config import get_data_directory

# Keeps the number of training images small to reduce testing time.
NUM_TRAIN = 100


def test_sparse_filtering():
    mnist_path = os.path.join(get_data_directory(), 'mnist/')
    mnist = input_data.read_data_sets(mnist_path, one_hot=False)
    train_images = mnist.train.images[:NUM_TRAIN]

    # Creates a simple model.
    model = ks.models.Sequential([
        ks.layers.Dense(20, input_shape=(28 * 28,), name='a'),
        ks.layers.Dense(20, name='b'),
    ])

    # Checks the four ways to pass layers.
    sf_model = SparseFiltering(model=model)
    assert_equal(len(sf_model.layer_names), len(model.layers))
    with assert_raises(ValueError):
        sf_model = SparseFiltering(model=model, layers=1)
    sf_model = SparseFiltering(model=model, layers='a')
    assert_equal(sf_model.layer_names, ['a'])

    # Checks model compilation.
    sf_model.compile('sgd')
    assert_raises(RuntimeError, sf_model.compile, 'sgd')

    sf_model = SparseFiltering(model=model, layers=['a', 'b'])
    assert_equal(sf_model.layer_names, ['a', 'b'])

    # Checks that the submodels attribute is not available yet.
    with assert_raises(RuntimeError):
        print(sf_model.submodels)

    # Checks getting a submodel.
    with assert_raises(RuntimeError):
        sf_model.get_submodel('a')

    # Checks model freezing.
    sf_model.compile('sgd', freeze=True)
    assert_equal(len(sf_model.submodels), 2)

    # Checks getting an invalid submodel.
    with assert_raises(ValueError):
        sf_model.get_submodel('c')

    # Checks model fitting.
    h = sf_model.fit(x=train_images, epochs=1)
    assert_equal(len(h), 2)  # One history for each layer name.

    # Checks the iterable cleaning part.
    assert_raises(ValueError, sf_model._clean_maybe_iterable_param, ['a'], 'a')

    def _check_works(p):
        cleaned_v = sf_model._clean_maybe_iterable_param(p, '1337')
        assert_equal(len(cleaned_v), 2)

    _check_works('a')
    _check_works(1)
    _check_works(['a', 'b'])
