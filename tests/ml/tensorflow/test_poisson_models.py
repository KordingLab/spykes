from __future__ import absolute_import

import os
import uuid

import numpy as np
from nose.tools import (
    assert_raises,
    assert_false,
)

from tempfile import TemporaryFile

from tensorflow import keras as ks

from spykes.ml.tensorflow.poisson_models import PoissonLayer
from spykes.io.datasets import load_reaching_rates


def _build_model(model_type, num_features, num_neurons):
    i = ks.layers.Input(shape=(num_features,))
    x = PoissonLayer(model_type, num_neurons)(i)
    model = ks.models.Model(inputs=i, outputs=x)
    model.compile(optimizer='sgd', loss='poisson')
    return model


def test_poisson_layer():
    with assert_raises(ValueError):
        PoissonLayer('invalid_type', 1)

    with assert_raises(AssertionError):
        i = ks.layers.Input(shape=(1,2))
        x = PoissonLayer('glm', 3, num_features=2)(i)

    # Loads the reaching dataset (with default parameters).
    x, y = load_reaching_rates()
    num_features, num_neurons = x.shape[1], y.shape[1]

    for model_type in ('gvm', 'glm'):
        model = _build_model(model_type, num_features, num_neurons)
        p = model.predict(x)
        h = model.fit(x, y, epochs=1)  # , verbose=0)
        assert_false(np.any(np.isnan(h.history['loss'])))

        if model_type == 'gvm':
            tmploc = '/tmp/{}'.format(uuid.uuid4)
            model.save(tmploc)
            os.remove(tmploc)
