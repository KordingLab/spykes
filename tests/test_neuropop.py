from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as p
from nose.tools import (
    assert_true,
    assert_equal,
    assert_raises,
)

from spykes.neuropop import NeuroPop

np.random.seed(42)
p.switch_backend('Agg')


def test_neuropop():

    num_samples = 500
    num_neurons = 10

    for tunemodel in ['glm', 'gvm']:

        pop = NeuroPop(tunemodel=tunemodel, n_neurons=num_neurons)
        pop.set_params()
        x, Y, mu, k0, k, g, b = pop.simulate(tunemodel)

        # Splits into training and testing parts.
        N = int(Y.shape[0] * 0.5)
        idxs = np.arange(Y.shape[0])
        train_idxs, test_idxs =  Y[idxs[:N]], Y[idxs[N:]]
        Y_train, Y_test = Y[idxs[:N]], Y[idxs[N:]]
        x_train, x_test = x[idxs[:N]], x[idxs[N:]]

        pop.fit(x_train, Y_train)

        Yhat_test = pop.predict(x_test)

        assert_equal(Yhat_test.shape[0], x_test.shape[0])
        assert_equal(Yhat_test.shape[1], num_neurons)

        Ynull = np.mean(Y_train, axis=0)

        score = pop.score(Y_test, Yhat_test, Ynull, method='pseudo_R2')
        assert_equal(len(score), num_neurons)

        xhat_test = pop.decode(Y_test)
        assert_equal(xhat_test.shape[0], Y_test.shape[0])

        for method in ['circ_corr', 'cosine_dist']:
            score = pop.score(x_test, xhat_test, method=method)

        pop.display(x, Y, 0)
