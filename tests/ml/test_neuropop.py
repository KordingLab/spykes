from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as p
from nose.tools import (
    assert_true,
    assert_equal,
    assert_raises,
)

from spykes.ml.neuropop import NeuroPop
from spykes.utils import train_test_split

np.random.seed(42)
p.switch_backend('Agg')


def test_neuropop():

    np.random.seed(1738)

    num_neurons = 10

    for num_neurons in [1, 10]:
        for tunemodel in ['glm', 'gvm']:
            for i in range(2):

                pop = NeuroPop(tunemodel=tunemodel, n_neurons=num_neurons,
                               verbose=True)

                if i == 0:
                    pop.set_params()
                else:
                    pop.set_params(mu=np.random.randn(),
                                   k0=np.random.randn(),
                                   k=np.random.randn(),
                                   g=np.random.randn(),
                                   b=np.random.randn())

                x, Y, mu, k0, k, g, b = pop.simulate(tunemodel)

                _helper_test_neuropop(pop, num_neurons, x, Y)


def _helper_test_neuropop(pop, num_neurons, x, Y):

    # Splits into training and testing parts.
    x_split, Y_split = train_test_split(x, Y, percent=0.5)
    (x_train, x_test), (Y_train, Y_test) = x_split, Y_split

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

    pop.display(x, Y, 0, xjitter=True, yjitter=True)
