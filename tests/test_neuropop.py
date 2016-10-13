from spykes.neuropop import NeuroPop
from nose.tools import assert_true, assert_equal, assert_raises
import numpy as np


def test_neuropop():

    num_samples = 500
    num_neurons = 10

    for tunemodel in ['glm', 'gvm']:

        pop = NeuroPop(tunemodel=tunemodel, n_neurons=num_neurons)
        pop.set_params()
        x, Y, mu, k0, k, g, b = pop.simulate(tunemodel)

        from sklearn.cross_validation import train_test_split
        Y_train, Y_test, x_train, x_test = train_test_split(Y, x, test_size=0.5, 
            random_state=42)

        pop.fit(x_train, Y_train)

        Yhat_test = pop.predict(x_test)

        Ynull = np.mean(Y_train, axis=0)
        pop.score(Y_test, Yhat_test, Ynull, method='pseudo_R2')

        xhat_test = pop.decode(Y_test)

        for method in ['circ_corr', 'cosine_dist']:

            pop.score(x_test, xhat_test, method=method)