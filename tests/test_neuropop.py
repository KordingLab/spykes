import matplotlib.pyplot as p
p.switch_backend('Agg')
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
