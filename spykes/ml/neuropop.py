from __future__ import absolute_import

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from .. import utils


class NeuroPop(object):
    '''Implements conveniences for plotting, fitting and decoding.

    Implements convenience methods for plotting, fitting and decoding
    population tuning curves. We allow the fitting of two classes of parametric
    tuning curves.

    Two types of models are available. `The Generalized von Mises model by
    Amirikan & Georgopulos (2000) <http://brain.umn.edu/pdfs/BA118.pdf>`_ is
    defined by

    .. math::

        f(x) = b + g * exp(k * cos(x - mu))

        f(x) = b + g * exp(k1 * cos(x) + k2 * sin(x))

    The Poisson generalized linear model is defined by

    .. math::

        f(x) = exp(k0 + k * cos(x - mu))

        f(x) = exp(k0 + k1 * cos(x) + k2 * sin(x))

    Args:
        tunemodel (str): Can be either :data:`gvm`, the Generalized von Mises
            model, or :data:`glm`, the Poisson generalized linear model.
        n_neurons (float): Number of neurons in the population.
        random_state (int): Seed for :data:`numpy.random`.
        eta (float): Linearizes the exponent above :data:`eta`.
        learning_rate (float): The learning rate for fitting.
        convergence_threshold (float): The convergence threshold.
        maxiter (float): Max number of iterations.
        n_repeats (float): Number of repetitions.
        verbose (bool): Whether to print convergence and loss at each
            iteration.
    '''

    def __init__(self,
                 tunemodel='glm',
                 n_neurons=100,
                 random_state=1,
                 eta=0.4,
                 learning_rate=2e-1,
                 convergence_threshold=1e-5,
                 maxiter=1000,
                 n_repeats=1,
                 verbose=False):
        self.tunemodel = tunemodel
        self.n_neurons = n_neurons

        # Assign random tuning parameters
        # --------------------------------
        self.mu_ = np.zeros(n_neurons)
        self.k0_ = np.zeros(n_neurons)
        self.k_ = np.zeros(n_neurons)
        self.k1_ = np.zeros(n_neurons)
        self.k2_ = np.zeros(n_neurons)
        self.g_ = np.zeros(n_neurons)
        self.b_ = np.zeros(n_neurons)
        np.random.seed(random_state)
        self.set_params(tunemodel, list(range(self.n_neurons)))

        # Assign optimization parameters
        # -------------------------------
        self.eta = eta
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.maxiter = maxiter
        self.n_repeats = n_repeats

        self.verbose = verbose

    def default_mu(self, n):
        return np.pi * (2.0 * np.random.rand(n) - 1.0)

    def default_k0(self, n, tunemodel):
        return np.random.rand(n) if tunemodel == 'glm' else np.zeros(n)

    def default_k(self, n):
        return 20.0 * np.random.rand(n)

    def default_g(self, n, tunemodel):
        return 5.0 * np.random.rand(n) if tunemodel == 'gvm' else np.ones(n)

    def default_b(self, n, tunemodel):
        return 10.0 * np.random.rand(n) if tunemodel == 'gvm' else np.zeros(n)

    def set_params(self, tunemodel=None, neurons=None, mu=None, k0=None,
                   k=None, g=None, b=None):
        '''A function that sets tuning curve parameters as specified.

        If any of the parameters is None, it is randomly initialized for all
        neurons.

        Args:
            tunemodel (str): Either 'gvm' or 'glm'.
            neurons (list): A list of integers which specifies the subset of
                neurons to set.
            mu (float): :data:`len(neurons) x 1`, feature of interest.
            k0 (float): :data:`len(neurons) x 1`, baseline.
            k (float): :data:`len(neurons) x 1`, gain.
            g (float): :data:`len(neurons) x 1`, gain.
            b (float): :data:`len(neurons) x 1`, baseline.
        '''
        # Does an argument check on "tunemodel".
        if tunemodel is not None and tunemodel not in ('gvm', 'glm'):
            raise ValueError('Invalid value for `tunemodel`: Expected either '
                             '"gvm" or "glm", but got "{}".'.format(tunemodel))

        # Disambiguates some paremeters to use.
        model = self.tunemodel if tunemodel is None else tunemodel
        idx = list(range(self.n_neurons)) if neurons is None else neurons
        n_neurons = len(idx) if hasattr(idx, '__len__') else 1

        # Updates the model's parameters to be the specified values.
        self.mu_[idx] = self.default_mu(n_neurons) if mu is None else mu
        self.k0_[idx] = self.default_k0(n_neurons, model) if k0 is None else k0
        self.g_[idx] = self.default_g(n_neurons, model) if g is None else g
        self.b_[idx] = self.default_b(n_neurons, model) if b is None else b
        self.k_[idx] = self.default_k(n_neurons) if k is None else k
        self.k1_[idx] = self.k_[idx] * np.cos(self.mu_[idx])
        self.k2_[idx] = self.k_[idx] * np.sin(self.mu_[idx])

    def _tunefun(self, x, k0, k1, k2, g, b):
        '''Defines the tuning function as specified in self.tunemodel.

        Args:
            x (float): :data:`n_samples x 1`, feature of interest.
            k0 (float): :data:`n_neurons x 1`, baseline.
            k1 (float): :data:`n_neurons x 1`, convenience parameter.
            k2 (float): :data:`n_neurons x 1`, convenience parameter.
            g (float): :data:`n_neurons x 1`, gain.
            b (float): :data:`n_neurons x 1`, baseline.

        Returns
            array: :data:`n_samples x 1` array, the firing rates.
        '''
        y = b + g * utils.slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x),
                                   self.eta)
        return y

    def _loss(self, x, y, k0, k1, k2, g, b):
        '''The loss function, negative Poisson log likelihood.

        This is the negative Poisson log likelihood under the von Mises tuning
        model.

        Args:
            x (float): :data:`n_samples x 1` (encoding) or
                a scalar (decoding), feature of interest.
            y (float): :data:`n_samples x 1` (encoding) or
                :data:`n_neurons x 1` (decoding), firing rates.
            mu (float): :data:`n_neurons x 1`, preferred feature
                :data:`[-pi, pi]`.
            k0 (float): :data:`n_neurons x 1`, baseline.
            k1 (float): :data:`n_neurons x 1`, convenience parameter.
            k2 (float): :data:`n_neurons x 1`, convenience parameter.
            g (float): :data:`n_neurons x 1`, gain.
            b (float): :data:`n_neurons x 1`, baseline.

        Returns:
            scalar float: The loss, a scalar float.
        '''
        lmb = self._tunefun(x, k0, k1, k2, g, b)
        J = np.sum(lmb) - np.sum(y * lmb)
        return J

    def _grad_theta_loss(self, tunemodel, x, y, k0, k1, k2, g, b):
        '''The gradient of the loss function for the parameters of the model.

        Args:
            x (float array): :data:`n_samples x 1`, feature of interest.
            y (float array): :data:`n_samples x 1`, firing rates.
            k0 (float array): :data:`n_neurons x 1`, baseline.
            k1 (float array): :data:`n_neurons x 1`, convenience parameter.
            k2 (float array): :data:`n_neurons x 1`, convenience parameter.
            g (float): Scalar, gain.
            b (float): Scalar, baseline.

        Returns:
            tuple: The gradients of the loss with respect to each parameter.

            * :data:`grad_k0`: scalar
            * :data:`grad_k1`: scalar
            * :data:`grad_k2`: scalar
            * :data:`grad_g`: scalar
            * :data:`grad_b`: scalar
        '''
        lmb = self._tunefun(x, k0, k1, k2, g, b)

        n_samples = np.float(x.shape[0])
        grad_k1 = 1. / n_samples * np.sum(g * utils.grad_slow_exp(k0 + k1 *
                                          np.cos(x) + k2 * np.sin(x),
                                          self.eta) * np.cos(x) * (
                                          1 - y / lmb))
        grad_k2 = 1. / n_samples * np.sum(g * utils.grad_slow_exp(k0 + k1 *
                                          np.cos(x) + k2 * np.sin(x),
                                          self.eta) * np.sin(x) *
                                          (1 - y / lmb))
        if tunemodel == 'glm':
            grad_k0 = 1. / n_samples * np.sum(utils.grad_slow_exp(k0 + k1 *
                                              np.cos(x) + k2 * np.sin(x),
                                              self.eta) * (1 - y / lmb))
            grad_g = 0.0
            grad_b = 0.0
        elif tunemodel == 'gvm':
            grad_k0 = 0.0
            grad_g = 1. / n_samples *\
                np.sum(utils.slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x),
                                      self.eta) * (1 - y / lmb))
            grad_b = 1. / n_samples * np.sum((1 - y / lmb))

        return grad_k0, grad_k1, grad_k2, grad_g, grad_b

    def _grad_x_loss(self, x, y, k0, k1, k2, g, b):
        '''The gradient of the loss function with respect to X.

        Args:
            x (float): Scalar, feature of interest.
            y (float array): :data:`n_neurons x 1`, firing rates.
            k0 (float array): :data:`n_neurons x 1`, baseline.
            k1 (float array): :data:`n_neurons x 1`, convenience parameter.
            k2 (float array): :data:`n_neurons x 1`, convenience parameter.
            g (float array): :data:`n_neurons x 1`, gain.
            b (float array): :data:`n_neurons x 1`, baseline.

        Returns:
            array: :data:`grad_x`, the gradient with respect to :data:`x`.
        '''
        n_neurons = np.float(self.n_neurons)

        lmb = self._tunefun(x, k0, k1, k2, g, b)
        grad_x = 1. / n_neurons * np.sum(g * utils.grad_slow_exp(k0 + k1 *
                                                                 np.cos(x) +
                                                                 k2 *
                                                                 np.sin(x),
                                                                 self.eta) *
                                         (k2 * np.cos(x) - k1 * np.sin(x)) *
                                         (1 - y / lmb))
        return grad_x

    def simulate(self, tunemodel, n_samples=500, winsize=200):
        '''Simulates firing rates from a neural population.

        Simulates firing rates from a neural population by randomly sampling
        circular variables (feature of interest), as well as parameters
        (:data:`mu`, :data:`k0`, :data:`k`, :data:`g`, :data:`b`).

        Args:
            tunemodel (str): Can be either :data:`gvm`, the Generalized von
                Mises model, or :data:`glm`, the Poisson generalized linear
                model.
            n_samples (int): Number of samples required.
            winsize (float): Time interval in which to simulate spike counts,
                milliseconds.

        Returns:
            tuple: The simulation parameters.

            * `x`, :data:`n_samples x 1` array, features of interest
            * `Y`, :data:`n_samples x n_neurons` array, population activity
            * `mu`, :data:`n_neurons x 1` array, preferred feature,
              :data:`[-pi, pi]`; `k0`, :data:`n_neurons x 1`, baseline
            * `k`, :data:`n_neurons x 1` array, shape (width)
            * `g`, :data:`n_neurons x 1` array, gain
            * `b`, :data:`n_neurons x 1`, baseline
        '''

        # Sample parameters randomly
        mu = np.pi * (2.0 * np.random.rand(self.n_neurons) - 1.0)

        if tunemodel == 'glm':
            k0 = np.random.rand(self.n_neurons)
        else:
            k0 = np.zeros(self.n_neurons)

        k = 20.0 * np.random.rand(self.n_neurons)

        k1 = k * np.cos(mu)
        k2 = k * np.sin(mu)

        if tunemodel == 'gvm':
            g = 5.0 * np.random.rand(self.n_neurons)
        else:
            g = np.ones(self.n_neurons)

        if tunemodel == 'gvm':
            b = 10.0 * np.random.rand(self.n_neurons)
        else:
            b = np.zeros(self.n_neurons)

        # Sample features of interest randomly [-pi, pi]
        x = 2.0 * np.pi * np.random.rand(n_samples) - np.pi

        # Calculate firing rates under the desired model
        Y = np.zeros([n_samples, self.n_neurons])
        for n in range(0, self.n_neurons):
            # Compute the spike count under the tuning model for given window
            # size
            lam = 1e-3 * winsize * self._tunefun(x, k0[n], k1[n], k2[n], g[n],
                                                 b[n])

            # Sample Poisson distributed spike counts and convert back to
            # firing rate
            Y[:, n] = 1e3 / winsize * np.random.poisson(lam)

        return x, Y, mu, k0, k, g, b

    def predict(self, x):
        '''Predicts the firing rates for the population.

        Computes the firing rates for the population based on the fit or
        specified tuning models.

        Args:
            x (float): :data:`n_samples x 1`, feature of interest.

        Returns:
            float array: :data:`n_samples x n_neurons`, population activity.
        '''
        n_samples = x.shape[0]

        Y = np.zeros([n_samples, self.n_neurons])
        # For each neuron
        for n in range(0, self.n_neurons):
            # Compute the firing rate under the von Mises model
            Y[:, n] = self._tunefun(x, self.k0_[n], self.k1_[n], self.k2_[n],
                                    self.g_[n], self.b_[n])
        return Y

    def fit(self, x, Y):
        '''Fits the parameters of the model.

        Estimate the parameters of the tuning curve under the model specified
        by :meth:`tunemodel`, given features and population activity.

        Args:
            x (float): :data:`n_samples x 1`, feature of interest.
            Y (float): :data:`n_samples x n_neurons`, population activity.
        '''
        if(len(Y.shape) == 1):
            Y = deepcopy(np.expand_dims(Y, axis=1))

        learning_rate = self.learning_rate
        convergence_threshold = self.convergence_threshold
        n_repeats = self.n_repeats
        maxiter = self.maxiter

        # Fit model for each neuron
        for n in range(0, self.n_neurons):

            # Collect parameters for each repeat
            fit_params = list()

            # Repeat several times over random initializations
            # (global optimization)
            for repeat in range(0, n_repeats):
                self.set_params(self.tunemodel, n)
                fit_params.append({'k0': self.k0_[n], 'k1': self.k1_[n],
                                   'k2': self.k2_[n], 'g': self.g_[n],
                                   'b': self.b_[n], 'loss': 0.0})

                # Collect loss and delta loss for each iteration
                L, DL = list(), list()

                # Gradient descent iterations (local optimization)
                for t in range(0, maxiter):

                    converged = False

                    # Compute gradients
                    grad_k0_, grad_k1_, grad_k2_, grad_g_, grad_b_ = \
                        self._grad_theta_loss(self.tunemodel, x, Y[:, n],
                                              fit_params[repeat]['k0'],
                                              fit_params[repeat]['k1'],
                                              fit_params[repeat]['k2'],
                                              fit_params[repeat]['g'],
                                              fit_params[repeat]['b'])

                    # Update parameters
                    fit_params[repeat]['k1'] =\
                        fit_params[repeat]['k1'] - learning_rate * grad_k1_
                    fit_params[repeat]['k2'] =\
                        fit_params[repeat]['k2'] - learning_rate * grad_k2_

                    if self.tunemodel == 'glm':
                        fit_params[repeat]['k0'] =\
                            fit_params[repeat]['k0'] - learning_rate * grad_k0_
                    if self.tunemodel == 'gvm':
                        fit_params[repeat]['g'] =\
                            fit_params[repeat]['g'] - learning_rate * grad_g_
                        fit_params[repeat]['b'] =\
                            fit_params[repeat]['b'] - learning_rate * grad_b_

                    # Update loss
                    L.append(self._loss(x, Y[:, n],
                                        fit_params[repeat]['k0'],
                                        fit_params[repeat]['k1'],
                                        fit_params[repeat]['k2'],
                                        fit_params[repeat]['g'],
                                        fit_params[repeat]['b']))

                    # Update delta loss and check for convergence
                    if t > 1:
                        DL.append(L[-1] - L[-2])
                        if np.abs(DL[-1] / L[-1]) < convergence_threshold:
                            converged = True

                    # Back out gain from k1 and k2
                    fit_params[repeat]['k'] = np.sqrt(
                        fit_params[repeat]['k1'] ** 2 +
                        fit_params[repeat]['k2'] ** 2)

                    # Back out preferred feature (mu) from k1 and k2
                    fit_params[repeat]['mu'] = \
                        np.arctan2(fit_params[repeat]['k2'],
                                   fit_params[repeat]['k1'])

                    # Check for convergence
                    if converged is True:
                        break

                # Store the converged loss function
                msg = '\tConverged. Loss function: {0:.2f}'.format(L[-1])
                # logger.info(msg)
                # logger.info('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                if self.verbose is True:
                    print(msg)
                fit_params[repeat]['loss'] = L[-1]

            # Assign the global optimum
            amin = np.array([d['loss'] for d in fit_params]).argmin()
            self.mu_[n] = fit_params[amin]['mu']
            self.k0_[n] = fit_params[amin]['k0']
            self.k1_[n] = fit_params[amin]['k1']
            self.k2_[n] = fit_params[amin]['k2']
            self.k_[n] = fit_params[amin]['k']
            self.g_[n] = fit_params[amin]['g']
            self.b_[n] = fit_params[amin]['b']

    def decode(self, Y):
        '''Estimates the features that generated a given population activity.

        Args:
            Y (float): :data:`n_samples x n_neurons`, population activity.

        Returns:
            float array: :data:`n_samples x 1`, feature of interest.
        '''

        n_samples = Y.shape[0]

        maxiter = self.maxiter
        learning_rate = self.learning_rate
        convergence_threshold = self.convergence_threshold

        # Initialize feature
        x = np.pi * (2.0 * np.random.rand(n_samples) - 1.0)

        # For each sample
        for s in range(0, n_samples):

            # Collect loss and delta loss for each iteration
            L, DL = list(), list()

            # Gradient descent iterations (local optimization)
            for t in range(0, maxiter):

                # Compute gradients
                grad_x_ = self._grad_x_loss(x[s], Y[s, :],
                                            self.k0_, self.k1_, self.k2_,
                                            self.g_, self.b_)

                # Update parameters
                x[s] = x[s] - learning_rate * grad_x_

                # Update loss
                L.append(self._loss(x[s], Y[s, :],
                                    self.k0_, self.k1_, self.k2_, self.g_,
                                    self.b_))

                # Update delta loss and check for convergence
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < convergence_threshold:
                        msg = '\t Converged. Loss function: {0:.2f}'.format(
                            L[-1])
                        # logger.info(msg)
                        if self.verbose is True:
                            # if True:
                            print(msg)
                        break

        # Make sure x is between [-pi, pi]
        x = np.arctan2(np.sin(x), np.cos(x))

        return x

    def display(self, x, Y, neuron, colors=['k', 'c'], alpha=0.5, ylim=[0, 25],
                xlabel='direction [radians]', ylabel='firing rate [spk/s]',
                style='../mpl_styles/spykes.mplstyle',
                xjitter=False, yjitter=False):
        '''
        Visualize data and estimated tuning curves

        Args:
            x (float): :data:`n_samples x 1`, feature of interest.
            Y (float): :data:`n_samples x 1`, firing rates.
            neuron (int): Which neuron's fit to plot from the population?
            colors (list of str): Plot strings that specify color for raw data
                and fit.
            alpha (float): Transparency for raw data.
            ylim (list of float): Y axis limits.
            xlabel (str): X label (typically name of the feature).
            ylabel (str): Y label (typically firing rate).
            style (str): Name of the mpl style file to use with path.
            xjitter (bool): Whether to add jitter to x variable while plotting.
            ylitter (bool): Whether to add jitter to y variable while plotting.
        '''

        utils.set_matplotlib_defaults(plt)

        if xjitter is True:
            x_jitter = np.pi / 32 * np.random.standard_normal(x.shape)
        else:
            x_jitter = np.zeros(x.shape)

        if yjitter is True:
            y_range = np.max(Y) - np.min(Y)
            Y_jitter = y_range / 20.0 * np.random.standard_normal(Y.shape)
        else:
            Y_jitter = np.zeros(Y.shape)

        plt.plot(x + x_jitter, Y + Y_jitter, '.', color=colors[0], alpha=alpha)

        x_range = np.arange(-np.pi, np.pi, np.pi / 32)
        Yhat_range = self._tunefun(x_range,
                                   self.k0_[neuron],
                                   self.k1_[neuron],
                                   self.k2_[neuron],
                                   self.g_[neuron],
                                   self.b_[neuron])

        plt.plot(x_range, Yhat_range, '-', linewidth=4, color=colors[1])
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tick_params(axis='y', right='off')
        plt.tick_params(axis='x', top='off')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def score(self, Y, Yhat, Ynull=None, method='circ_corr'):
        '''Scores the model.

        Args:
            Y (array): The true firing rates, an array with shape
                :data:`(n_samples, n_neurons)`.
            Yhat (array): The estimated firing rates, an array with shape
                :data:`(n_samples, [n_neurons])`.
            Ynull (array or None): The labels of the null model. Must be None
                if :data:`method` is not :data:`pseudo_R2`. The array has
                shape :data:`(n_samples, [n_classes])`.
            method (str): One of :data:`pseudo_R2`, :data:`circ_corr`, or
                :data:`cosine_dist`.

        Returns:
            scalar float: The computed score.
        '''

        if method == 'pseudo_R2':
            if(len(Y.shape) > 1):
                # There are many neurons, so calculate and return the score for
                # each neuron
                score = list()
                for neuron in range(Y.shape[1]):
                    L1 = utils.log_likelihood(Y[:, neuron], Yhat[:, neuron])
                    LS = utils.log_likelihood(Y[:, neuron], Y[:, neuron])
                    L0 = utils.log_likelihood(Y[:, neuron], Ynull[neuron])
                    score.append(1 - (LS - L1) / (LS - L0))
            else:
                L1 = utils.log_likelihood(Y, Yhat)
                LS = utils.log_likelihood(Y, Y)
                L0 = utils.log_likelihood(Y, Ynull)
                score = 1 - (LS - L1) / (LS - L0)
        elif method == 'circ_corr':
            score = utils.circ_corr(np.squeeze(Y), np.squeeze(Yhat))
        elif method == 'cosine_dist':
            score = np.mean(np.cos(np.squeeze(Y) - np.squeeze(Yhat)))
        else:
            raise ValueError('Invalid method: "{}". Must "pseudo_R2", '
                             '"circ_corr" or "cosine_dist".'.format(method))

        return score
