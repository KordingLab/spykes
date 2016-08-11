import os
import numpy as np
from scipy import stats
from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../mpl_styles/spykes.mplstyle')
    )


from . import utils
from numba.decorators import autojit
slow_exp = autojit(utils.slow_exp_python)
grad_slow_exp = autojit(utils.grad_slow_exp_python)

class NeuroPop(object):
    """
    This class implements several conveniences for
    plotting, fitting and decoding from population tuning curves

    We allow the fitting of two classes of parametric tuning curves.

    Parameters
    ----------
    tunemodel: str, can be either 'gvm' or 'glm'
        tunemodel = 'gvm'
        Generalized von Mises model
        Amirikan & Georgopulos (2000):
        http://brain.umn.edu/pdfs/BA118.pdf
        f(x) = b_ + g_ * exp(k_ * cos(x - mu_))
        f(x) = b_ + g_ * exp(k1_ * cos(x) + k2_ * sin(x))

        tunemodel = 'glm'
        Poisson generalized linear model
        f(x) = exp(k0_ + k_ * cos(x - mu_))
        f(x) = exp(k0_ + k1_ * cos(x) + k2_ * sin(x))

    n_neurons: float, number of neurons in the population
    random_state: int, seed for numpy.random
    eta: float, linearizes the exp above eta, default: 4.0
    learning_rate: float, default: 2e-1
    convergence_threshold: float, default, 1e-5
    maxiter: float, default: 1000
    n_repeats: float, default: 5
    verbose: bool, whether to print convergence / loss, default: False

    Internal variables
    ------------------
    mu_: float,  n_neurons x 1, preferred feature [-pi, pi]
    k0_: float,  n_neurons x 1, baseline
    k_: float,  n_neurons x 1, shape (width)
    k1_: float,  n_neurons x 1, convenience parameter
    k2_: float,  n_neurons x 1, convenience parameter
    g_: float,  n_neurons x 1, gain
    b_: float,  n_neurons x 1, baseline

    Callable methods
    ----------------
    set_params
    simulate
    fit
    predict
    decode
    display
    score

    Class methods
    -------------
    _tunefun
    _loss
    _grad_theta_loss
    _grad_x_loss
    """

    def __init__(self, tunemodel='glm', n_neurons=100,
                 random_state=1,
                 eta=0.4,
                 learning_rate=2e-1, convergence_threshold=1e-5,
                 maxiter=1000, n_repeats=1,
                 verbose=False):
        """
        Initialize the object
        """
        self.tunemodel = tunemodel
        self.n_neurons = n_neurons

        # Assign random tuning parameters
        #--------------------------------
        self.mu_ = np.zeros(n_neurons)
        self.k0_ = np.zeros(n_neurons)
        self.k_ = np.zeros(n_neurons)
        self.k1_ = np.zeros(n_neurons)
        self.k2_ = np.zeros(n_neurons)
        self.g_ = np.zeros(n_neurons)
        self.b_ = np.zeros(n_neurons)
        np.random.seed(random_state)
        self.set_params(tunemodel, range(self.n_neurons))

        # Assign optimization parameters
        #-------------------------------
        self.eta = eta
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.maxiter = maxiter
        self.n_repeats = n_repeats

        self.verbose = verbose

    #-----------------------------------------------------------------------
    def set_params(self, tunemodel=None, neurons=None, mu=None, k0=None, k=None, g=None, b=None):
        """
        A function that sets tuning curve parameters as specified

        Parameters
        ----------
        tunemodel: str, either 'gvm' or 'glm'
        neurons: list,
            a list of integers which specifies the subset of neurons to set
            default: all neurons

        mu: float,  len(neurons) x 1, feature of interest
        k0: float,  len(neurons) x 1, baseline
        k: float,  len(neurons) x 1, gain
        g: float,  len(neurons) x 1, gain
        b: float,  len(neurons) x 1, baseline

        if any of the above are None, it randomly initializes parameters for
        all neurons
        """
        if tunemodel is None:
            tunemodel = self.tunemodel

        if neurons is None:
            neurons = range(self.n_neurons)

        if isinstance(neurons, list):
            n_neurons = len(neurons)
        else:
            n_neurons = 1

        # Assign parameters; if None, assign random
        if mu is None:
            self.mu_[neurons] = np.pi * (2.0 * np.random.rand(n_neurons) - 1.0)
        else:
            self.mu_[neurons] = mu

        if k0 is None:
            if tunemodel == 'glm':
                self.k0_[neurons] = np.random.rand(n_neurons)
            else:
                self.k0_[neurons] = np.zeros(n_neurons)
        else:
            self.k0_[neurons] = k0

        if k is None:
            self.k_[neurons] = 20.0 * np.random.rand(n_neurons)
        else:
            self.k_[neurons] = k

        self.k1_[neurons] = self.k_[neurons] * np.cos(self.mu_[neurons])
        self.k2_[neurons] = self.k_[neurons] * np.sin(self.mu_[neurons])

        if g is None:
            if tunemodel == 'gvm':
                self.g_[neurons] = 5.0 * np.random.rand(n_neurons)
            else:
                self.g_[neurons] = np.ones(n_neurons)
        else:
            self.g_[neurons] = g

        if b is None:
            if tunemodel == 'gvm':
                self.b_[neurons] = 10.0 * np.random.rand(n_neurons)
            else:
                self.b_[neurons] = np.zeros(n_neurons)
        else:
            self.b_[neurons] = b

    #-----------------------------------------------------------------------
    def _tunefun(self, x, k0, k1, k2, g, b):
        """
        The tuning function as specified in self.tunemodel

        Parameters
        ----------
        x: float, n_samples x 1, feature of interest
        k0: float,  n_neurons x 1, baseline
        k1: float,  n_neurons x 1, convenience parameter
        k2: float,  n_neurons x 1, convenience parameter
        g: float,  n_neurons x 1, gain
        b: float,  n_neurons x 1, baseline

        Outputs
        -------
        Y: float, n_samples x 1, firing rates
        """
        y = b + g * slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta)
        return y

    #-----------------------------------------------------------------------
    def _loss(self, x, y, k0, k1, k2, g, b):
        """
        The loss function: negative Poisson log likelihood function
        under the von mises tuning model

        Parameters
        ----------
        x: float, n_samples x 1 (encoding) | scalar (decoding), feature of interest
        y: float, n_samples x 1 (encoding) | n_neurons x 1 (decoding), firing rates
        mu: float,  n_neurons x 1, preferred feature [-pi, pi]
        k0: float,  n_neurons x 1, baseline
        k1: float,  n_neurons x 1, convenience parameter
        k2: float,  n_neurons x 1, convenience parameter
        g: float,  n_neurons x 1, gain
        b: float,  n_neurons x 1, baseline

        Outputs
        -------
        loss: float, scalar
        """
        lmb = self._tunefun(x, k0, k1, k2, g, b)
        J = np.sum(lmb) - np.sum(y * lmb)
        return J

    #-----------------------------------------------------------------------
    def _grad_theta_loss(self, tunemodel, x, y, k0, k1, k2, g, b):
        """
        The gradient of the loss function:
        wrt parameters of the tuning model (theta)

        Parameters
        ----------
        x: float,  n_samples x 1, feature of interest
        y: float,  n_samples x 1, firing rates
        k0: float,  n_neurons x 1, baseline
        k1: float,  n_neurons x 1, convenience parameter
        k2: float,  n_neurons x 1, convenience parameter
        g: float,  scalar, gain
        b: float,  scalar, baseline

        Outputs
        -------
        grad_k0: float, scalar
        grad_k1: float, scalar
        grad_k2: float, scalar
        grad_g: float, scalar
        grad_b: float, scalar
        """
        lmb = self._tunefun(x, k0, k1, k2, g, b)

        n_samples = np.float(x.shape[0])
        grad_k1 = 1./n_samples * np.sum(g * grad_slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta) * np.cos(x) * (1 - y/lmb))
        grad_k2 = 1./n_samples * np.sum(g * grad_slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta) * np.sin(x) * (1 - y/lmb))
        if tunemodel == 'glm':
            grad_k0 = 1./n_samples * np.sum(grad_slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta) * (1 - y/lmb))
            grad_g = 0.0
            grad_b = 0.0
        elif tunemodel == 'gvm':
            grad_k0 = 0.0
            grad_g = 1./n_samples * np.sum(slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta) * (1 - y/lmb))
            grad_b = 1./n_samples * np.sum((1-y/lmb))


        return grad_k0, grad_k1, grad_k2, grad_g, grad_b

    #-----------------------------------------------------------------------
    def _grad_x_loss(self, x, y, k0, k1, k2, g, b):
        """
        The gradient of the loss function:
        wrt encoded feature x

        Parameters
        ----------
        x: float, scalar, feature of interest
        y: float, n_neurons x 1, firing rates
        k0: float,  n_neurons x 1, baseline
        k1: float,  n_neurons x 1, convenience parameter
        k2: float,  n_neurons x 1, convenience parameter
        g: float,  n_neurons x 1, gain
        b: float,  n_neurons x 1, baseline

        Outputs
        -------
        grad_x: float, scalar
        """
        n_neurons = np.float(self.n_neurons)

        lmb = self._tunefun(x, k0, k1, k2, g, b)
        grad_x = 1./n_neurons * np.sum(g * grad_slow_exp(k0 + k1 * np.cos(x) + k2 * np.sin(x), self.eta) * \
                (k2 * np.cos(x) - k1 * np.sin(x)) * (1 - y/lmb))
        return grad_x

    #-----------------------------------------------------------------------
    def simulate(self, tunemodel, n_samples=500, winsize=200):
        """
        Simulates firing rates from a neural population by randomly sampling
        circular variables (feature of interest)
        as well as parameters (mu, k0, k, g, b)

        Parameters
        ----------
        n_samples, int, number of samples required
        winsize, float, time interval in which to simulate spike counts, milliseconds
        Outputs
        -------
        x: float, n_samples x 1, feature of interest
        Y: float, n_samples x n_neurons, population activity
        mu: float,  n_neurons x 1, preferred feature [-pi, pi]
        k0: float,  n_neurons x 1, baseline
        k: float,  n_neurons x 1, shape (width)
        g: float,  n_neurons x 1, gain
        b: float,  n_neurons x 1, baseline
        """

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
            # Compute the spike count under the tuning model for given window size
            lam = 1e-3 * winsize * self._tunefun(x, k0[n], k1[n], k2[n], g[n], b[n])

            # Sample Poisson distributed spike counts and convert back to firing rate
            Y[:, n] = 1e3/ winsize * np.random.poisson(lam)

        return x, Y, mu, k0, k, g, b

    def predict(self, x):
        """
        Compute the firing rates for the population
        based on the fit or specified tuning models

        Parameters
        ----------
        x: float, n_samples x 1, feature of interest

        Outputs
        -------
        Y: float, n_samples x n_neurons, population activity
        """
        n_samples = x.shape[0]

        Y = np.zeros([n_samples, self.n_neurons])
        # For each neuron
        for n in range(0, self.n_neurons):
            # Compute the firing rate under the von Mises model
            Y[:, n] = self._tunefun(x, self.k0_[n], self.k1_[n], self.k2_[n], self.g_[n], self.b_[n])
        return Y

    #-----------------------------------------------------------------------
    def fit(self, x, Y):
        """
        Estimate the parameters of the tuning curve under the
        model specified by self.tunemodel,
        given features and population activity

        Parameters
        ----------
        x: float, n_samples x 1, feature of interest
        Y: float, n_samples x n_neurons, population activity
        """
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

            # Repeat several times over random initializations (global optimization)
            for repeat in range(0, n_repeats):
                self.set_params(self.tunemodel, n)
                fit_params.append({'k0': self.k0_[n], 'k1': self.k1_[n],
                'k2': self.k2_[n], 'g': self.g_[n], 'b': self.b_[n], 'loss': 0.0})

                # Collect loss and delta loss for each iteration
                L, DL = list(), list()

                # Gradient descent iterations (local optimization)
                for t in range(0, maxiter):

                    converged = False

                    # Compute gradients
                    grad_k0_, grad_k1_, grad_k2_, grad_g_, grad_b_ = \
                    self._grad_theta_loss(self.tunemodel, x, Y[:,n],
                                    fit_params[repeat]['k0'],
                                    fit_params[repeat]['k1'],
                                    fit_params[repeat]['k2'],
                                    fit_params[repeat]['g'],
                                    fit_params[repeat]['b'])

                    # Update parameters
                    fit_params[repeat]['k1'] = fit_params[repeat]['k1'] - learning_rate*grad_k1_
                    fit_params[repeat]['k2'] = fit_params[repeat]['k2'] - learning_rate*grad_k2_

                    if self.tunemodel == 'glm':
                        fit_params[repeat]['k0'] = fit_params[repeat]['k0'] - learning_rate*grad_k0_
                    if self.tunemodel == 'gvm':
                        fit_params[repeat]['g'] = fit_params[repeat]['g'] - learning_rate*grad_g_
                        fit_params[repeat]['b'] = fit_params[repeat]['b'] - learning_rate*grad_b_

                    # Update loss
                    L.append(self._loss(x, Y[:,n],\
                                        fit_params[repeat]['k0'],\
                                        fit_params[repeat]['k1'],
                                        fit_params[repeat]['k2'],\
                                        fit_params[repeat]['g'],\
                                        fit_params[repeat]['b']))

                    # Update delta loss and check for convergence
                    if t > 1:
                        DL.append(L[-1] - L[-2])
                        if np.abs(DL[-1] / L[-1]) < convergence_threshold:
                            converged = True

                    # Back out gain from k1 and k2
                    fit_params[repeat]['k'] = np.sqrt(
                            fit_params[repeat]['k1'] ** 2 + \
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
                #logger.info(msg)
                #logger.info('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
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

    #-----------------------------------------------------------------------
    def decode(self, Y):
        """
        Given population activity estimate the feature that generated it

        Parameters
        ----------
        Y: float, n_samples x n_neurons, population activity

        Outputs
        -------
        x: float, n_samples x 1, feature of interest
        """

        n_samples = Y.shape[0]

        maxiter = self.maxiter
        learning_rate = self.learning_rate
        convergence_threshold = self.convergence_threshold

        # Initialize feature
        x = np.pi*(2.0*np.random.rand(n_samples) - 1.0)

        # For each sample
        for s in range(0, n_samples):

            # Collect loss and delta loss for each iteration
            L, DL = list(), list()

            # Gradient descent iterations (local optimization)
            for t in range(0, maxiter):

                # Compute gradients
                grad_x_ = self._grad_x_loss(x[s], Y[s,:],
                         self.k0_, self.k1_, self.k2_, self.g_, self.b_)

                # Update parameters
                x[s] = x[s] - learning_rate*grad_x_

                # Update loss
                L.append(self._loss(x[s], Y[s,:],
                         self.k0_, self.k1_, self.k2_, self.g_, self.b_))

                # Update delta loss and check for convergence
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < convergence_threshold:
                        msg = '\t Converged. Loss function: {0:.2f}'.format(L[-1])
                        #logger.info(msg)
                        #logger.info('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                        if self.verbose == True:
                        #if True:
                            print(msg)
                        break

        # Make sure x is between [-pi, pi]
        x = np.arctan2(np.sin(x), np.cos(x))

        return x
    #-----------------------------------------------------------------------
    def display(self, x, Y, neuron, colors=['k', 'c'], alpha=0.5, ylim=[0, 25],
                xlabel='direction [radians]', ylabel='firing rate [spk/s]',
                style='../mpl_styles/spykes.mplstyle',
                xjitter=False, yjitter=False):
        """
        Visualize data and estimated tuning curves

        Parameters
        ----------
        x: float, n_samples x 1, feature of interest
        Y: float, n_samples x 1, firing rates
        neuron: int, which neuron's fit to plot from the population?
        colors: list of str, plot strings that specify color for raw data and fit
        alpha: float, transparency for raw data
        ylim: list of float, y axis limits
        xlabel: str, x label (typically name of the feature)
        ylabel: str, y label (typically firing rate)
        style: str, name of the mpl style file to use with path
        xjitter: bool, whether to add jitter to x variable while plotting
        ylitter: bool, whether to add jitter to y variable while plotting
        """

        plt.style.use(style)

        if xjitter is True:
            x_jitter = np.pi/32*np.random.randn(x.shape)
        else:
            x_jitter = np.zeros(x.shape)

        if yjitter is True:
            y_range = np.max(Y) - np.min(Y)
            Y_jitter = y_range/20.0*np.random.randn(Y.shape)
        else:
            Y_jitter = np.zeros(Y.shape)

        plt.plot(x + x_jitter, Y + Y_jitter, '.', color=colors[0], alpha=alpha)

        x_range = np.arange(-np.pi, np.pi, np.pi/32)
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

    #-----------------------------------------------------------------------
    def score(self, Y, Yhat, Ynull=None, method='circ_corr'):
        """Score the model.
        Parameters
        ----------
        Y : array, shape (n_samples, [n_neurons])
            The true firing rates.
        Yhat : array, shape (n_samples, [n_neurons])
            The estimated firing rates.
        Ynull : None | array, shape (n_samples, [n_classes])
            The labels for the null model. Must be None if method is not 'pseudo_R2'
        method : str
            One of 'pseudo_R2' or 'circ_corr' or 'cosine_dist'
        """

        if method == 'pseudo_R2':
            if(len(Y.shape) > 1):
                # There are many neurons, so calculate and return the score for each neuron
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
            score = np.mean(np.cos(np.squeeze(Y)-np.squeeze(Yhat)))

        return score
