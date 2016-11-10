
===========================================
Tutorial
===========================================

Fitting Tuning Curves with Gradient Descent
-------------------------------------------

The firing rates :math:`y_j` of neuron :math:`j` can be modeled as a
Poisson random variable.

.. math::


   y_j = \text{Poisson}(\lambda_j)

We will drop the subscript :math:`j` for convenience of notation and
figure out how to fit the tuning curves of a given neuron :math:`j`.

The mean :math:`\lambda` is given by the von Mises tuning model as
follows.

.. math::


   \lambda = b + g\exp\Big(\kappa_0 + \kappa \cos(x - \mu)\Big)

However, this formulation is non-convex in :math:`\mu`. Therefore, we
re-parameterize it to be more tractable (still non-convex in :math:`b`
and :math:`g`) as follows.

.. math::


   \lambda = b + g\exp\Big(\kappa_0 + \kappa_1 \cos(x) + \kappa_2 \sin(x) \Big),

where :math:`\kappa_1 = \kappa \cos(\mu)` and
:math:`\kappa_2 = \kappa \sin(\mu)`.

Once we estimate :math:`\kappa_1` and :math:`\kappa_2`, we can back out
:math:`\kappa` and :math:`\mu` as
:math:`\kappa = \sqrt{\kappa_1^2 + \kappa_2^2}`, and
:math:`\mu = \tan^{-1}\Big(\frac{\kappa_2}{\kappa_1}\Big)`.

We estimate two special cases of this generalized von Mises model.

Special Case 1: Poisson Generalized Linear Model (GLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we set :math:`b = 0` and :math:`g =1`, we get:

.. math::


   \lambda = \exp\Big(\kappa_0 + \kappa_1 \cos(x) + \kappa_2 \sin(x) \Big),

This is identical to a Poisson GLM.

The advantage of this formulation is that it is convex and the
disadvantage is that all parameters are not straightforward to
interpret, with :math:`\kappa_0` playing the role of both a baseline and
a gain term.

Special Case 2: Generalized von Mises Model (GVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we set :math:`\kappa_0 = 0`, we get:

.. math::


   \lambda = b + g\exp\Big(\kappa_1 \cos(x) + \kappa_2 \sin(x) \Big),

This is identical to a Eq. (4) in `Amirikan & Georgopulos
(2000) <http://brain.umn.edu/pdfs/BA118.pdf>`__.

The advantage of this formulation is is that although it is non-convex,
we can easily interpret the parameters: - :math:`b`, as the baseline
firing rate - :math:`g`, as the gain - :math:`\kappa`, as the width or
shape

Minimizing Negative Log Likelihood with Gradient Descent
--------------------------------------------------------

Given a set of observations :math:`(x_i, y_i)`, to identify the
parameters
:math:`\Theta = \left\{\kappa_0, \kappa_1, \kappa_2, g, b\right\}` we
use gradient descent on the loss function :math:`J`, specified by the
negative Poisson log-likelihood,

.. math::


   J = -\log\mathcal{L} = \sum_{i} \lambda_i - y_i \log \lambda_i

Taking the gradients, we get:

.. math::


   \frac{\partial J}{\partial \kappa_0} = \sum_{i} g \exp\Big(\kappa_0 + \kappa_1 \cos(x_i) + \kappa_2 \sin(x_i) \Big) \bigg(1 - \frac{y_i}{\lambda_i}\bigg)

.. math::


   \frac{\partial J}{\partial \kappa_1} = \sum_{i} g \exp\Big(\kappa_0 + \kappa_1 \cos(x_i) + \kappa_2 \sin(x_i) \Big) \cos(x_i) \bigg(1 - \frac{y_i}{\lambda_i}\bigg)

.. math::


   \frac{\partial J}{\partial \kappa_2} = \sum_{i} g \exp\Big(\kappa_0 + \kappa_1 \cos(x_i) + \kappa_2 \sin(x_i) \Big) \sin(x_i) \bigg(1 - \frac{y_i}{\lambda_i}\bigg)

.. math::


   \frac{\partial J}{\partial g} = \sum_{i} g \exp\Big(\kappa_0 + \kappa_1 \cos(x_i) + \kappa_2 \sin(x_i) \Big) \bigg(1 - \frac{y_i}{\lambda_i}\bigg)

.. math::


   \frac{\partial J}{\partial b} = \sum_{i} \bigg(1 - \frac{y_i}{\lambda_i}\bigg)

Decoding Feature from Population Activity
--------------------------------------------------------

Under the same Poisson firing rate model for each neuron, whose mean is
specified by the von Mises tuning curve, as above, we can decode the
stimulus :math:`\hat{x}` that is most likely to have produced the
observed population activity
:math:`Y = \left\{y_j, j = 1, 2, \dots \text{n_neurons}\right\}`.

We will assume that the neurons are conditionally independent given the
tuning parameters :math:`\Theta`. Thus the likelihood of observing the
population activity :math:`Y` is given by

.. math::


   P(Y | \Theta) = \prod_j P(y_j | \Theta)

As before, the loss function for the decoder is given by the negative
Poisson log-likelihood:

.. math::


   J = -\log\mathcal{L} = \sum_j \lambda_j - y_j \log \lambda_j

where

.. math::


   \lambda_j = b_j + g_j \exp\Big(\kappa_{0,j} + \kappa_{1,j} \cos(x) + \kappa_{1,j} \sin(x) \Big)

To minimize this loss function with gradient descent, we need to take
the gradient of :math:`J` with respect to :math:`x`

.. math::


   \frac{\partial J}{\partial x} = \sum_{j} g_j \exp\Big(\kappa_{0,j} + \kappa_{1,j} \cos(x) + \kappa_{2,j} \sin(x) \Big) \Big(\kappa_{2,j} \cos(x) - \kappa_{1,j} \sin(x)\Big) \bigg(1 - \frac{y_j}{\lambda_j}\bigg)

