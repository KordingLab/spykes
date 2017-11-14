from math import pi as PI

import tensorflow as tf
from tensorflow import keras as ks


class PoissonLayer(ks.layers.Layer):
    '''Defines a TensorFlow implementation of the NeuroPop layers.

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
        model_type (str): Can be either :data:`gvm`, the Generalized von Mises
            model, or :data:`glm`, the Poisson generalized linear model.
        num_neurons (int): Number of neurons in the population (being inferred
            from the input features).
        num_features (int): Number of input features. Convenience parameter for
            for setting the input shape.
        mu_initializer (Keras initializer): The initializer for the :data:`mu`.
        k_initializer (Keras initializer): The initializer for the :data:`k`.
        g_initializer (Keras initializer): The initializer for the :data:`g`.
            If :data:`model_type` is :data:`glm`, this is ignored.
        b_initializer (Keras initializer): The initializer for the :data:`b`.
            If :data:`model_type` is :data:`glm`, this is ignored.
        k0_initializer (Keras initializer): The initializer for the :data:`k0`.
            If :data:`model_type` is :data:`gvm`, this is ignored.
    '''

    def __init__(self,
                 model_type,
                 num_neurons,
                 num_features=None,
                 mu_initializer=ks.initializers.RandomUniform(-PI, PI),
                 k_initializer=ks.initializers.RandomNormal(stddev=.2),
                 g_initializer=ks.initializers.RandomNormal(stddev=.05),
                 b_initializer=ks.initializers.RandomNormal(stddev=.1),
                 k0_initializer=ks.initializers.RandomNormal(stddev=.01),
                 **kwargs):
        if num_features is not None:
            kwargs['input_shape'] = (num_features,)
        super(PoissonLayer, self).__init__(**kwargs)
        self.model_type = model_type.lower()
        if self.model_type not in ('gvm', 'glm'):
            raise ValueError('Invalid model type: "{}" Must be either "gvm" '
                             '(generalised Von Mises model) or "glm" '
                             '(generalized linear model)'.format(model_type))

        self.num_neurons = num_neurons
        self.mu_initializer = ks.initializers.get(mu_initializer)
        self.g_initializer = ks.initializers.get(g_initializer)
        self.b_initializer = ks.initializers.get(b_initializer)
        self.k_initializer = ks.initializers.get(k_initializer)
        self.k0_initializer = ks.initializers.get(k0_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]

        self.mu = self.add_weight(
            shape=(input_dim, self.num_neurons),
            initializer=self.mu_initializer,
            name='mu',
        )
        self.k1 = self.add_weight(
            shape=(input_dim, self.num_neurons),
            initializer=self.k_initializer,
            name='k1',
        )
        self.k2 = self.add_weight(
            shape=(input_dim, self.num_neurons),
            initializer=self.k_initializer,
            name='k2',
        )

        # Adds generalized Von Mises parameters.
        if self.model_type == 'gvm':
            self.g = self.add_weight(
                shape=(1, input_dim),
                initializer=self.g_initializer,
                name='g',
            )
            self.b = self.add_weight(
                shape=(1, input_dim),
                initializer=self.b_initializer,
                name='b',
            )

        # Adds generalized linear model parameters.
        if self.model_type == 'glm':
            self.k0 = self.add_weight(
                shape=(1, input_dim),
                initializer=self.k_initializer,
                name='k0',
            )

    def call(self, inputs):
        k1 = tf.matmul(tf.cos(inputs), self.k1 * tf.cos(self.mu))
        k2 = tf.matmul(tf.sin(inputs), self.k2 * tf.sin(self.mu))

        # Defines the two model formulations: "glm" vs "gvm".
        if self.model_type == 'glm':
            return tf.exp(k1 + k2 + self.k0)
        else:
            return tf.nn.softplus(self.b) + self.g * tf.exp(k1 + k2)

    def get_config(self):
        config = {
            'model_type': self.model_type,
            'mu_initializer': ks.initializers.serialize(self.mu_initializer),
            'g_initializer': ks.initializers.serialize(self.g_initializer),
            'b_initializer': ks.initializers.serialize(self.b_initializer),
            'k_initializer': ks.initializers.serialize(self.k_initializer),
            'k0_initializer': ks.initializers.serialize(self.k0_initializer),
        }
        base_config = super(PoissonLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.num_neurons
        return tuple(output_shape)
