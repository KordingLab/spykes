from __future__ import print_function

import six
import collections

import tensorflow as tf
from tensorflow import keras as ks


def sparse_filtering_loss(_, y_pred):
    '''Defines the sparse filtering loss function.

    Args:
        y_true (tensor): The ground truth tensor (not used, since this is an
            unsupervised learning algorithm).
        y_pred (tensor): Tensor representing the feature vector at a
            particular layer.

    Returns:
        scalar tensor: The sparse filtering loss.
    '''
    y = tf.reshape(y_pred, tf.stack([-1, tf.reduce_prod(y_pred.shape[1:])]))
    l2_normed = tf.nn.l2_normalize(y, dim=1)
    l1_norm = tf.norm(l2_normed, ord=1, axis=1)
    return tf.reduce_sum(l1_norm)


class SparseFiltering(object):
    '''Defines a class for performing sparse filtering on a dataset.

    The MATLAB code on which this is based is available `here
    <https://github.com/jngiam/sparseFiltering>`_, from the paper
    `Sparse Filtering by Ngiam et. al.
    <https://papers.nips.cc/paper/4334-sparse-filtering.pdf>`_.

    Args:
        model (Keras model): The trainable model, which takes as inputs the
            data you are training on and outputs a feature vector, which is
            minimized according to the loss function described above.
        layers (str or list of str): An optional name or list of names of
            layers in the provided model whose outputs to apply the sparse
            filtering loss to. If none are provided, the sparse filtering loss
            is applied to each layer in the model.
    '''

    def __init__(self, model, layers=None):
        assert isinstance(model, ks.models.Model)
        assert len(model.inputs) == 1 and len(model.outputs) == 1
        self.model = model

        # Parses the "layers" argument.
        if layers is None:
            self.layer_names = [layer.name for layer in model.layers]
        elif isinstance(layers, six.string_types):
            self.layer_names = [layers]
        elif isinstance(layers, collections.Iterable):
            self.layer_names = list(layers)
        else:
            raise ValueError('`layers` must be a string (a single layer) or '
                             'a list of strings. Got: "{}"'.format(layers))

        self._submodels = None

    @property
    def submodels(self):
        if self._submodels is None:
            raise RuntimeError('This model must be compiled before you can '
                               'access the `submodels` parameter.')
        return self._submodels

    @property
    def num_layers(self):
        return len(self.layer_names)

    def _clean_maybe_iterable_param(self, it, param):
        '''Converts a potential iterable or single value to a list of values.

        After being cleaned, the iterable is guarenteed to be a list of the
        same length as the number of layer names.

        Args:
            it (single value or iterable): The iterable to clean.
            param (str): The name of the parameter being set.

        Returns:
            list: a list of values of the same length as the layer names.
        '''
        if isinstance(it, six.string_types):
            return [it] * self.num_layers
        elif isinstance(it, collections.Iterable):
            it = list(it)
            if len(it) != self.num_layers:
                raise ValueError('Provided {} values for `{}`, '
                                 'but one parameter is needed for each '
                                 'requested layer ({} layers).'
                                 .format(len(it), param, self.num_layers))
            return it
        else:
            return [it] * self.num_layers

    def compile(self, optimizer, freeze=False, **kwargs):
        '''Compiles the model to create all the submodels.

        Args:
            optimizer (str or list of str): The optimizer to use. If a list is
                provided, it must specify one optimizer for each layer
                passed to the constructor.
            freeze (bool): If set, for each submodel, all the previous layers
                are frozen, so that only the last layer is "learned".
            kwargs (dict): Extra arguments to be passed to the :data:`compile`
                function of each submodel.
        '''
        if self._submodels is not None:
            raise RuntimeError('This model has already been compiled!')
        optimizer = self._clean_maybe_iterable_param(optimizer, 'optimizer')

        # Creates each submodel.
        self._submodels = []
        input_layer = self.model.input
        for layer_name, o in zip(self.layer_names, optimizer):
            output_layer = self.model.get_layer(layer_name).output
            submodel = ks.models.Model(
                inputs=input_layer,
                outputs=output_layer,
            )
            submodel.compile(loss=sparse_filtering_loss, optimizer=o, **kwargs)
            self._submodels.append(submodel)

    def fit(self, x, epochs=1, **kwargs):
        '''Fits the model to the provided data.

        Args:
            x (Numpy array): An array where the first dimension is the batch
                size, and the remaining dimensions match the input dimensions
                of the provided model.
            epochs (int or list of ints): The number of epochs to train.
                If a list is provided, there must be one value for each named
                layer, specifying the number of epochs to train that layer for.

        Returns:
            list: A list of histories, the training history for each submodel.
        '''
        histories = []
        nb_epochs = self._clean_maybe_iterable_param(epochs, 'epochs')
        for submodel, n in zip(self._submodels, nb_epochs):
            histories.append(submodel.fit(x=x, y=x, epochs=n, **kwargs))
        return histories
