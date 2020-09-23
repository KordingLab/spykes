from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as p
from nose.tools import assert_equal

from spykes.ml.strf import STRF
p.switch_backend('Agg')


def test_strf():

    n_spatial_basis = 36
    n_temporal_basis = 7
    patch_size = 50

    # Instantiate strf object
    strf_ = STRF(patch_size=patch_size, sigma=5,
                 n_spatial_basis=n_spatial_basis,
                 n_temporal_basis=n_temporal_basis)

    # Design a spatial basis
    spatial_basis = strf_.make_cosine_basis()
    assert_equal(len(spatial_basis), 2)
    for basis in spatial_basis:
        assert_equal(basis.shape[0], patch_size)
        assert_equal(basis.shape[1], patch_size)

    spatial_basis = strf_.make_gaussian_basis()
    assert_equal(len(spatial_basis), n_spatial_basis)

    # Visualize spatial basis
    strf_.visualize_gaussian_basis(spatial_basis)

    # Design temporal basis
    time_points = np.linspace(-100., 100., 10)
    centers = [-75., -50., -25., 0, 25., 50., 75.]
    width = 10.
    temporal_basis = strf_.make_raised_cosine_temporal_basis(
        time_points=time_points,
        centers=centers,
        widths=width * np.ones(7))
    assert_equal(temporal_basis.shape[0], len(time_points))
    assert_equal(temporal_basis.shape[1], n_temporal_basis)

    # Project to spatial basis
    I = np.zeros(shape=(patch_size, patch_size))
    row = 5
    col = 10
    I[row, col] = 1
    basis_projection = strf_.project_to_spatial_basis(I, spatial_basis)
    assert_equal(len(basis_projection), n_spatial_basis)

    # Recover image from basis projection
    weights = np.random.normal(size=n_spatial_basis)
    RF = strf_.make_image_from_spatial_basis(spatial_basis, weights)
    assert_equal(RF.shape[0], patch_size)
    assert_equal(RF.shape[1], patch_size)

    # Convolve with temporal basis
    n_samples = 100
    n_features = n_spatial_basis
    design_matrix = np.random.normal(size=(n_samples, n_features))
    features = strf_.convolve_with_temporal_basis(
        design_matrix, temporal_basis)
    assert_equal(features.shape[0], n_samples)
    assert_equal(features.shape[1], n_features * n_temporal_basis)

    # Design prior covariance
    PriorCov = strf_.design_prior_covariance(
        sigma_temporal=3., sigma_spatial=5.)
    assert_equal(PriorCov.shape[0], PriorCov.shape[1])
    assert_equal(PriorCov.shape[0], n_spatial_basis * n_temporal_basis)
