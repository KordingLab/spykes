import numpy as np
import matplotlib.pyplot as plt


class STRF(object):
    '''Allows the estimation of spatiotemporal receptive fields

    Args:
        patch_size (int): Dimension of the square patch spanned by the
            spatial basis.
        sigma (float): Standard deviation of the Gaussian distribution.
        n_spatial_basis (int): Number of spatial basis functions for the
            Gaussian basis (has to be a perfect square).
        n_temporal_basis (int): Number of temporal basis functions.
    '''
    def __init__(self, patch_size=100, sigma=0.5,
                 n_spatial_basis=25, n_temporal_basis=3):
        self.patch_size = patch_size
        self.sigma = sigma
        self.n_spatial_basis = n_spatial_basis
        self.n_temporal_basis = n_temporal_basis

    def make_2d_gaussian(self, center=(0, 0)):
        '''Makes a 2D Gaussian filter with arbitary mean and variance.

        Args:
            center (tuple): The coordinates of the center of the Gaussian,
                specified as :data:`(row, col)`. The center of the image is
                :data:`(0, 0)`.

        Returns:
            numpy array: The Gaussian mask.
        '''
        sigma = self.sigma
        n_rows = (self.patch_size - 1.) / 2.
        n_cols = (self.patch_size - 1.) / 2.

        y, x = np.ogrid[-n_rows: n_rows + 1, -n_cols: n_cols + 1]
        y0, x0 = center[1], center[0]
        gaussian_mask = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) /
                               (2. * sigma ** 2))
        gaussian_mask[gaussian_mask <
                      np.finfo(gaussian_mask.dtype).eps *
                      gaussian_mask.max()] = 0
        gaussian_mask = 1. / gaussian_mask.max() * gaussian_mask
        return gaussian_mask

    def make_gaussian_basis(self):
        '''Makes a list of Gaussian filters.

        Returns:
            list: A list where each entry is a 2D array of size
            :data:`(patch_size, patch_size)` specifing the spatial basis.
        '''
        spatial_basis = list()
        n_tiles = np.sqrt(self.n_spatial_basis)
        n_pixels = self.patch_size
        centers = np.linspace(start=(-n_pixels / 2. +
                                     n_pixels / (n_tiles + 1.)),
                              stop=(n_pixels / 2. -
                                    n_pixels / (n_tiles + 1.)),
                              num=n_tiles)

        for y in range(n_tiles.astype(int)):
            for x in range(n_tiles.astype(int)):
                gaussian_mask = self.make_2d_gaussian(center=(centers[x],
                                                              centers[y]))
                spatial_basis.append(gaussian_mask)
        return spatial_basis

    def make_cosine_basis(self):
        '''Makes a spatial cosine and sine basis.

        Returns:
            list: A list where each entry is a 2D array of size
            :data:`(patch_size, patch_size)` specifing the spatial basis.
        '''
        patch_size = self.patch_size
        cosine_mask = np.zeros((patch_size, patch_size))
        sine_mask = np.zeros((patch_size, patch_size))
        for row in np.arange(patch_size):
            for col in np.arange(patch_size):
                theta = np.arctan2(patch_size / 2 - row, col - patch_size / 2)
                cosine_mask[row, col] = np.cos(theta)
                sine_mask[row, col] = np.sin(theta)

        spatial_basis = list()
        spatial_basis.append(cosine_mask)
        spatial_basis.append(sine_mask)
        return spatial_basis

    def visualize_gaussian_basis(self, spatial_basis, color='Greys',
                                 show=True):
        '''Plots spatial basis functions in a tile of images.

        Args:
            spatial_basis (list): A list where each entry is a 2D array of size
                :data:`(patch_size, patch_size)` specifing the spatial basis.
            color (str): The color for the figure.
            show (bool): Whether or not to show the image when it is plotted.
        '''
        n_spatial_basis = len(spatial_basis)
        n_tiles = np.sqrt(n_spatial_basis)
        plt.figure(figsize=(7, 7))
        for i in range(n_spatial_basis):
            plt.subplot(np.int(n_tiles), np.int(n_tiles), i + 1)
            plt.imshow(spatial_basis[i], cmap=color)
            plt.axis('off')
        if show:
            plt.show()

    def project_to_spatial_basis(self, image, spatial_basis):
        '''Projects a given image into a spatial basis.

        Args:
            image (numpy array): Image that must be projected into the
                spatial basis 2D array of size :data:`(patch_size,
                patch_size)`.
            spatial_basis (list): A list where each entry is a 2D array of size
                :data:`(patch_size, patch_size)` specifing the spatial basis.

        Returns:
            numpy array: A 1D array of coefficients for the projection.
        '''
        n_spatial_basis = len(spatial_basis)
        weights = np.zeros(n_spatial_basis)
        for b in range(n_spatial_basis):
            weights[b] = np.sum(spatial_basis[b] * image)
        return weights

    def make_image_from_spatial_basis(self, basis, weights):
        '''Recovers an image from a basis given a set of weights.

        Args:
            spatial_basis (list): A list where each entry is a 2D array of size
                :data:`(patch_size, patch_size)` specifing the spatial basis.
            weights (numpy array): A 1D array of coefficients.

        Returns:
            numpy array: 2D array of size :data:`(patch_size, patch_size)`, the
            resulting image.
        '''
        image = np.zeros(basis[0].shape)
        n_basis = len(basis)
        for b in range(n_basis):
            image += weights[b] * basis[b]
        return image

    def make_raised_cosine_temporal_basis(self, time_points, centers, widths):
        '''Makes a series of raised cosine temporal basis.

        Args:
            time_points (numpy array): List of time points at which the basis
                function is computed.
            centers (numpy array or list): List of coordinates at which each
                basis function is centered 1D array of size
                :data:`(n_temporal_basis)`.
            widths (numpy array or list): List of widths, one per basis
                function; each is a 1D array of size
                :data:`(n_temporal_basis)`.

        Returns:
            numpy array: 2D array of size :data:`(n_basis, n_timepoints)`.
        '''
        temporal_basis = list()
        for idx, center in enumerate(centers):
            this_basis = np.zeros(len(time_points))
            arg_cos = (time_points - center) * np.pi / widths[idx] / 2.
            arg_cos[arg_cos > np.pi] = np.pi
            arg_cos[arg_cos < -np.pi] = -np.pi
            this_basis = 0.5 * (np.cos(arg_cos) + 1.)
            temporal_basis.append(this_basis)
        temporal_basis = np.transpose(np.array(temporal_basis))
        return temporal_basis

    def convolve_with_temporal_basis(self, design_matrix, temporal_basis):
        '''Convolves a design matrix with a temporal basis function.

        Convolves each column of the design matrix with a series of temporal
        basis functions.

        Args:
            design_matrix (numpy array): 2D array of size
                :data:`(n_samples, n_features)`.
            temporal_basis (numpy array): 2D array of size
                :data:`(n_basis, n_timepoints)`.

        Returns:
            numpy array: 2D array of size
            :data:`(n_samples, n_features * n_basis)`.
        '''
        n_temporal_basis = temporal_basis.shape[1]
        n_features = design_matrix.shape[1]
        convolved_design_matrix = list()
        for feat in range(n_features):
            for b in range(n_temporal_basis):
                convolved_design_matrix.append(
                    np.convolve(design_matrix[:, feat], temporal_basis[:, b],
                                mode='same'))
        convolved_design_matrix = \
            np.transpose(np.array(convolved_design_matrix))
        return convolved_design_matrix

    def design_prior_covariance(self, sigma_temporal=2., sigma_spatial=5.):
        '''Design a prior covariance matrix for STRF estimation.

        Args:
            sigma_temporal (float): Standard deviation of temporal prior
                covariance.
            sigma_spatial (float): Standard deviation of spatial prior
                covariance.

        Returns:
            numpy array: 2-d array of size :data:`(n_spatial_basis *
            n_temporal_basis, n_spatial_basis * n_temporal_basis)`, the
            ordering of rows and columns is so that all temporal basis are
            consecutive for each spatial basis.
        '''
        n_spatial_basis = self.n_spatial_basis
        n_temporal_basis = self.n_temporal_basis

        n_features = n_temporal_basis * n_spatial_basis
        sp_covariance = np.zeros([n_features, n_features])
        te_covariance = np.zeros([n_features, n_features])
        prior_covariance = np.zeros([n_features, n_features])
        for i in np.arange(0, n_features):
            # Get spatiotemporal indices
            s_i = np.floor(np.float(i) %
                           (n_temporal_basis * n_spatial_basis) /
                           n_temporal_basis)
            t_i = i % n_temporal_basis
            # Convert spatial indices to (x,y) coordinates
            x_i = s_i % np.sqrt(n_spatial_basis)
            y_i = np.floor(np.float(s_i) / np.sqrt(n_spatial_basis))

            for j in np.arange(i, n_features):
                # Get spatiotemporal indices
                s_j = np.floor(np.float(j) %
                               (n_temporal_basis * n_spatial_basis) /
                               n_temporal_basis)
                t_j = j % n_temporal_basis
                # Convert spatial indices to (x,y) coordinates
                x_j = s_j % np.sqrt(n_spatial_basis)
                y_j = np.floor(np.float(s_j) / np.sqrt(n_spatial_basis))

                sp_covariance[i, j] = np.exp(-1. / (sigma_spatial ** 2) *
                                             ((x_i - x_j) ** 2 +
                                              (y_i - y_j) ** 2))
                sp_covariance[j, i] = sp_covariance[i, j]
                te_covariance[i, j] = np.exp(-1. / (sigma_temporal ** 2) *
                                             (t_i - t_j) ** 2)
                te_covariance[j, i] = te_covariance[i, j]

        prior_covariance = sp_covariance * te_covariance
        prior_covariance = 1. / np.max(prior_covariance) * prior_covariance
        return prior_covariance
