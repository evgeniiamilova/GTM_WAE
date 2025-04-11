import warnings
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from typing import Optional, Union, Tuple
from tqdm.auto import tqdm


def _initialize_grid(min_x, max_x, num_points_x, min_y, max_y, num_points_y):
    # We need to rewrite it to make it compatible with the 3D GTM
    x_points = torch.linspace(min_x, max_x, steps=num_points_x)
    y_points = torch.linspace(min_y, max_y, steps=num_points_y)
    x, y = torch.meshgrid(x_points, y_points, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=1)


class DataStandardizer:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.data_mean = None
        self.data_std = None

    @classmethod
    def nanstd(cls, x, dim, keepdim=False):

        result = torch.sqrt(
            torch.nanmean(
                torch.pow(torch.abs(x - torch.nanmean(x, dim=dim).unsqueeze(dim)), 2),
                dim=dim
            )
        )

        if keepdim:
            result = result.unsqueeze(dim)

        return result

    def fit_transform(self, X, axis=0):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if self.with_mean:
            mean_ = torch.nanmean(X, dim=axis, keepdim=True)
            self.data_mean = mean_

        if self.with_std:
            scale_ = self.nanstd(X, dim=axis, keepdim=True)
            self.data_std = scale_

        # Center the data if required
        if self.with_mean:
            X = X - self.data_mean
            mean_1 = torch.nanmean(X, dim=axis)
            if not torch.allclose(mean_1, torch.zeros_like(mean_1), atol=1e-8):
                warnings.warn(
                    "Numerical issues were encountered when centering the data. "
                    "Dataset may contain very large values. You may need to prescale your features."
                )
                X = X - mean_1

        # Scale the data if required
        if self.with_std:
            scale_ = torch.clamp(self.data_std, min=1e-8)  # Handle zeros in scale
            X = X / scale_

            if self.with_mean:
                mean_2 = torch.nanmean(X, dim=axis)
                if not torch.allclose(mean_2, torch.zeros_like(mean_2), atol=1e-8):
                    warnings.warn(
                        "Numerical issues were encountered when scaling the data. "
                        "The standard deviation of the data is probably very close to 0."
                    )
                    X = X - mean_2

        return X

class BaseGTM(ABC, nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    @abstractmethod
    def _init_weights(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _init_beta(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _init_grid(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _init_rbfs(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def kernel(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def e_step(self, data, distances):
        raise NotImplementedError()

    @abstractmethod
    def m_step(self, data, responsibilities):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abstractmethod
    def project(self, data):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data):
        raise NotImplementedError()


def spherical_distance_vectorized(xi, m):
    # Calculate dot product between each pair of vectors in xi and m
    dot_product = torch.matmul(xi, m.T)

    # Calculate norms for xi and m
    norm_i = torch.norm(xi, dim=1, keepdim=True)
    norm_j = torch.norm(m, dim=1, keepdim=True)

    # Calculate cosine of angles
    cos_angle = dot_product / (norm_i * norm_j.T)

    # Clamp values to ensure they are within valid range for acos
    cos_angle = torch.clamp(cos_angle, -1, 1)

    # Calculate angles and their squares
    angle = torch.acos(cos_angle)
    angle_squared = angle ** 2

    return angle_squared


class VanillaGTM(BaseGTM, ABC):
    def __init__(
            self,
            num_nodes: int,
            num_basis_functions: int,
            basis_width: float,
            reg_coeff: float,
            standardize: bool = True,
            max_iter: int = 100,
            tolerance: float = 1e-3,
            n_components: int = 2,
            use_cholesky: bool = False,
            seed: int = 1234,
            topology: str = 'square',
            device: str = "cpu",
    ):
        super().__init__(device=device)
        self.to(self.device)
        torch.manual_seed(seed)
        self.n_components = n_components
        if self.n_components == 2:
            assert np.sqrt(num_nodes).is_integer(), "num_nodes must be square"
            assert np.sqrt(num_basis_functions).is_integer(), "num_basis_functions must be square"
        elif self.n_components == 3:
            assert round(np.cbrt(num_nodes)).is_integer(), f"{num_nodes} {type(num_nodes)} must be a cube"
            assert round(np.cbrt(num_basis_functions)).is_integer(), "num_basis_functions must be a cube"
        self.num_nodes = num_nodes
        self.num_basis_functions = num_basis_functions
        self.basis_width = basis_width
        self.reg_coeff = reg_coeff
        self.standardize = standardize
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.use_cholesky = use_cholesky
        self.topology = topology

        # Here we will add parameters of GTM
        nodes, mu, basis_width = self._init_grid()
        self.nodes = nodes.to(self.device).double()
        self.mu = mu.to(self.device).double()
        self.basis_width = basis_width
        # always initialised after grid
        self.phi = self._init_rbfs()

        self.data_mean = None
        self.data_std = None
        self.weights = None
        self.beta = None

    def _init_weights(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(
            self.num_basis_functions + 1,
            data.shape[-1],
            dtype=torch.double,
            device=self.device,
        )

    def _init_beta(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.double, device=self.device)

    def hexagonal_grid(self, x_dim, y_dim):
        if x_dim < 2 or y_dim < 2 or x_dim != int(x_dim) or y_dim != int(y_dim):
            raise ValueError(f"Invalid grid dimensions: {x_dim}, {y_dim}")

        # Create meshgrid for hexagonal layout
        x = torch.linspace(0, x_dim - 1, x_dim)
        y = torch.linspace(y_dim - 1, 0, y_dim)
        X, Y = torch.meshgrid(x, y)

        # Clone Y before adjusting for hexagonal pattern to avoid in-place operation issues
        Y = Y.clone()
        Y[:, 1::2] += 0.5

        # Flatten and merge the X and Y grids
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

        # Scale the grid
        max_val = grid.abs().max()
        grid = grid * (2 / max_val)

        # Center the grid
        max_XY = grid.max(0).values
        grid[:, 0] = grid[:, 0] - max_XY[0] / 2
        grid[:, 1] = grid[:, 1] - max_XY[1] / 2

        return grid

    def rectangular_grid(self, x_dim, y_dim):
        if x_dim < 2 or y_dim < 2 or x_dim != int(x_dim) or y_dim != int(y_dim):
            raise ValueError(f"Invalid grid dimensions: {x_dim}, {y_dim}")

        # Generate a meshgrid
        x = torch.linspace(0, x_dim - 1, x_dim)
        y = torch.linspace(y_dim - 1, 0, y_dim)
        X, Y = torch.meshgrid(x, y)

        # Flatten and merge the X and Y grids
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

        # Scale the grid
        max_val = grid.abs().max()
        grid = grid * (2 / max_val)

        # Center the grid
        max_XY = grid.max(0).values
        grid[:, 0] = grid[:, 0] - max_XY[0] / 2
        grid[:, 1] = grid[:, 1] - max_XY[1] / 2

        return grid

    def rectangular_grid_louis(self, x_dim, y_dim):
        """
        Initializes a 2D grid of points between -1 and 1 for x and y dimensions.
        """
        x_points = torch.linspace(-1, 1, x_dim, device=self.device)
        y_points = torch.linspace(1, -1, y_dim, device=self.device)
        grid = torch.meshgrid(x_points, y_points, indexing="ij")
        x = grid[0]
        y = grid[1]
        return torch.stack([x.flatten(), y.flatten()], dim=1).to(self.device)


    def _init_grid(self):
        if self.n_components == 2:
            if self.topology == 'sphere':
                raise NotImplementedError
            elif self.topology == 'hexagon':
                # Calculate dimensions for the hexagonal grid
                dim = round(np.sqrt(self.num_nodes))
                nodes = self.hexagonal_grid(dim, dim)
                dim_basis = round(np.sqrt(self.num_basis_functions))
                mu = self.hexagonal_grid(dim_basis, dim_basis)

                # Scale and calculate basis function width as before
                mu = mu * (self.num_basis_functions / (self.num_basis_functions - 1))
                basis_width = self.basis_width * (mu[0, 1] - mu[1, 1])
            else:
                nodes = self.rectangular_grid(
                    int(np.sqrt(self.num_nodes)), int(np.sqrt(self.num_nodes))
                )
                mu = self.rectangular_grid(
                    int(np.sqrt(self.num_basis_functions)),
                    int(np.sqrt(self.num_basis_functions)),
                )

                mu = mu * (self.num_basis_functions / (self.num_basis_functions - 1))
                basis_width = self.basis_width * (mu[0, 1] - mu[1, 1])
            return nodes, mu, basis_width
        elif self.n_components == 3:

            if self.topology == 'sphere':
                nodes = self.spherical_grid_3d(self.num_nodes)
                mu = self.spherical_grid_3d(self.num_basis_functions)
                basis_width = self.basis_width * (mu[1, 2] - mu[0, 2])
            else:
                dim = round(np.cbrt(self.num_nodes))
                nodes = self.rectangular_grid_3d(dim, dim, dim)
                dim_basis = round(np.cbrt(self.num_basis_functions))
                mu = self.rectangular_grid_3d(dim_basis, dim_basis, dim_basis)

                mu = mu * (self.num_basis_functions / (self.num_basis_functions - 1))
                basis_width = self.basis_width * (mu[1, 2] - mu[0, 2])

            return nodes, mu, basis_width

    def rectangular_grid_3d(self, x_dim, y_dim, z_dim):
        if x_dim < 2 or y_dim < 2 or z_dim < 2 or x_dim != int(x_dim) or y_dim != int(y_dim) or z_dim != int(z_dim):
            raise ValueError(f"Invalid grid dimensions: {x_dim}, {y_dim}, {z_dim}")

        # Generate a meshgrid
        x = torch.linspace(0, x_dim - 1, x_dim)
        y = torch.linspace(y_dim - 1, 0, y_dim)
        z = torch.linspace(0, z_dim - 1, z_dim)
        X, Y, Z = torch.meshgrid(x, y, z)

        # Flatten and merge the X, Y, and Z grids
        grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        # Scale the grid
        max_val = grid.abs().max()
        grid = grid * (2 / max_val)

        # Center the grid
        max_XYZ = grid.max(0).values
        grid[:, 0] = grid[:, 0] - max_XYZ[0] / 2
        grid[:, 1] = grid[:, 1] - max_XYZ[1] / 2
        grid[:, 2] = grid[:, 2] - max_XYZ[2] / 2

        return grid

    def _init_rbfs(self):
        # if self.topology == 'square' or 'sphere':
        dist_nodes_rbfs = (
                torch.cdist(
                    self.nodes, self.mu, compute_mode="donot_use_mm_for_euclid_dist"
                ).to(self.device) ** 2
        )
        """
        elif self.topology == 'sphere':
            dist_nodes_rbfs = spherical_distance_vectorized(self.nodes, self.mu)

            pi = torch.pi
            theta_nodes = torch.acos(self.nodes[:, 2])  # θ for nodes
            phi_nodes = torch.atan2(self.nodes[:, 1], self.nodes[:, 0])  # φ for nodes
            theta_mu = torch.acos(self.mu[:, 2])  # θ for RBF centers
            phi_mu = torch.atan2(self.mu[:, 1], self.mu[:, 0])  # φ for RBF centers
            r_nodes = torch.sqrt((self.nodes ** 2).sum(dim=1))  # r for nodes
            r_mu = torch.sqrt((self.mu ** 2).sum(dim=1))  # r for RBF centers

            # Create mesh grids for vectorized operations
            r_nodes_mesh, r_mu_mesh = torch.meshgrid(r_nodes, r_mu, indexing='ij')
            phi_nodes_mesh, phi_mu_mesh = torch.meshgrid(phi_nodes, phi_mu, indexing='ij')
            theta_nodes_mesh, theta_mu_mesh = torch.meshgrid(theta_nodes, theta_mu, indexing='ij')

            # Calculate the differences and distances
            delta_r = torch.abs(r_nodes_mesh - r_mu_mesh)
            delta_phi = torch.abs(phi_nodes_mesh - phi_mu_mesh)

            dist_nodes_rbfs = delta_r ** 2 + 2 * r_nodes_mesh * r_mu_mesh * (
                1 - (torch.sin(theta_nodes_mesh) * torch.sin(theta_mu_mesh) * torch.cos(delta_phi)
                     + torch.cos(theta_nodes_mesh) * torch.cos(theta_mu_mesh))
            )
            dist_nodes_rbfs = dist_nodes_rbfs.to(self.device)
            """
        phi = torch.exp(-0.5 * dist_nodes_rbfs / self.basis_width ** 2)
        return torch.cat(
            (phi, torch.ones(phi.size(0), 1, dtype=torch.float64, device=self.device)),
            dim=1,
        )

    def _standardize(self, x, with_mean=True, with_std=True):
        standardizer = DataStandardizer(with_mean, with_std)
        x = standardizer.fit_transform(x)
        return x

    @staticmethod
    def _log_matrix_stats(matrix, message=""):
        logging.debug(
            f"{message}: "
            f"max - {matrix.max()} "
            f"min - {matrix.min()} "
            f"mean - {matrix.mean()} "
            f"std - {matrix.std()} "
        )

    def kernel(self, a, b):
        return (
                torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist").to(
                    self.device
                )
                ** 2
        )

    def spherical_grid_3d(self, num_points):
        # Use spherical coordinates to create a uniform grid on a sphere
        indices = torch.arange(0, num_points, dtype=torch.float64) + 0.5

        phi = torch.acos(1 - 2 * indices / num_points)  # polar angle
        theta = torch.pi * (1 + 5 ** 0.5) * indices  # azimuthal angle

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        return torch.stack([x, y, z], dim=1)

    def e_step(self, data, distances):
        """
        :param data: number of training points
        :param distances: A KxN matrix of squared distances between data points and mixture centers.
        :return:
        A num_latents x num_data matrix of responsibilities computed using softmax.
        Log-likelihood of data under the Gaussian mixture.
        """
        exponent_terms = -(self.beta / 2) * distances
        responsibilities = torch.softmax(exponent_terms, dim=0, dtype=torch.float64)
        resp_logsum = torch.logsumexp(exponent_terms, dim=0)

        llhs = (
                resp_logsum
                + (data.shape[-1] / 2) * torch.log(self.beta / (2.0 * torch.pi))
                - torch.log(torch.tensor(self.num_nodes, dtype=torch.float64))
        )
        return responsibilities, llhs

    def m_step(self, data, responsibilities):
        # TODO: check if everything is optimal here in terms of the linear algebra/numerical stability
        G = torch.diag(responsibilities.sum(dim=1))
        # A is phi'* G * phi + lambda * I / beta
        # print(self.phi.device, self.G.device, data.device)
        A = self.phi.T @ G @ self.phi + self.reg_coeff / self.beta * torch.eye(
            self.num_basis_functions + 1, dtype=torch.double, device=self.device
        )
        B = self.phi.T @ (responsibilities @ data)
        if self.use_cholesky:
            # here we can use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(A)
            Y = torch.linalg.solve(L, B)
            self.weights = torch.linalg.solve(L.T, Y)
        else:
            self.weights = torch.linalg.solve(A, B)
        distance = self.kernel(self.phi @ self.weights, data)
        self.beta = (data.shape[0] * data.shape[1]) / (
                responsibilities * distance
        ).sum()
        return distance

    def _fit_loop(self, data):
        # Initial llh
        llh_old = torch.tensor(0).double()
        
        # Initialize the distance matrix

        init_space_posit = self.phi @ self.weights  # Initial space positions (Y-matrix)
        self.init_space_posit = deepcopy(init_space_posit)
        distances = self.kernel(init_space_posit, data)
        # Calculate the distance matrix in the data space
        self._log_matrix_stats(distances, "First distances RBFs-data in N-dimensions")

        pbar = tqdm(range(self.max_iter))
        for index, _ in enumerate(pbar):
            responsibilities, llhs = self.e_step(data, distances)
            # self.responsibilities = responsibilities
            llh = torch.mean(llhs)  # normalisation by data
            llh_diff = torch.abs(llh_old - llh)

            # Logging part
            info = {
                "LLh": float(torch.round(llh, decimals=5)),
                "deltaLLh": float(torch.round(llh_diff, decimals=5)),
                "beta": float(torch.round(self.beta, decimals=5)),
            }
            logging.info(" ".join([f"{k}: {v}" for k, v in info.items()]))
            pbar.set_postfix(info)

            # Convergence check part
            if llh_diff < self.tolerance:  # Helena checks for several cycles
                break
            llh_old = llh
            if index < self.max_iter - 1:
                distances = self.m_step(data, responsibilities)

    def fit(self, x):
        x = x.to(self.device)
        # Calculate mean and standard deviation along each column (axis 0)

        if self.standardize:    
            # Scale the tensor using mean and standard deviation
            x = self._standardize(x, with_mean=True, with_std=True)

        self.data_mean = torch.mean(x, dim=0)
        self.data_std = torch.std(x, dim=0)
        
        # initialise weights and beta from the data
        self.weights = self._init_weights(x)
        self.weights[-1, :] = self.data_mean
        self.beta = self._init_beta()
        self._fit_loop(x)

    def project(self, x):
        x = x.to(self.device)
        if self.standardize:
            x = self._standardize(x, with_mean=True, with_std=True)
        distance = self.kernel(self.phi @ self.weights, x)
        responsibilities, llhs = self.e_step(x, distance)
        return responsibilities, llhs


class BishopGTM(VanillaGTM):
    def __init__(
            self, pca_engine="sklearn", pca_scale=True, pca_lowrank=False, *args, **kwargs
    ):
        self.pca_engine = pca_engine
        self.pca_scale = pca_scale
        self.pca_lowrank = pca_lowrank
        super(BishopGTM, self).__init__(*args, **kwargs)

    def _init_beta_mixture_components(self):
        y = self.phi @ self.weights
        lat_space_dist = (
                torch.cdist(
                    y, y, compute_mode="donot_use_mm_for_euclid_dist"
                ).to(self.device)
                ** 2
        )  # TODO low_dim_space
        self._log_matrix_stats(
            lat_space_dist, "Distances between nodes in N-dimensions"
        )

        # Add a large number to the diagonal (similar to realmax in MATLAB)
        lat_space_dist.fill_diagonal_(torch.finfo(lat_space_dist.dtype).max)
        # lat_space_dist = lat_space_dist + torch.finfo(lat_space_dist.dtype).max

        # Find the average distance between nearest neighbors
        mean_nn = torch.min(lat_space_dist, dim=1).values.mean()
        # torch.min(lat_space_dist, dim=1).values.mean()

        # Calculate options for the initial beta
        beta = mean_nn / 2
        return beta

    def _pca_torch(self, data):
        if self.standardize:
            # Assuming data is already standardized or you want to standardize it here
            pca_data = data
        else:
            pca_data = data - data.mean(dim=0)

        if self.pca_lowrank:
            _, singular_values, eigenvectors = torch.pca_lowrank(
                pca_data, q=20, center=False
            )
            eigenvectors = eigenvectors[:, : self.n_components + 1].T
        else:
            _, singular_values, eigenvectors = torch.linalg.svd(
                pca_data, full_matrices=False
            )
            eigenvectors = eigenvectors[: self.n_components + 1, :]
        
        eigenvalues = singular_values[: self.n_components + 1].reshape(-1, 1)**2 / (data.shape[0] - 1)
        
        if self.pca_scale:
            # Scale by sqrt(eigenvalues)
            eigenvectors = eigenvectors * torch.sqrt(eigenvalues)
        return eigenvectors, eigenvalues

    def _pca_sklearn(self, data):
        if self.standardize:
            # If you need standardization, do it here, else just use data as is.
            pca_data = data
        else:
            pca_data = data - data.mean(dim=0)

        pca = PCA(n_components=self.n_components + 1)
        pca.fit(pca_data.cpu().numpy())

        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        if self.pca_scale:
            # The provided code scales eigenvectors by sqrt of singular values
            # Torch code scales eigenvectors by sqrt of eigenvalues.
            # Here we use the given scaling for sklearn (as stated in the code):
            eigenvectors = eigenvectors * np.sqrt(eigenvalues[:, np.newaxis])#np.sqrt(pca.singular_values_[:, np.newaxis])

        eigenvectors = torch.from_numpy(eigenvectors).double()
        eigenvalues = torch.from_numpy(eigenvalues).double()

        return eigenvectors, eigenvalues

    def _get_pca(self, data):
        if self.pca_engine == "sklearn":
            eigenvectors, eigenvalues = self._pca_sklearn(data)
        elif self.pca_engine == "torch":
            eigenvectors, eigenvalues = self._pca_torch(data)
        elif isinstance(self.pca_engine, dict):
            eigenvectors, eigenvalues = self.pca_engine['eigenvectors'], self.pca_engine['eigenvalues']
        else:
            raise ValueError(f"Unknown pca_engine: {self.pca_engine}")
        eigenvectors = eigenvectors.to(data.dtype).to(self.device)
        eigenvalues = eigenvalues.to(data.dtype).to(self.device)
        return eigenvectors, eigenvalues

    def _init_weights(self, eigenvectors):
        ## Can't be run on GPU. see https://github.com/pytorch/pytorch/issues/71222
        self.nodes = (self.nodes - self.nodes.mean(dim=0)) / (self.nodes.std(dim=0))
        nodes_pca_projection = (
                self.nodes @ eigenvectors[: self.n_components]
        )  # TODO: check if shapes are good
        return torch.linalg.lstsq(
            self.phi, nodes_pca_projection, driver="gels"
        ).solution  # .cuda()

    def _init_beta(self, eigenvalues) -> torch.Tensor:
        # Calculating the initial beta
        beta_1 = self._init_beta_mixture_components()
        beta_2 = eigenvalues[self.n_components]

        logging.debug(f"Beta from distances: {beta_1}")
        logging.debug(f"Beta from PCA: {beta_2}")
        return max(beta_1, beta_2)

    def fit(self, x):
        x = x.to(self.device)
       
        if self.standardize:
            # Calculate mean and standard deviation along each column (axis 0)

            # Scale the tensor using mean and standard deviation
            x = self._standardize(x, with_mean=True, with_std=True)
            #print(x, x.shape)
        self.data_mean = torch.mean(x, dim=0)
        self.data_std = torch.std(x, dim=0)
        # initialise weights and beta from the data
        eigenvectors, eigenvalues = self._get_pca(x)
        self.weights = self._init_weights(eigenvectors)
        self.weights[-1, :] = self.data_mean
        self.beta = self._init_beta(eigenvalues)
        self.beta_init = deepcopy(self.beta)
        self._fit_loop(x)


class KernelGTM(BishopGTM):
    """
    ‖ψ (x) − ym‖2 = 〈ψ (x) , ψ (x)〉 + 〈ym, ym〉 − 2 〈ψ (x) , ym〉
    """

    def __init__(self, seed, kernel_matrix, *args, **kwargs):
        torch.manual_seed(seed)
        self.kernel_matrix = kernel_matrix
        super().__init__(*args, **kwargs)

    def _init_beta(self, init_kernel, *args, **kwargs) -> torch.Tensor:

        return (init_kernel.shape[0] * init_kernel.shape[1]) / torch.sum(
            init_kernel / self.num_nodes
        )

    """
    def kernel(self, lambda_phi, kernel_matrix):

        Result = torch.zeros(self.num_nodes, self.kernel_matrix.shape[0])

        # Precompute the dot products needed
        LPhiM = lambda_phi.T  # shape (n_individuals, n_nodes)

        # Compute the distances
        for i in range(self.num_nodes):
            # Efficiently compute the inner product and scale
            LPhim = LPhiM[:, i]  # Select the i-th column
            thefloat = torch.dot(LPhim, torch.matmul(self.kernel_matrix, LPhim))

            # Update distance matrix
            for j in range(self.kernel_matrix.shape[0]):
                Result[i, j] = torch.dot(self.kernel_matrix[j], (self.kernel_matrix[j])) + thefloat - 2 * torch.dot(self.kernel_matrix[j], LPhim)
        print(Result)
        return Result
    """

    def kernel(self, lambda_phi, kernel_matrix):
        lambda_phi = lambda_phi.T
        # Calculate y_y for all nodes using batched matrix multiplication
        # (N, D) x (D, D) -> (N, D) then (N, D) x (D, N) -> (N,) for each node's y_y
        y_y = torch.einsum(
            "ni,ij,nj->n", lambda_phi.T, kernel_matrix, lambda_phi.T
        )  # TODO rewrite in a normal fashion
        # y_y = (lambda_phi.T @ kernel_matrix @ lambda_phi).diag()

        # Calculate k_phi for all nodes using matrix multiplication
        # (D, D) x (D, N) -> (D, N)
        k_phi = torch.mm(kernel_matrix, lambda_phi)

        # Get the diagonal of kernel_matrix, which remains constant for all nodes
        k_nn = torch.diag(kernel_matrix)

        # Now, construct J_s matrix by combining the above computations
        # Use broadcasting for k_nn and y_y which are 1D tensors
        J = k_nn.unsqueeze(0) + y_y.unsqueeze(1) - 2 * k_phi.T
        print(J)
        return J

    def m_step(self, data, responsibilities):
        # TODO: check if everything is optimal here in terms of the linear algebra/numerical stability
        G = torch.diag(responsibilities.sum(dim=1))
        # A is phi'* G * phi + lambda * I / beta
        # print(self.phi.device, self.G.device, data.device)
        A = self.phi.T @ G @ self.phi + self.reg_coeff / self.beta * torch.eye(
            self.num_basis_functions + 1, dtype=torch.double, device=self.device
        )
        B = self.phi.T @ (responsibilities)  # @ data)
        if self.use_cholesky:
            # here we can use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(A)
            Y = torch.linalg.solve(L, B)
            self.weights = torch.linalg.solve(L.T, Y)
        else:
            self.weights = torch.linalg.solve(A, B)
        distance = self.kernel(self.phi @ self.weights, data)
        self.beta = (data.shape[0] * data.shape[1]) / (
                responsibilities * distance
        ).sum()
        # print(self.beta)
        self.responsabilities = responsibilities
        return distance

    def fit(self, x):
        x = x.to(self.device)
        # Calculate mean and standard deviation along each column (axis 0)
       
        if self.standardize:
            
            # Scale the tensor using mean and standard deviation
            x = self._standardize(x, with_mean=True, with_std=True)

        # initialise weights and beta from the data
        self.data_mean = torch.mean(x, dim=0)
        self.data_std = torch.std(x, dim=0)
        eigenvectors, eigenvalues = self._get_pca(x)
        self.weights = self._init_weights(eigenvectors)
        self.weights[-1, :] = self.data_mean
        init_kernel = self.kernel(self.phi @ self.weights, x)
        self.beta = self._init_beta(init_kernel)
        print(f'Init {self.beta}')
        self._fit_loop(x)


class BishopGTM3D(BishopGTM):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def rectangular_grid(self, x_dim, y_dim, z_dim):
        if x_dim < 2 or y_dim < 2 or z_dim < 2 or x_dim != int(x_dim) or y_dim != int(y_dim) or z_dim != int(z_dim):
            raise ValueError(f"Invalid grid dimensions: {x_dim}, {y_dim}, {z_dim}")

        # Generate a meshgrid
        x = torch.linspace(0, x_dim - 1, x_dim)
        y = torch.linspace(y_dim - 1, 0, y_dim)
        z = torch.linspace(0, z_dim - 1, z_dim)
        X, Y, Z = torch.meshgrid(x, y, z)

        # Flatten and merge the X, Y, and Z grids
        grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        # Scale the grid
        max_val = grid.abs().max()
        grid = grid * (2 / max_val)

        # Center the grid
        max_XYZ = grid.max(0).values
        grid[:, 0] = grid[:, 0] - max_XYZ[0] / 2
        grid[:, 1] = grid[:, 1] - max_XYZ[1] / 2
        grid[:, 2] = grid[:, 2] - max_XYZ[2] / 2

        return grid

    def _init_grid(self):
        # Calculate dimensions based on square roots; might need revision for 3D to make it cubic
        dim = round(np.cbrt(self.num_nodes))  # Using cubic root for 3D
        nodes = self.rectangular_grid(dim, dim, dim)
        dim_basis = round(np.cbrt(self.num_basis_functions))
        mu = self.rectangular_grid(dim_basis, dim_basis, dim_basis)

        # Scale and calculate basis function width as before
        mu = mu * (self.num_basis_functions / (self.num_basis_functions - 1))
        basis_width = self.basis_width * (mu[1, 2] - mu[0, 2])

        return nodes, mu, basis_width

    def fit_transform(self, x, y=None):
        pass

    def transform(self, x, y=None):
        pass


class ScikitLearnGTM(BishopGTM):
    def __init__(self, *args, **kwargs):
        """
        Initializes the ScikitLearnGTM class.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self._magnification_factor: Optional[torch.Tensor] = None  # Cache the magnification factor

    def fit(self, x: Union[torch.Tensor, np.ndarray], y: Optional[torch.Tensor] = None) -> "ScikitLearnGTM":
        """
        Fits the GTM model to the data.

        Args:
            x: Input data, either as a torch.Tensor or numpy array.
            y: (Optional) Target data, not used in GTM but added for API consistency.

        Returns:
            ScikitLearnGTM: The fitted model.
        """
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.from_numpy(x).double()
        # Call the fit method from the parent class BishopGTM
        super().fit(x)
        return self

    def transform(self, x: Union[torch.Tensor, np.ndarray], y: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Projects the data into the latent space of the GTM model.

        Args:
            x: Input data, either as a torch.Tensor or numpy array.
            y: (Optional) Target data, not used in GTM but added for API consistency.

        Returns:
            np.ndarray: The coordinates in latent space.
        """
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.from_numpy(x).double()
        x = x.to(self.device)
        if self.standardize:
            x = self._standardize(x, with_mean=True, with_std=True)

        distance = self.kernel(self.phi @ self.weights, x)
        responsibilities, _ = self.e_step(x, distance)
        coordinates = responsibilities.T @ self.nodes
        return coordinates.cpu().numpy()

    def fit_transform(self, x: Union[torch.Tensor, np.ndarray], y: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Fits the GTM model and then projects the data into the latent space.

        Args:
            x: Input data, either as a torch.Tensor or numpy array.
            y: (Optional) Target data, not used in GTM but added for API consistency.

        Returns:
            np.ndarray: The coordinates in latent space.
        """
        if isinstance(x, torch.Tensor):
            pass
        else:
            x = torch.from_numpy(x).double()
        self.fit(x, y)
        return self.transform(x, y)

    def compute_phi_and_dphi(self, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the basis functions and their derivatives at a given latent point x_i.

        Args:
            x_i: A latent point in the L-dimensional latent space (tensor of shape [L]).

        Returns:
            phi_i: The basis function values at x_i (tensor of shape [M], where M is the number of basis functions).
            dphi_i: The derivatives of the basis functions with respect to x_i (tensor of shape [M, L]).
        """
        # Difference between centers and the current latent point x_i
        diff = x_i - self.mu  # Shape: [M, L]

        # Compute squared distances
        squared_dist = torch.sum(diff ** 2, dim=1)  # Shape: [M]

        # Compute basis function values at x_i
        exponent = -squared_dist / (2 * self.basis_width ** 2)
        phi_i = torch.exp(exponent)  # Shape: [M]

        # Compute the derivatives of the basis functions at x_i
        dphi_i = -diff / (self.basis_width ** 2) * phi_i.unsqueeze(1)  # Shape: [M, L]

        return phi_i, dphi_i

    # @property
    def magnification_factor(self) -> torch.Tensor:
        """
        Computes the magnification factor dA'/dA for the GTM algorithm, based on differential geometry.

        Returns:
            torch.Tensor: The magnification factor for each point in latent space.
        """
        if self._magnification_factor is None:
            if self.weights is None:
                raise RuntimeError(
                    "Model has not been fitted yet. Please call 'fit' before accessing the magnification factor.")
            W_no_bias = self.weights[:-1, :]  # Exclude bias term; Shape: [M, D]

            magnification_factors = []

            for i in range(self.nodes.shape[0]):  # Iterate over each latent node
                x_i = self.nodes[i]  # Shape: [L]

                # Compute phi_i and dphi_i at x_i
                phi_i, dphi_i = self.compute_phi_and_dphi(x_i)  # Shapes: [M], [M, L]

                # Compute A = W_no_bias^T dphi_i
                A = W_no_bias.T @ dphi_i  # Shape: [D, L]

                # Compute B = A^T A
                B = A.T @ A  # Shape: [L, L]

                # Compute the determinant of B
                det_value = torch.det(B)

                # Handle potential numerical issues
                det_value = torch.abs(det_value) + 1e-12

                # Compute the magnification factor
                magnification_factor = torch.sqrt(det_value)

                magnification_factors.append(magnification_factor)
            # Convert the list to a tensor
            self._magnification_factor = torch.stack(magnification_factors)

        return self._magnification_factor
