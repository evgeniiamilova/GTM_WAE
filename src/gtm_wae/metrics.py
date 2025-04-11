import math
import torch
import math
import torch
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Union, Callable, Dict, Tuple, Any, Sequence
from numba import njit, prange, jit
import torch
#from geomloss import SamplesLoss  # GeomLoss library for OT computation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
#from umap import UMAP
from matplotlib import pyplot as plt
import numpy.typing as npt
from scipy.spatial import cKDTree

from numba import jit, njit, prange
from typing import List, Any, Callable, Dict, Union, Optional, Tuple

import scipy.linalg as spl
import ot
from typing import Tuple
from numpy.typing import NDArray

import jax.numpy as jnp
from ott.math import matrix_square_root
from ott.math.fixed_point_loop import fixpoint_iter
import functools
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
from ott.math import matrix_square_root
from numba import njit, prange
from pandarallel import pandarallel

import jax.numpy as jnp
from ott.solvers import linear
from ott.solvers.linear import sinkhorn
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.geometry import geometry
from ott.geometry.geometry import Geometry
from ott.geometry.geometry import Geometry
from ott.solvers.linear.sinkhorn import Sinkhorn


import pandas as pd


def calculate_nn_preservation(
    X_high_dim: np.ndarray,
    X_low_dim: np.ndarray,
    k_neighbors: Union[int, List[int]],
    high_dim_indexes: np.ndarray = None,
    high_dim_metric: str = 'euclidean'
) -> Union[float, List[float]]:
    """
    Calculate the nearest neighbor preservation scores for different k values.

    Args:
        X_high_dim (np.ndarray): High-dimensional data of shape (n_samples, n_features_high).
        X_low_dim (np.ndarray): Low-dimensional data of shape (n_samples, n_features_low).
        k_neighbors (int or List[int]): Single k value or list of k values.
        high_dim_indexes (np.ndarray, optional): Precomputed high-dimensional neighbor indices.
        high_dim_metric (str, optional): Metric to use in low-dimensional space

    Returns:
        float or List[float]: Preservation score(s) as a percentage.
    """
    # Ensure k_neighbors is a list
    if isinstance(k_neighbors, int):
        k_list = [k_neighbors]
        single_k = True
    else:
        k_list = k_neighbors
        single_k = False

    nn_preservation_scores = []

    # Precompute high-dimensional nearest neighbors if not provided
    if high_dim_indexes is None:
        max_k = max(k_list)
        nbrs_high = NearestNeighbors(n_neighbors=max_k + 1, metric=high_dim_metric).fit(X_high_dim)
        _, indices_high = nbrs_high.kneighbors(X_high_dim)
        indices_high = indices_high[:, 1:]  # Exclude self
    else:
        indices_high = high_dim_indexes
        max_k = indices_high.shape[1]

    # Precompute nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=max_k + 1).fit(X_low_dim)
    _, indices_low = nbrs_low.kneighbors(X_low_dim)
    indices_low = indices_low[:, 1:]  # Exclude self

    n_samples = X_high_dim.shape[0]

    for k in k_list:
        indices_high_k = indices_high[:, :k]  # shape (n_samples, k)
        indices_low_k = indices_low[:, :k]    # shape (n_samples, k)

        # Vectorized computation
        combined_indices = np.concatenate((indices_high_k, indices_low_k), axis=1)
        sorted_indices = np.sort(combined_indices, axis=1)
        diffs = np.diff(sorted_indices, axis=1)
        overlaps_per_sample = np.sum(diffs == 0, axis=1)
        overlap_counts = overlaps_per_sample / k
        avg_preservation = np.mean(overlap_counts) * 100
        nn_preservation_scores.append(avg_preservation)

    if single_k:
        return nn_preservation_scores[0]
    else:
        return nn_preservation_scores

def threshold_analysis(dist_mat: np.ndarray, thresh: float) -> tuple[float, float]:
    """
    1) Identify rows 'below threshold': rows that have at least one distance < thresh.
    2) Among those rows, calculate the percentage that also have
       at least one neighbor with distance < thresh.
    
    Parameters
    ----------
    dist_mat : np.ndarray
        An N×N distance matrix.
    thresh : float
        Distance threshold.
    
    Returns
    -------
    tuple of (float, float)
        - The percentage of all rows that are 'below threshold'.
        - The percentage of those 'below threshold' rows
          that have at least 1 neighbor < thresh (relative to that same row).
    """
    # Step 1) Create a boolean mask of where distances are below threshold
    below_mask = (dist_mat < thresh)  # shape (N, N)
    
    # Step 2) Find which rows have ANY distance < thresh
    row_below = np.any(below_mask, axis=1)  # shape (N,)
    n_rows_below = np.sum(row_below)
    
    # Overall fraction (in %) of rows that have any distance < thresh
    frac_rows_below = (n_rows_below / dist_mat.shape[0]) * 100

    return tc_threshold_analysis


def llh_threshold_by_limit(llhs: np.ndarray, lower_limit: float = 0.05) -> Tuple[float, int]:
    """
    Determine a log-likelihood threshold based on a specified lower percentile.
    
    Parameters
    ----------
    llhs : np.ndarray
        Array of log-likelihood scores.
    lower_limit : float, optional
        The percentile at which to determine the threshold. Must be between 0 and 1.
        Default is 0.05 (i.e., 5%).
        
    Returns
    -------
    float
        The log-likelihood threshold at the specified percentile.
    int
        The approximate number of scores that fall below or at this threshold.
        
    Notes
    -----
    - This function uses `np.quantile` to directly compute the threshold, avoiding any binning.
    - If you specifically need discrete bins or a coarser granularity for some reason (e.g., 
      in histogram analysis), you can revert to binning strategies.
    - Values for `lower_limit` above 0.5 would find an upper threshold (e.g., 0.95 for the 95th percentile).
    
    Examples
    --------
    >>> import numpy as np
    >>> llhs = np.array([-1.2, 0.5, 2.8, -3.4, 1.1, 0.9, -2.0])
    >>> threshold, count = llh_threshold_by_limit(llhs, lower_limit=0.05)
    >>> threshold  # approximate 5th percentile
    -3.26  # Example output
    >>> count
    1      # E.g., one score is below or at this threshold
    """
    # Convert to float array if needed
    llhs = np.array(llhs, dtype=float)
    
    # Compute the threshold at the desired percentile
    threshold = np.quantile(llhs, lower_limit)
    
    # Count how many scores are below or at this threshold
    count_limit = int(np.sum(llhs <= threshold))
    
    return threshold, count_limit

def percentage_out_of_threshold(llhs: np.ndarray, threshold: float) -> float:
    """
    Compute the percentage of values in `llhs` that are below or equal to a given threshold.

    Parameters
    ----------
    llhs : np.ndarray
        Array of log-likelihood scores (or any numeric values).
    threshold : float
        The cutoff value. All entries less than or equal to this value are considered 
        'out of threshold'.

    Returns
    -------
    float
        The percentage (between 0 and 100) of scores that are <= `threshold`.

    Examples
    --------
    >>> llhs = np.array([1.1, -0.5, 2.7, 0.9, -1.2])
    >>> pct_below_thresh = percentage_out_of_threshold(llhs, 0.0)
    >>> pct_below_thresh
    40.0
    """
    llhs = np.array(llhs, dtype=float)
    out_of_thresh_count = np.sum(llhs <= threshold)
    total_count = llhs.size
    # Calculate percentage
    return (out_of_thresh_count / total_count) * 100

def calculate_shannon_entropy(resps: np.ndarray) -> float:
    """
    Calculate the Shannon entropy for a given responsibility matrix.

    Parameters:
    resps (np.ndarray): Responsibility matrix of shape (n_samples, n_nodes).

    Returns:
    float: The calculated Shannon entropy.
    """
    # Calculate cumulative responsibilities for each GTM node
    CumR_ki = np.sum(resps, axis=0)

    # Normalize to ensure we avoid log(0)
    CumR_ki = CumR_ki / np.sum(CumR_ki)

    # Calculate Shannon entropy
    entropy = -np.sum(CumR_ki * np.log(CumR_ki + np.finfo(float).eps))  # Adding eps for numerical stability
    
    return entropy

def normalize_entropy(entropy: float, K: int) -> float:
    """
    Normalize the Shannon entropy using the total number of nodes.

    Parameters:
    entropy (float): The calculated Shannon entropy.
    K (int): The total number of nodes (columns in the responsibility matrix).

    Returns:
    float: The normalized entropy in the range [0, 1].
    """
    return entropy / np.log(K)


# Helper function to calculate scaffold frequencies and F50
def calculate_scaffold_frequencies_and_f50(scaffolds: List[str], save_distribution: bool=False) -> Tuple[pd.DataFrame, float]:
    """
    Calculate scaffold frequencies and the F50 metric, which is the minimum fraction
    of unique scaffolds needed to represent 50% of the dataset.

    Args:
        scaffolds (List[str]): List of scaffold SMILES strings.
        save_distribution (bool): If the dataframe with a distribution of scaffolds should be saved (default=False)

    Returns:
        Tuple[pd.DataFrame, float]: DataFrame with scaffold frequencies and F50 metric.
    """
    scaffold_counts = pd.Series(scaffolds).value_counts()
    scaffold_df = pd.DataFrame({'scaffold': scaffold_counts.index, 'frequency': scaffold_counts.values})
    scaffold_df = scaffold_df[scaffold_df['scaffold'] != '']
    scaffold_df['cumulative_fraction'] = scaffold_df['frequency'].cumsum() / scaffold_df['frequency'].sum()

    # F50 is the minimum fraction of scaffolds required to cover 50% of molecules
    f50 = scaffold_df[scaffold_df['cumulative_fraction'] >= 0.5].index[0] / len(scaffold_df)
    if save_distribution:
        return scaffold_df, f50
    else:
        return f50

def compute_sinkhorn(dist_mat, reg=0.05, min_iterations=5, max_iterations=10000):
    """
    Compute the Sinkhorn distance for a precomputed Tanimoto *distance* matrix
    using ott-jax. Pass `epsilon` to Geometry, not to Sinkhorn's constructor.
    """

    # 1) Convert to JAX array
    dist_mat_jnp = jnp.asarray(dist_mat, dtype=jnp.float32)

    # 2) Define uniform histograms for an NxN matrix
    n = dist_mat_jnp.shape[0]
    m = dist_mat_jnp.shape[1]
    a = jnp.ones((n,), dtype=jnp.float32) / n
    b = jnp.ones((m,), dtype=jnp.float32) / m

    # 3) Call the Sinkhorn solver using the given function structure
    return linear.solve(
        geometry.Geometry(cost_matrix=dist_mat_jnp, epsilon=reg),
        a=a,
        b=b,
        lse_mode=False,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    ).reg_ot_cost


def cost_matrix_spherical(
    M1: jnp.ndarray,        # shape (G, d) -> means of G components
    sigmas1: jnp.ndarray,   # shape (G,)   -> sqrt-variances for G components
    M2: jnp.ndarray,        # shape (T, d) -> means of T components
    sigmas2: jnp.ndarray,   # shape (T,)   -> sqrt-variances for T components
) -> jnp.ndarray:
    """
    Build a cost matrix of squared W2 distances between:
      N(m_i, sigma_i^2 I), i in [1..G]
      and
      N(m_j, sigma_j^2 I), j in [1..T].
    """

    # --- 1) Pairwise mean distances in O(G * T * d) ---
    # norms1: shape (G,)
    norms1 = jnp.sum(M1 * M1, axis=1)
    # norms2: shape (T,)
    norms2 = jnp.sum(M2 * M2, axis=1)

    # Dot products => shape (G, T)
    dot_products = M1 @ M2.T

    # Expand to pairwise squared dist => (G, T)
    #   ||m_i - m_j||^2 = norms1[i] + norms2[j] - 2 * dot_products[i,j]
    pairwise_mean_dist2 = (
        norms1[:, None] + norms2[None, :] - 2.0 * dot_products
    )

    # --- 2) Add spherical variance differences in O(G * T) ---
    #   W2^2 = ||m_i - m_j||^2 + d * (sigma_i - sigma_j)^2
    d = M1.shape[1]  # dimension
    sig_diff2 = (sigmas1[:, None] - sigmas2[None, :])**2  # shape (G, T)
    pairwise_sig_term = d * sig_diff2

    # Final cost matrix
    cost_mat = pairwise_mean_dist2 + pairwise_sig_term
    return cost_mat

def cost_matrix_gmm_gtm(
    gmm_means: jnp.ndarray,
    gmm_covs: jnp.ndarray,
    gtm_means: jnp.ndarray,
    gtm_covs: jnp.ndarray
) -> jnp.ndarray:
    """
    Build a cost matrix of shape (G, T) where each entry (i, j)
    is the squared Bures distance between the i-th GMM component
    and the j-th GTM component.
    """
    G = gmm_means.shape[0]  # number of components in GMM
    T = gtm_means.shape[0]  # number of components in GTM

    # Initialize cost matrix
    cost_mat = jnp.zeros((G, T))

    # Fill in each entry by computing Bures^2 distance
    for i in tqdm(range(G)):
        for j in range(T):
            cost_mat = cost_mat.at[i, j].set(
                calculate_bures_squared(
                    gmm_means[i], gmm_covs[i],
                    gtm_means[j], gtm_covs[j]
                )
            )

    return cost_mat

def cost_matrix_gmm_gtm_vmap(
    gmm_means: jnp.ndarray,
    gmm_covs: jnp.ndarray,
    gtm_means: jnp.ndarray,
    gtm_covs: jnp.ndarray
) -> jnp.ndarray:
    """
    Build a cost matrix using jax.vmap for vectorized computation.
    Output shape: (G, T)
    """
    # This function computes the cost between a single GMM component and
    # ALL GTM components, returning shape (T,).
    def cost_gmm_vs_all_gtm(mean_x, cov_x, all_means_y, all_covs_y):
        # vmaps over GTM components
        return jax.vmap(
            lambda my, cy: calculate_bures_squared(mean_x, cov_x, my, cy)
        )(all_means_y, all_covs_y)

    # Now, we vmap *again* over the GMM components, returning shape (G, T).
    return jax.vmap(
        lambda mx, cx: cost_gmm_vs_all_gtm(mx, cx, gtm_means, gtm_covs)
    )(gmm_means, gmm_covs)

def calculate_bures_squared(
    mean_x: jnp.ndarray,
    cov_x: jnp.ndarray,
    mean_y: jnp.ndarray,
    cov_y: jnp.ndarray,
    sqrtm_kw: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate the squared Bures (Wasserstein-2) distance between two Gaussians.

    Mathematically, for Gaussians with means m_x, m_y and
    covariance matrices Σ_x, Σ_y, the squared W2 distance is:
    
      W2^2 = ||m_x - m_y||^2
             + trace(Σ_x + Σ_y - 2 * (Σ_x^(1/2) Σ_y Σ_x^(1/2))^(1/2))

    Args:
        mean_x: Mean vector for the first Gaussian, shape (..., d).
        cov_x:  Covariance matrix for the first Gaussian, shape (..., d, d).
        mean_y: Mean vector for the second Gaussian, shape (..., d).
        cov_y:  Covariance matrix for the second Gaussian, shape (..., d, d).
        sqrtm_kw: Optional dict for controlling the matrix square root routine
                  (e.g. iteration count, tolerance, etc.).

    Returns:
        The squared Bures distance as a scalar (or batch of scalars if inputs
        are batched).

    Note:
        If you need the actual Bures (W2) distance (not squared), take the square root
        of this result.
    """
    sqrtm_kw = {} if sqrtm_kw is None else sqrtm_kw

    # Compute cross-term between means:
    mean_dot_prod = jnp.vdot(mean_x, mean_y)
    
    # Compute Σ_x^(1/2):
    sq_x = matrix_square_root.sqrtm(cov_x, cov_x.shape[-1], **sqrtm_kw)[0]
    # Compute (Σ_x^(1/2) Σ_y Σ_x^(1/2))^(1/2):
    sq_x_y_sq_x = sq_x @ cov_y @ sq_x
    sq_sq_x_y_sq_x = matrix_square_root.sqrtm(
        sq_x_y_sq_x, sq_x_y_sq_x.shape[-1], **sqrtm_kw
    )[0]

    # Cross-term: -2 * (mean_x · mean_y + trace((Σ_x^(1/2) Σ_y Σ_x^(1/2))^(1/2)))
    cross_term = -2.0 * (
        mean_dot_prod + jnp.trace(sq_sq_x_y_sq_x, axis1=-2, axis2=-1)
    )

    # Norm-like terms: ||mean||^2 + trace(cov)
    norm_x = jnp.sum(mean_x**2, axis=-1) + jnp.trace(cov_x, axis1=-2, axis2=-1)
    norm_y = jnp.sum(mean_y**2, axis=-1) + jnp.trace(cov_y, axis1=-2, axis2=-1)

    return norm_x + norm_y + cross_term

def calculate_bures(
    x: jnp.ndarray,
    y: jnp.ndarray,
    dimension: int,
    sqrtm_kw: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate the Bures distance between two Gaussian distributions represented
    by (mean, covariance matrix).

    Args:
        x: First input as a concatenated array of mean and covariance matrix (raveled).
        y: Second input as a concatenated array of mean and covariance matrix (raveled).
        dimension: Dimensionality of the data.
        sqrtm_kw: Optional dictionary of keyword arguments for the matrix square root computation.

    Returns:
        The Bures distance as a float.
    """
    def norm(z: jnp.ndarray) -> jnp.ndarray:
        """Compute norm of Gaussian, sq. 2-norm of mean + trace of covariance."""
        mean, cov = x_to_means_and_covs(z, dimension)
        norm_val = jnp.sum(mean ** 2, axis=-1) + jnp.trace(cov, axis1=-2, axis2=-1)
        return norm_val

    sqrtm_kw = {} if sqrtm_kw is None else sqrtm_kw

    mean_x, cov_x = x_to_means_and_covs(x, dimension)
    mean_y, cov_y = x_to_means_and_covs(y, dimension)

    mean_dot_prod = jnp.vdot(mean_x, mean_y)
    sq_x = matrix_square_root.sqrtm(cov_x, dimension, **sqrtm_kw)[0]
    sq_x_y_sq_x = jnp.matmul(sq_x, jnp.matmul(cov_y, sq_x))
    sq_sq_x_y_sq_x = matrix_square_root.sqrtm(sq_x_y_sq_x, dimension, **sqrtm_kw)[0]
    cross_term = -2.0 * (mean_dot_prod + jnp.trace(sq_sq_x_y_sq_x, axis1=-2, axis2=-1))

    return norm(x) + norm(y) + cross_term

def x_to_means_and_covs(x: jnp.ndarray, dimension: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract means and covariance matrices from the concatenated input array.

    Args:
        x: Concatenated input array of means and raveled covariance matrix.
        dimension: Dimensionality of the data.

    Returns:
        mean: Mean vector.
        cov: Covariance matrix.
    """
    mean = x[..., :dimension]
    cov = x[..., dimension:].reshape((-1, dimension, dimension))
    return mean, cov


def gaussian_w2(
    m0: NDArray[np.float64],
    m1: NDArray[np.float64],
    sigma0: NDArray[np.float64],
    sigma1: NDArray[np.float64]
) -> float:
    """
    Compute the squared 2-Wasserstein distance (a.k.a. quadratic Wasserstein distance)
    between two multivariate normal distributions:
      N(m0, sigma0) and N(m1, sigma1).

    Parameters
    ----------
    m0 : NDArray[np.float64]
        Mean of the first Gaussian, shape (d,) or (1, d).
    m1 : NDArray[np.float64]
        Mean of the second Gaussian, shape (d,) or (1, d).
    sigma0 : NDArray[np.float64]
        Covariance of the first Gaussian, shape (d, d).
    sigma1 : NDArray[np.float64]
        Covariance of the second Gaussian, shape (d, d).

    Returns
    -------
    float
        The squared 2-Wasserstein distance.
    """
    # Ensure shapes are correct
    m0 = m0.reshape(-1)
    m1 = m1.reshape(-1)

    sqrt_sigma0 = spl.sqrtm(sigma0)  # sigma0^(1/2)
    inside_root = sqrt_sigma0 @ sigma1 @ sqrt_sigma0
    # (sigma0^(1/2) sigma1 sigma0^(1/2))^(1/2)
    cross_term = spl.sqrtm(inside_root)

    mean_diff = np.linalg.norm(m0 - m1)**2
    trace_term = np.trace(sigma0 + sigma1 - 2 * cross_term)
    return mean_diff + trace_term


def gaussian_map(
    m0: NDArray[np.float64],
    m1: NDArray[np.float64],
    sigma0: NDArray[np.float64],
    sigma1: NDArray[np.float64],
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the optimal transport map between two Gaussians N(m0, sigma0) and
    N(m1, sigma1), evaluated at input points x.

    The map T satisfies T(x) = m1 + (x - m0) * A, where
    A = sigma0^(-1/2) (sigma0^(1/2) sigma1 sigma0^(1/2))^(1/2) sigma0^(-1/2),
    and here we simplify to: A = inv(sigma0) @ sqrtm(sigma0 @ sigma1).

    Parameters
    ----------
    m0 : NDArray[np.float64]
        Mean of the first Gaussian, shape (d,) or (1, d).
    m1 : NDArray[np.float64]
        Mean of the second Gaussian, shape (d,) or (1, d).
    sigma0 : NDArray[np.float64]
        Covariance of the first Gaussian, shape (d, d).
    sigma1 : NDArray[np.float64]
        Covariance of the second Gaussian, shape (d, d).
    x : NDArray[np.float64]
        Points at which the map is evaluated, shape (n, d).

    Returns
    -------
    NDArray[np.float64]
        The mapped points T(x), shape (n, d).
    """
    # Reshape to ensure consistent dimensions
    d = sigma0.shape[0]
    m0 = m0.reshape(1, d)
    m1 = m1.reshape(1, d)

    # Compute map
    sigma_map = np.linalg.inv(sigma0) @ spl.sqrtm(sigma0 @ sigma1)
    return m1 + (x - m0) @ sigma_map


def gw2(
    pi0: NDArray[np.float64],
    pi1: NDArray[np.float64],
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
    s0: NDArray[np.float64],
    s1: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], float]:
    """
    Compute the optimal transport plan and the GW2 distance between two Gaussian
    Mixture Models (GMMs).

    Parameters
    ----------
    pi0 : NDArray[np.float64]
        Mixing weights for GMM0, shape (K0,).
    pi1 : NDArray[np.float64]
        Mixing weights for GMM1, shape (K1,).
    mu0 : NDArray[np.float64]
        Means of GMM0, shape (K0, d).
    mu1 : NDArray[np.float64]
        Means of GMM1, shape (K1, d).
    s0 : NDArray[np.float64]
        Covariances of GMM0, shape (K0, d, d).
    s1 : NDArray[np.float64]
        Covariances of GMM1, shape (K1, d, d).

    Returns
    -------
    wstar : NDArray[np.float64]
        The optimal transport plan between the two GMMs, shape (K0, K1).
    dist_gw2 : float
        The GW2 distance between the two GMMs (sum of wstar * cost_matrix).
    """
    k0 = mu0.shape[0]
    k1 = mu1.shape[0]
    d = mu0.shape[1]
    # Ensure the covariance matrices are the right shape
    #s0 = s0.reshape(k0, -1, -1)
    #s1 = s1.reshape(k1, -1, -1)
    # Covariance matrices
    s0 = s0.reshape(k0, d, d)  # Ensure shape (k0, d, d)
    s1 = s1.reshape(k1, d, d)  # Ensure shape (k1, d, d)


    # Compute pairwise cost matrix
    cost_matrix = np.zeros((k0, k1), dtype=np.float64)
    for i in tqdm(range(k0)):
        for j in range(k1):
            cost_matrix[i, j] = gaussian_w2(mu0[i], mu1[j], s0[i], s1[j])
    print('cost matrix was calculated')
    # Solve the discrete OT problem with the cost matrix
    wstar = ot.emd(pi0, pi1, cost_matrix)
    dist_gw2 = np.sum(wstar * cost_matrix)
    return wstar, dist_gw2


def gw2_cost(
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
    s0: NDArray[np.float64],
    s1: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Return the pairwise GW2 cost matrix of size (K0, K1) between two sets of Gaussians
    specified by (mu0, s0) and (mu1, s1).

    Parameters
    ----------
    mu0 : NDArray[np.float64]
        Means of the first set of Gaussians, shape (K0, d).
    mu1 : NDArray[np.float64]
        Means of the second set of Gaussians, shape (K1, d).
    s0 : NDArray[np.float64]
        Covariances of the first set of Gaussians, shape (K0, d, d).
    s1 : NDArray[np.float64]
        Covariances of the second set of Gaussians, shape (K1, d, d).

    Returns
    -------
    NDArray[np.float64]
        The cost matrix of size (K0, K1), where entry (k, l) is the squared
        2-Wasserstein distance between Gaussian k of the first set and
        Gaussian l of the second set.
    """
    k0 = mu0.shape[0]
    k1 = mu1.shape[0]

    s0 = s0.reshape(k0, -1, -1)
    s1 = s1.reshape(k1, -1, -1)

    cost_matrix = np.zeros((k0, k1), dtype=np.float64)
    for i in range(k0):
        for j in range(k1):
            cost_matrix[i, j] = gaussian_w2(mu0[i], mu1[j], s0[i], s1[j])
    return cost_matrix



@njit
def mahalanobis_squared(mu1: np.ndarray, mu2: np.ndarray, inv_sigma: np.ndarray) -> float:
    """
    Compute the Mahalanobis distance squared between two mean vectors.
    
    Args:
        mu1 (np.ndarray): Mean vector of the first Gaussian component.
        mu2 (np.ndarray): Mean vector of the second Gaussian component.
        inv_sigma (np.ndarray): Inverse of the covariance matrix.
        
    Returns:
        float: The Mahalanobis distance squared.
    """
    diff = mu1 - mu2
    return np.dot(diff.T, np.dot(inv_sigma, diff))

@njit(parallel=True)
def tanimoto_distance_counts_numba_parallel(matrix_1, matrix_2):
    """
    Compute the Tanimoto distance between two matrices.

    The Tanimoto distance is calculated based on formula:
        1 - (\sum min(A,B)) / (\sum A + \sum B - \sum min(A,B))

    Parameters
    ----------
    matrix_1 : numpy.ndarray
        The first matrix of shape (N1, D), where N1 is the number of rows and D is the number of columns.
    matrix_2 : numpy.ndarray
        The second matrix of shape (N2, D), where N2 is the number of rows and D is the number of columns.

    Returns
    -------
    result : numpy.ndarray
        The matrix of Tanimoto distances of shape (N1, N2), where result[i, j] is the distance between matrix_1[i, :] and matrix_2[j, :].

    Raises
    ------
    ValueError
        If the two matrices have different numbers of columns.
    """
    # Extracting sizes of matrices
    N1, D1 = matrix_1.shape
    N2, D2 = matrix_2.shape

    if D1 != D2:
        raise ValueError("The two matrices should have the same number of columns.")

    # Initialize result matrix
    result = np.zeros((N1, N2))

    # Compute Tanimoto distances using parallel loops
    for i in prange(N1):
        for j in prange(N2):
            a_sum = 0.0
            b_sum = 0.0
            c_sum = 0.0
            for k in range(D1):
                a_sum += matrix_1[i, k]
                b_sum += matrix_2[j, k]
                c_sum += min(matrix_1[i, k], matrix_2[j, k])

            result[i, j] = 1 - (c_sum / (a_sum + b_sum - c_sum))

    return result
@njit
def bhattacharyya_distance_matrix_classical(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute the Bhattacharyya distance between rows of two matrices.

    Parameters:
        p (np.ndarray): First matrix of probability distributions (m x n).
        q (np.ndarray): Second matrix of probability distributions (m x n).

    Returns:
        np.ndarray: Distance matrix (m x m), where each entry (i, j) represents the 
                    Bhattacharyya distance between p[i] and q[j].
    """
    # Check if dimensions match
    if p.shape[1] != q.shape[1]:
        raise ValueError("Both matrices must have the same number of columns")

    # Number of rows
    m, n = p.shape
    k, _ = q.shape

    # Initialize the distance matrix
    distance_matrix = np.zeros((m, k))

    # Iterate over all row pairs
    for i in range(m):
        for j in range(k):
            bc = 0.0
            for l in range(n):
                bc += np.sqrt(p[i, l] * q[j, l])

            # Calculate Bhattacharyya distance
            if bc == 0.0:
                distance_matrix[i, j] = np.inf
            else:
                distance_matrix[i, j] = -np.log(bc)

    return distance_matrix

@njit
def bhattacharyya_distance_classical(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the Bhattacharyya distance between two probability distributions.

    Parameters:
        p (np.ndarray): First probability distribution.
        q (np.ndarray): Second probability distribution.

    Returns:
        float: Bhattacharyya distance between p and q.
    """
    # Ensure inputs are 1D arrays
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("Both inputs must be 1D arrays")

    # Normalize the distributions
    #_sum = np.sum(p)
    #q_sum = np.sum(q)
    #if p_sum == 0 or q_sum == 0:
    #    raise ValueError("Distributions must not be zero vectors")

    #p = p / p_sum
    #q = q / q_sum

    # Compute the Bhattacharyya coefficient
    bc = 0.0
    for i in range(p.shape[0]):
        bc += np.sqrt(p[i] * q[i])

    # Calculate Bhattacharyya distance
    if bc == 0.0:
        return np.inf  # To avoid log(0) issues

    distance = -np.log(bc)
    return distance




@njit(parallel=True)
def c2_distance_matrix(mixing_weights1, means1, mixing_weights2, means2, inv_sigma):
    """
    Compute the C2 distance matrix between two sets of Gaussian mixture models.

    Args:
        mixing_weights1 (np.ndarray): Mixing weights for the first set of GMMs (N1 x K).
        means1 (np.ndarray): Means of Gaussian components in the first set of GMMs (N1 x K x D).
        mixing_weights2 (np.ndarray): Mixing weights for the second set of GMMs (N2 x K).
        means2 (np.ndarray): Means of Gaussian components in the second set of GMMs (N2 x K x D).
        inv_sigma (np.ndarray): Inverse of the shared covariance matrix (D x D).

    Returns:
        np.ndarray: The matrix of C2 distances of shape (N1, N2), where result[i, j] is the C2 distance
                    between the i-th GMM from the first set and the j-th GMM from the second set.
    """
    N1, K1, D1 = means1.shape
    N2, K2, D2 = means2.shape

    if K1 != K2 or D1 != D2:
        raise ValueError("The GMMs must have the same number of components and dimensions.")

    result = np.zeros((N1, N2))

    for i in prange(N1):
        for j in prange(N2):
            num = 0.0  # Numerator: Cross terms between GMM1 and GMM2
            den1 = 0.0  # Denominator: Self-overlap of GMM1
            den2 = 0.0  # Denominator: Self-overlap of GMM2

            # Compute the numerator: overlap between GMM1[i] and GMM2[j]
            for k1 in range(K1):
                for k2 in range(K2):
                    dm_sq = mahalanobis_squared(means1[i, k1], means2[j, k2], inv_sigma)
                    weight = mixing_weights1[i, k1] * mixing_weights2[j, k2] * np.exp(-dm_sq / 2)
                    num += weight

            # Compute the first denominator: self-overlap of GMM1[i]
            for k1 in range(K1):
                for k2 in range(K1):
                    dm_sq = mahalanobis_squared(means1[i, k1], means1[i, k2], inv_sigma)
                    weight = mixing_weights1[i, k1] * mixing_weights1[i, k2] * np.exp(-dm_sq / 2)
                    den1 += weight

            # Compute the second denominator: self-overlap of GMM2[j]
            for k1 in range(K2):
                for k2 in range(K2):
                    dm_sq = mahalanobis_squared(means2[j, k1], means2[j, k2], inv_sigma)
                    weight = mixing_weights2[j, k1] * mixing_weights2[j, k2] * np.exp(-dm_sq / 2)
                    den2 += weight

            # Combine the results for the C2 distance
            denom = den1 + den2
            if denom > 0:
                result[i, j] = -np.log(2 * num / denom)
            else:
                result[i, j] = np.inf

    return result

@njit
def c2_distance(
    mixing_weights1: List[float],
    means1: List[np.ndarray],
    mixing_weights2: List[float],
    means2: List[np.ndarray],
    inv_sigma: np.ndarray
) -> float:
    """
    Compute the C2 distance between two Gaussian mixture models.

    Args:
        mixing_weights1 (List[float]): Mixing weights for the first GMM.
        means1 (List[np.ndarray]): Means of Gaussian components in the first GMM.
        mixing_weights2 (List[float]): Mixing weights for the second GMM.
        means2 (List[np.ndarray]): Means of Gaussian components in the second GMM.
        inv_sigma (np.ndarray): Inverse of the shared covariance matrix.

    Returns:
        float: The C2 distance.
    """
    num = 0.0  # Numerator: Cross terms between GMM1 and GMM2
    den1 = 0.0  # Denominator: Self-overlap of GMM1
    den2 = 0.0  # Denominator: Self-overlap of GMM2
    
    # Compute the numerator: overlap between GMM1 and GMM2
    for i in range(len(mixing_weights1)):
        for j in range(len(mixing_weights2)):
            dm_sq = mahalanobis_squared(means1[i], means2[j], inv_sigma)
            weight = mixing_weights1[i] * mixing_weights2[j] * np.exp(-dm_sq / 2)
            num += weight
    
    # Compute the first denominator: self-overlap of GMM1
    for i in range(len(mixing_weights1)):
        for j in range(len(mixing_weights1)):
            dm_sq = mahalanobis_squared(means1[i], means1[j], inv_sigma)
            weight = mixing_weights1[i] * mixing_weights1[j] * np.exp(-dm_sq / 2)
            den1 += weight
    
    # Compute the second denominator: self-overlap of GMM2
    for i in range(len(mixing_weights2)):
        for j in range(len(mixing_weights2)):
            dm_sq = mahalanobis_squared(means2[i], means2[j], inv_sigma)
            weight = mixing_weights2[i] * mixing_weights2[j] * np.exp(-dm_sq / 2)
            den2 += weight
    
    # Combine the results for the C2 distance
    denom = den1 + den2
    c2 = -np.log(2 * num / denom)
    return c2




@njit
def bhattacharyya_component(mu1: np.ndarray, mu2: np.ndarray, inv_sigma: np.ndarray) -> float:
    """
    Compute the Bhattacharyya distance between two Gaussian components.
    """
    # Bhattacharyya distance for isotropic covariance
    return (1 / 8) * mahalanobis(mu1, mu2, inv_sigma)#mahalanobis_squared(mu1, mu2, inv_sigma)


@njit
def bhattacharyya_gmm(
    mixing_weights1: List[float],
    means1: List[np.ndarray],
    mixing_weights2: List[float],
    means2: List[np.ndarray],
    inv_sigma: np.ndarray
) -> float:
    """
    Compute the Bhattacharyya distance for two Gaussian Mixture Models (GMMs).

    Args:
        mixing_weights1 (List[float]): Mixture weights for GMM1.
        means1 (List[np.ndarray]): Means of Gaussian components in GMM1.
        mixing_weights2 (List[float]): Mixture weights for GMM2.
        means2 (List[np.ndarray]): Means of Gaussian components in GMM2.
        inv_sigma (np.ndarray): Inverse of the covariance matrix.

    Returns:
        float: The Bhattacharyya distance between the two GMMs.
    """
    bh_distance = 0.0
    for i in range(len(mixing_weights1)):
        for j in range(len(mixing_weights2)):
            b_comp = bhattacharyya_component(means1[i], means2[j], inv_sigma)
            bh_distance_ij = (mixing_weights1[i] * mixing_weights2[j] * np.exp(-b_comp))
            #bh_distance += (mixing_weights1[i] * mixing_weights2[j] * np.exp(-b_comp))
            bh_distance = bh_distance + bh_distance_ij
            #print(mixing_weights1[i], mixing_weights2[j])
            print(b_comp, mixing_weights1[i] * mixing_weights2[j], bh_distance_ij, bh_distance)
    return -np.log(bh_distance)
    


@njit
def bhattacharyya_distance(mu1, Sigma1, mu2, Sigma2):
    """
    Calculate the Bhattacharyya distance between two multivariate Gaussian distributions.
    
    Parameters:
    - mu1: Mean vector of the first Gaussian (numpy array of shape (n_features,))
    - Sigma1: Covariance matrix of the first Gaussian (numpy array of shape (n_features, n_features))
    - mu2: Mean vector of the second Gaussian (numpy array of shape (n_features,))
    - Sigma2: Covariance matrix of the second Gaussian (numpy array of shape (n_features, n_features))
    
    Returns:
    - db: Bhattacharyya distance between the two distributions
    """
    # Calculate the average covariance matrix
    Sigma = 0.5 * (Sigma1 + Sigma2)
    
    # Calculate the difference between the means
    delta_mu = mu1 - mu2
    
    # Add a small value to the diagonal of Sigma to ensure it is invertible (for numerical stability)
    epsilon = 1e-10
    Sigma += epsilon * np.eye(Sigma.shape[0])
    
    # Compute the inverse and determinant of the average covariance matrix
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    
    # Compute the determinants of the individual covariance matrices
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)
    
    # Compute the Mahalanobis-like distance (first term in the Bhattacharyya formula)
    mahalanobis_term = 0.125 * np.dot(delta_mu.T, np.dot(Sigma_inv, delta_mu))
    
    # Compute the determinant term (second term in the Bhattacharyya formula)
    det_term = 0.5 * np.log(det_Sigma / np.sqrt(det_Sigma1 * det_Sigma2))
    
    # Bhattacharyya distance
    db = mahalanobis_term + det_term
    return db

@njit
def bhattacharyya_distance_matrix(means1, covs1, means2, covs2):
    """
    Compute the Bhattacharyya distance matrix between two arrays of Gaussian distributions.
    
    Parameters:
    - means1: Array of mean vectors for the first set of distributions (shape: (N, d))
    - covs1: Array of covariance matrices for the first set of distributions (shape: (N, d, d))
    - means2: Array of mean vectors for the second set of distributions (shape: (M, d))
    - covs2: Array of covariance matrices for the second set of distributions (shape: (M, d, d))
    
    Returns:
    - D: Distance matrix of shape (N, M), where D[i, j] is the Bhattacharyya distance between
         the i-th distribution in the first set and the j-th distribution in the second set.
    """
    N = means1.shape[0]
    M = means2.shape[0]
    D = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            D[i, j] = bhattacharyya_distance(means1[i], covs1[i], means2[j], covs2[j])
    
    return D





def s_weighted_knn_dist(dist_matrix: np.ndarray, K: int) -> np.ndarray:
    """
    Computes the weighted sum of distances over the Top-K nearest samples
    for each sample, using numerically stable softmax weights over negative distances.

    Parameters:
    - dist_matrix (np.ndarray): Distance matrix of shape (N, M).
    - K (int): Number of nearest samples to consider.

    Returns:
    - np.ndarray: Array of weighted distances for each sample, shape (N,).
    """
    # Handle NaNs by replacing them with large values (to push them to the end of sorting)
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    
    K = min(K, dist_matrix.shape[1])

    # Get indices of the Top-K nearest samples
    sorted_indices = np.argsort(dist_matrix, axis=1)
    top_k_indices = sorted_indices[:, :K]

    # Extract the Top-K distances
    row_indices = np.arange(dist_matrix.shape[0])[:, None]
    top_k_distances = dist_matrix[row_indices, top_k_indices]  # Shape: (N, K)

    # Ensure distances are non-negative
    if np.any(top_k_distances < 0):
        raise ValueError("Distances must be non-negative.")

    # Compute numerically stable softmax weights over negative distances
    # Shift the negative distances for numerical stability
    # Since we take exponentials of negative distances, we shift them to prevent underflow
    max_neg_distances = -np.min(top_k_distances, axis=1, keepdims=True)  # Shape: (N, 1)
    shifted_neg_distances = -top_k_distances - max_neg_distances  # Shape: (N, K)

    # Exponentiate the shifted negative distances
    exp_neg_distances = np.exp(shifted_neg_distances)  # Shape: (N, K)

    # Compute the sum of exponentials
    sum_exp = np.sum(exp_neg_distances, axis=1, keepdims=True)  # Shape: (N, 1)

    # Compute the softmax weights
    weights = exp_neg_distances / sum_exp  # Shape: (N, K)

    # Compute weighted sum of distances
    weighted_distances = np.sum(weights * top_k_distances, axis=1)  # Shape: (N,)

    return weighted_distances

def s_batch_knn_dist(dist_matrix: np.ndarray, K: int) -> np.ndarray:
    """
    Computes the minimum distance among the Top-K nearest samples for each sample.

    Parameters:
    - dist_matrix (np.ndarray): Distance matrix of shape (N, M).
    - K (int): Number of nearest samples to consider.

    Returns:
    - np.ndarray: Array of minimum distances among Top-K for each sample, shape (N,).
    """
    # Handle NaNs by replacing them with large values (to push them to the end of sorting)
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    
    K = min(K, dist_matrix.shape[1])
    sorted_indices = np.argsort(dist_matrix, axis=1)
    top_k_indices = sorted_indices[:, :K]
    row_indices = np.arange(dist_matrix.shape[0])[:, None]
    top_k_distances = dist_matrix[row_indices, top_k_indices]
    mean_top_k_distances = np.mean(top_k_distances, axis=1)
    return mean_top_k_distances


def s_min(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the minimum distance for each sample.

    Parameters:
    - dist_matrix (np.ndarray): Distance matrix of shape (N, M).

    Returns:
    - np.ndarray: Array of minimum distances for each sample, shape (N,).
    """
    # Handle NaNs by replacing them with large values (to push them to the end of sorting)
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    min_distances = np.min(dist_matrix, axis=1)
    return min_distances


def chamfer_distance_from_dist_matrix(dist_matrix: np.ndarray) -> float:
    """
    Computes the Chamfer Distance between two point clouds using the precomputed distance matrix.

    Parameters:
    - dist_matrix (np.ndarray): A 2D array of shape (N_P, N_Q), where N_P is the number of points in P,
      and N_Q is the number of points in Q. Each element dist_matrix[i, j] represents the distance
      between point p_i in P and point q_j in Q.

    Returns:
    - float: The Chamfer Distance between point clouds P and Q.
    """
    if dist_matrix.ndim != 2:
        raise ValueError("dist_matrix must be a 2D array.")

    # From P to Q: For each point in P, find the minimum distance to Q
    min_dist_P_to_Q = np.min(dist_matrix, axis=1)  # Shape: (N_P,)
    sum_min_dist_P_to_Q = np.mean(min_dist_P_to_Q ** 2)

    # From Q to P: For each point in Q, find the minimum distance to P
    min_dist_Q_to_P = np.min(dist_matrix, axis=0)  # Shape: (N_Q,)
    sum_min_dist_Q_to_P = np.mean(min_dist_Q_to_P ** 2)

    # Chamfer Distance is the sum of both sums
    chamfer_dist = sum_min_dist_P_to_Q + sum_min_dist_Q_to_P

    return chamfer_dist


def chamfer_distance_ckdtree(
    P: Union[np.ndarray, list],
    Q: Union[np.ndarray, list]
) -> float:
    """
    Computes the Chamfer Distance between two point clouds P and Q using cKDTree for efficient nearest neighbor search.

    Parameters:
    - P (numpy.ndarray or list): An array-like object of shape (N, D), where N is the number of points
      and D is the dimensionality of each point.
    - Q (numpy.ndarray or list): An array-like object of shape (M, D), where M is the number of points
      and D is the dimensionality of each point.

    Returns:
    - cd (float): The Chamfer Distance between point clouds P and Q.
    """
    # Ensure inputs are numpy arrays
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Build KD-Trees for efficient nearest neighbor search
    tree_P = cKDTree(P)
    tree_Q = cKDTree(Q)

    # For each point in Q, find the nearest point in P
    dists_Q, _ = tree_P.query(Q)
    cd_Q = np.sum(dists_Q ** 2)

    # For each point in P, find the nearest point in Q
    dists_P, _ = tree_Q.query(P)
    cd_P = np.sum(dists_P ** 2)

    # The Chamfer Distance is the sum of both distances
    cd = cd_P + cd_Q
    return cd

def load_and_preprocess_with_centroids(config: Dict[str, Any]) -> Tuple[
    pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray,
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, np.ndarray], pd.DataFrame
]:
    """Load MNIST dataset, preprocess it, and include PCA, UMAP, and centroids for all variants."""
    output_dir = config['general']['output_dir']
    # 1. Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    mnist.target = mnist.target.astype(int)
    data = pd.DataFrame(mnist.data)
    data['label'] = mnist.target

    # 2. Uniform sampling for balanced data
    sampled_data = data.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=1000, random_state=42)
    ).reset_index(drop=True)

    X = sampled_data.drop(columns='label').values
    y = sampled_data['label'].values

    # 3. Preprocessing: Standardize, Min-Max scale, Center
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)

    scaler_mm = MinMaxScaler()
    X_minmax = scaler_mm.fit_transform(X)
    X_minmax_centered = X_minmax - X_minmax.mean(axis=0)

    X_centered = X - X.mean(axis=0)

    # 4. PCA: Determine components based on explained variance
    explained_variance_ratio = config['pca'].get('explained_variance_ratio', 0.95)

    def fit_pca(data: np.ndarray):
        pca = PCA(n_components=None)
        pca.fit(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= explained_variance_ratio) + 1
        pca_transform = PCA(n_components=n_components).fit_transform(data)
        pca_2d_transform = PCA(n_components=2).fit_transform(data)
        return pca_transform, pca_2d_transform, cumulative_variance, n_components



    # PCA for each variant
    pca_std, pca_std_2d, cumulative_variance_2d, n_components_2d = fit_pca(X_std)
    pca_minmax, pca_minmax_2d, cumulative_variance_minmax, n_components_minmax = fit_pca(X_minmax)
    # Plotting explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_minmax) + 1), cumulative_variance_minmax, marker='o', linestyle='--')
    plt.axhline(y=explained_variance_ratio, color='r', linestyle='--',
                label=f'{explained_variance_ratio * 100}% Threshold')
    plt.axvline(x=n_components_minmax, color='g', linestyle='--', label=f'{n_components_minmax} Components')
    plt.title("Explained Variance as a Function of Number of Components MinMax")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/minmax_pca.png')
    plt.close()
    pca_centered, pca_centered_2d, cumulative_variance_center, n_components_center = fit_pca(X_centered)
    # Plotting explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_center) + 1), cumulative_variance_center, marker='o', linestyle='--')
    plt.axhline(y=explained_variance_ratio, color='r', linestyle='--',
                label=f'{explained_variance_ratio * 100}% Threshold')
    plt.axvline(x=n_components_center, color='g', linestyle='--', label=f'{n_components_center} Components')
    plt.title("Explained Variance as a Function of Number of Components Centered Data")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'{output_dir}/center_pca.png')
    plt.close()

    # 5. UMAP transformations for each variant
    umap_components = config['umap'].get('n_components', 2)
    umap_neighbors = config['umap'].get('n_neighbors', 15)
    umap_model = UMAP(n_components=umap_components, n_neighbors=umap_neighbors, random_state=config['general']['seed'])

    umap_std = umap_model.fit_transform(X_std)
    umap_minmax = umap_model.fit_transform(X_minmax)
    umap_orig = umap_model.fit_transform(X)

    # 6. UMAP on PCA-transformed data
    umap_pca_std = umap_model.fit_transform(pca_std)
    umap_pca_minmax = umap_model.fit_transform(pca_minmax)
    umap_pca_centered = umap_model.fit_transform(pca_centered)


    return X, y, X_std, X_minmax, X_centered, \
           [pca_std, pca_minmax, pca_centered], \
           [pca_std_2d, pca_minmax_2d, pca_centered_2d], \
           [umap_std, umap_minmax, umap_orig], \
           [umap_pca_std, umap_pca_minmax, umap_pca_centered], \
            sampled_data

def get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return the distance function based on the metric name."""
    if metric == "euclidean":
        return euclidean_distance_square_numba
    elif metric == "cosine":
        return cosine_distance
    elif metric == "tanimoto":
        return tanimoto_distance
    else:
        raise ValueError(f"Unsupported metric: {metric}")




def tanimoto_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute the Tanimoto distance matrix between two datasets."""
    similarity = tanimoto_int_similarity_matrix_numba(X, Y)
    distance = 1 - similarity
    return distance


def compute_pairwise_distances(
    datasets: List[pd.DataFrame],
    metric: str
) -> np.ndarray:
    """Compute pairwise distances between datasets using the specified metric."""
    num_datasets = len(datasets)
    pairwise_distances = np.zeros((num_datasets, num_datasets))

    distance_func = get_distance_function(metric)

    for i in tqdm(range(num_datasets), desc=f"Computing pairwise distances ({metric})"):
        data_i = datasets[i].drop(columns='label').to_numpy()
        for j in range(i + 1, num_datasets):
            data_j = datasets[j].drop(columns='label').to_numpy()
            dist_matrix = distance_func(data_i, data_j)
            dist = compute_mean_upper_triangle(dist_matrix)
            pairwise_distances[i, j] = dist
            pairwise_distances[j, i] = dist  # Symmetric assignment
    return pairwise_distances


def compute_mean_upper_triangle(matrix: np.ndarray) -> float:
    """Compute the mean of the upper triangular elements (excluding diagonal) of a matrix."""
    upper_triangle_indices = np.triu_indices_from(matrix, k=1)
    upper_triangle_values = matrix[upper_triangle_indices]
    mean_upper_triangle = upper_triangle_values.mean()
    return mean_upper_triangle


def scale_matrix_excluding_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Scale the off-diagonal elements of a matrix between 0 and 1."""
    scaled_matrix = matrix.copy()
    diagonal = np.diag(matrix)
    mask = np.ones_like(matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    min_val = scaled_matrix[mask].min()
    max_val = scaled_matrix[mask].max()
    scaled_matrix[mask] = (scaled_matrix[mask] - min_val) / (max_val - min_val)
    np.fill_diagonal(scaled_matrix, diagonal)
    return scaled_matrix


@jit(nopython=True, parallel=True)
def cosine_similarity_numba(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine similarity between each pair of vectors
    in two arrays using Numba for optimization.

    Args:
        x1 (np.ndarray): First array of shape (n_samples_1, n_features).
        x2 (np.ndarray): Second array of shape (n_samples_2, n_features).

    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_samples_1, n_samples_2).
    """
    n_samples_1, n_features = x1.shape
    n_samples_2, _ = x2.shape
    result = np.empty((n_samples_1, n_samples_2), dtype=np.float64)

    # Precompute norms of x1 and x2
    norms_x1 = np.zeros(n_samples_1, dtype=np.float64)
    norms_x2 = np.zeros(n_samples_2, dtype=np.float64)

    for i in prange(n_samples_1):
        sum_sq = 0.0
        for k in range(n_features):
            sum_sq += x1[i, k] * x1[i, k]
        norms_x1[i] = np.sqrt(sum_sq)

    for j in prange(n_samples_2):
        sum_sq = 0.0
        for k in range(n_features):
            sum_sq += x2[j, k] * x2[j, k]
        norms_x2[j] = np.sqrt(sum_sq)

    # Compute cosine similarity
    for i in prange(n_samples_1):
        for j in prange(n_samples_2):
            dot = 0.0
            for k in range(n_features):
                dot += x1[i, k] * x2[j, k]
            denom = norms_x1[i] * norms_x2[j]
            if denom > 0:
                result[i, j] = dot / denom
            else:
                result[i, j] = 0.0  # Handle zero division
    return result


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute the cosine distance matrix between two datasets."""
    similarity = cosine_similarity_numba(X, Y)
    return similarity

@jit(nopython=True, parallel=True)
def euclidean_distance_square_numba(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate the squared Euclidean distance between each pair of vectors
    in two arrays using Numba for optimization.
    """
    n_samples_1, n_features = x1.shape
    n_samples_2 = x2.shape[0]
    result = np.empty((n_samples_1, n_samples_2), dtype=np.float64)

    for i in prange(n_samples_1):
        for j in prange(n_samples_2):
            dist_sq = 0.0
            for k in prange(n_features):
                diff = x1[i, k] - x2[j, k]
                dist_sq += diff * diff
            result[i, j] = dist_sq

    return result

@njit(parallel=True, fastmath=True)
def tanimoto_int_similarity_matrix_numba(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """
    Implement the Tanimoto similarity measure for integer matrices, comparing each vector in v_a against each in v_b.

    Parameters:
    - v_a (np.ndarray): Numpy matrix where each row represents a vector a.
    - v_b (np.ndarray): Numpy matrix where each row represents a vector b.

    Returns:
    - np.ndarray: Matrix of computed similarity scores, where element (i, j) is the similarity between row i of v_a and row j of v_b.
    """

    num_rows_a = v_a.shape[0]
    num_rows_b = v_b.shape[0]
    similarity_matrix = np.empty((num_rows_a, num_rows_b), dtype=np.float32)

    sum_a_squared = np.sum(np.square(v_a), axis=1)
    sum_b_squared = np.sum(np.square(v_b), axis=1)

    for i in prange(num_rows_a):
        for j in prange(num_rows_b):
            numerator = np.dot(v_a[i], v_b[j])
            denominator = sum_a_squared[i] + sum_b_squared[j] - numerator

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = numerator / denominator

            #similarity_matrix[i, j] = similarity
            similarity_matrix[i, j] = max(1 - similarity, 0)

    return similarity_matrix



def tanimoto_similarity_continuous(matrix_1, matrix_2):
    """
    "The Tanimoto coefficient is a measure of the similarity between two sets.
    It is defined as the size of the intersection divided by the size of the union of the sample sets."

    The Tanimoto coefficient is also known as the Jaccard index

    Adoppted from https://github.com/cimm-kzn/CIMtools/blob/master/CIMtools/metrics/pairwise.py

    :param matrix_1: 2D array of features.
    :param matrix_2: 2D array of features.
    :return: The Tanimoto coefficient between the two arrays.
    """
    x_dot = np.dot(matrix_1, matrix_2.T)

    x2 = (matrix_1**2).sum(axis=1)
    y2 = (matrix_2**2).sum(axis=1)

    len_x2 = len(x2)
    len_y2 = len(y2)

    result = x_dot / (np.array([x2] * len_y2).T + np.array([y2] * len_x2) - x_dot)
    result[np.isnan(result)] = 0

    return result


@njit
def tanimoto_distance_counts_numba(matrix_1, matrix_2):
    """
    Compute the Tanimoto distance between two matrices.

    The Tanimoto distance is calculated based on formula:
        1 - (\sum min(A,B)) / (\sum A + \sum B - \sum min(A,B))

    Parameters
    ----------
    matrix_1 : numpy.ndarray
        The first matrix of shape (N1, D), where N1 is the number of rows and D is the number of columns.
    matrix_2 : numpy.ndarray
        The second matrix of shape (N2, D), where N2 is the number of rows and D is the number of columns.

    Returns
    -------
    result : numpy.ndarray
        The matrix of Tanimoto distances of shape (N1, N2), where result[i, j] is the distance between matrix_1[i, :] and matrix_2[j, :].

    Raises
    ------
    ValueError
        If the two matrices have different numbers of columns.
    """
    # Extracting sizes of matrices
    N1, D1 = matrix_1.shape
    N2, D2 = matrix_2.shape

    if D1 != D2:
        raise ValueError("The two matrices should have the same number of columns.")

    # Initialize result matrix
    result = np.zeros((N1, N2))

    # Compute Tanimoto distances
    for i in range(N1):
        for j in range(N2):
            a_sum = np.sum(matrix_1[i])
            b_sum = np.sum(matrix_2[j])
            c_sum = np.sum(np.minimum(matrix_1[i], matrix_2[j]))

            result[i, j] = 1 - (c_sum / (a_sum + b_sum - c_sum))

    return result


def tanimoto_distance_counts(matrix_1, matrix_2):
    """Compute the Tanimoto distance between two matrices.

    The Tanimoto distance is caclulated based on formula 1 - \frac{\sum \min(A,B)}{\sum A + \sum B - \sum \min(A,B)}

    Parameters
    ----------
    matrix_1 : numpy.ndarray
        The first matrix of shape (N1, D), where N1 is the number of rows and D is the number of columns.
    matrix_2 : numpy.ndarray
        The second matrix of shape (N2, D), where N2 is the number of rows and D is the number of columns.

    Returns
    -------
    result : numpy.ndarray
        The matrix of Tanimoto distances of shape (N1, N2), where result[i, j] is the distance between matrix_1[i, :] and matrix_2[j, :].

    Raises
    ------
    ValueError
        If the two matrices have different numbers of columns.
    """

    # Extracting sizes of matrices
    N1, D1 = matrix_1.shape
    N2, D2 = matrix_2.shape

    if D1 != D2:
        raise ValueError("The two matrices should have the same number of columns.")

    # Sums for all vectors in both matrices
    a = matrix_1.sum(axis=-1)
    b = matrix_2.sum(axis=-1)

    # Expand matrices to allow element-wise operations between all pairs of vectors
    expanded_matrix_1 = np.repeat(matrix_1[:, np.newaxis, :], N2, axis=1)
    expanded_matrix_2 = np.repeat(matrix_2[np.newaxis, :, :], N1, axis=0)

    # Calculate element-wise minimums between all pairs of vectors
    c = np.minimum(expanded_matrix_1, expanded_matrix_2).sum(axis=-1)

    # caclulate Tanimoto distance based on formula 1 - \frac{\sum \min(A,B)}{\sum A + \sum B - \sum \min(A,B)}
    result = 1 - (c / (a[:, np.newaxis] + b[np.newaxis, :] - c))

    return result

def optimal_transport_geomloss(batch_x: torch.Tensor, batch_pos: torch.Tensor, p: int = 2, blur: float = 0.05) -> torch.Tensor:
    """
    Optimal Transport using GeomLoss for batch-to-batch processing.

    Args:
        batch_x (torch.Tensor): Batch of evaluation molecules (shape: [batch_size, feature_dim]).
        batch_pos (torch.Tensor): Batch of positive molecule sets (shape: [batch_size, n_pos, feature_dim]).
        p (int): Ground metric parameter for OT (e.g., p=2 for squared Euclidean distance).
        blur (float): Entropic regularization parameter for Sinkhorn.

    Returns:
        torch.Tensor: Sinkhorn distances for each pair of batches (shape: [batch_size]).
    """
    # Define the Sinkhorn loss with the given parameters
    sinkhorn_loss = SamplesLoss("sinkhorn", p=p, blur=blur)

    # Compute Sinkhorn distance for the batch
    # batch_x is treated as individual measures in a batch
    ot_distances = sinkhorn_loss(batch_x, batch_pos)

    return ot_distances


from scipy.stats import spearmanr, pearsonr

def compute_correlations_with_pca_umap_2d(
    distances_dict: Dict[str, np.ndarray],
    pca_2d_variants: List[np.ndarray],
    umap_variants: List[np.ndarray]
) -> pd.DataFrame:
    """Compute correlations for distances between first two PCA and UMAP components."""
    correlation_results = []

    pca_2d_names = ["PCA_2D_Standardized", "PCA_2D_MinMax", "PCA_2D_Centered"]
    umap_names = ["UMAP_Standardized", "UMAP_MinMax", "UMAP_Centered"]

    for pca_2d_data, pca_name in zip(pca_2d_variants, pca_2d_names):
        for umap_data, umap_name in zip(umap_variants, umap_names):
            spearman_corr, _ = spearmanr(pca_2d_data.ravel(), umap_data.ravel())
            pearson_corr, _ = pearsonr(pca_2d_data.ravel(), umap_data.ravel())

            correlation_results.append({
                "Metric": f"{pca_name}_vs_{umap_name}",
                "Spearman": spearman_corr,
                "Pearson": pearson_corr
            })

    return pd.DataFrame(correlation_results)


def compute_pairwise_distances_with_pca_umap_on_pca_variants(
    datasets: List[pd.DataFrame],
    pca_variants: List[np.ndarray],
    umap_variants: List[np.ndarray],
    umap_on_pca_variants: List[np.ndarray],
    metrics: List[str]
) -> Dict[str, np.ndarray]:
    """Compute pairwise distances for raw, PCA, UMAP, and UMAP-on-PCA-transformed variants."""
    results = {}

    # Compute for raw data
    for metric in metrics:
        distances = compute_pairwise_distances(datasets, metric)
        results[f"raw_{metric}"] = distances

    # Compute for PCA-transformed data
    pca_names = ["PCA_Standardized", "PCA_MinMax", "PCA_Centered"]
    for pca_data, name in zip(pca_variants, pca_names):
        for metric in metrics:
            pca_distances = compute_pairwise_distances(
                [pd.DataFrame(pca_data[sampled_data.index]) for sampled_data in datasets], metric
            )
            results[f"{name}_{metric}"] = pca_distances

    # Compute for UMAP-transformed data
    umap_names = ["UMAP_Standardized", "UMAP_MinMax", "UMAP_Centered"]
    for umap_data, name in zip(umap_variants, umap_names):
        for metric in metrics:
            umap_distances = compute_pairwise_distances(
                [pd.DataFrame(umap_data[sampled_data.index]) for sampled_data in datasets], metric
            )
            results[f"{name}_{metric}"] = umap_distances

    # Compute for UMAP-on-PCA-transformed data
    umap_on_pca_names = ["UMAP_on_PCA_Standardized", "UMAP_on_PCA_MinMax", "UMAP_on_PCA_Centered"]
    for umap_on_pca_data, name in zip(umap_on_pca_variants, umap_on_pca_names):
        for metric in metrics:
            umap_on_pca_distances = compute_pairwise_distances(
                [pd.DataFrame(umap_on_pca_data[sampled_data.index]) for sampled_data in datasets], metric
            )
            results[f"{name}_{metric}"] = umap_on_pca_distances

    return results

def compute_pairwise_distances_with_centroids(
    datasets: List[pd.DataFrame],
    pca_2d_variants: List[np.ndarray],
    umap_variants: List[np.ndarray],
    centroids: Dict[str, np.ndarray],
    metrics: List[str]
) -> Dict[str, np.ndarray]:
    """Compute pairwise distances for raw data, PCA2D, UMAP, and centroids."""
    results = {}

    # Compute for raw data
    for metric in metrics:
        distances = compute_pairwise_distances(datasets, metric)
        results[f"raw_{metric}"] = distances

    # Compute for PCA2D
    pca_2d_names = ["PCA2D_Standardized", "PCA2D_MinMax", "PCA2D_Centered"]
    for pca_2d_data, name in zip(pca_2d_variants, pca_2d_names):
        for metric in metrics:
            pca_2d_distances = compute_pairwise_distances(
                [pd.DataFrame(pca_2d_data[sampled_data.index]) for sampled_data in datasets], metric
            )
            results[f"{name}_{metric}"] = pca_2d_distances

    # Compute for UMAP-transformed data
    umap_names = ["UMAP_Standardized", "UMAP_MinMax", "UMAP_Centered"]
    for umap_data, name in zip(umap_variants, umap_names):
        for metric in metrics:
            umap_distances = compute_pairwise_distances(
                [pd.DataFrame(umap_data[sampled_data.index]) for sampled_data in datasets], metric
            )
            results[f"{name}_{metric}"] = umap_distances

    # Compute for centroids
    for key, centroid_data in centroids.items():
        for metric in metrics:
            centroid_distances = compute_pairwise_distances([centroid_data], metric)
            results[f"centroids_{key}_{metric}"] = centroid_distances

    return results


def compute_correlations_with_centroids(
    distances_dict: Dict[str, np.ndarray],
    pca_2d_variants: List[np.ndarray],
    umap_variants: List[np.ndarray],
    centroids: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compute correlations for distances between PCA2D, UMAP, and centroids."""
    correlation_results = []

    pca_2d_names = ["PCA2D_Standardized", "PCA2D_MinMax", "PCA2D_Centered"]
    umap_names = ["UMAP_Standardized", "UMAP_MinMax", "UMAP_Centered"]

    for pca_2d_data, pca_name in zip(pca_2d_variants, pca_2d_names):
        for umap_data, umap_name in zip(umap_variants, umap_names):
            spearman_corr, _ = spearmanr(pca_2d_data.ravel(), umap_data.ravel())
            pearson_corr, _ = pearsonr(pca_2d_data.ravel(), umap_data.ravel())

            correlation_results.append({
                "Metric": f"{pca_name}_vs_{umap_name}",
                "Spearman": spearman_corr,
                "Pearson": pearson_corr
            })

    # Compute correlations for centroids
    for metric_name, metric_values in distances_dict.items():
        for key, centroid_data in centroids.items():
            spearman_corr, _ = spearmanr(metric_values.ravel(), centroid_data.ravel())
            pearson_corr, _ = pearsonr(metric_values.ravel(), centroid_data.ravel())

            correlation_results.append({
                "Metric": f"{metric_name}_vs_centroids_{key}",
                "Spearman": spearman_corr,
                "Pearson": pearson_corr
            })

    return pd.DataFrame(correlation_results)


def compute_mmd(
    dist_mat: np.ndarray,
    indexes_x: Union[List[int], np.ndarray],
    indexes_y: Union[List[int], np.ndarray],
    bandwidth: float
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two dataset subsets 
    given a precomputed distance matrix, ensuring no duplicate elements and 
    removing diagonal values from Kxx and Kyy.

    Parameters:
        dist_mat (np.ndarray): Precomputed pairwise distance matrix (shape: [N, N]).
        indexes_x (List[int] or np.ndarray): Indices of samples from distribution P (X).
        indexes_y (List[int] or np.ndarray): Indices of samples from distribution Q (Y).
        bandwidth (float): Bandwidth parameter for the RBF kernel.

    Returns:
        float: MMD squared value.
    """
    def rbf_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
        """Computes the RBF kernel values from a distance matrix."""
        return np.exp(-distances / bandwidth)

    # Extract relevant submatrices
    Kxx_full = rbf_kernel(dist_mat[np.ix_(indexes_x, indexes_x)], bandwidth)
    Kyy_full = rbf_kernel(dist_mat[np.ix_(indexes_y, indexes_y)], bandwidth)
    Kxy = rbf_kernel(dist_mat[np.ix_(indexes_x, indexes_y)], bandwidth)

    # Get upper triangular indices without diagonal (avoid duplicate values)
    triu_idx_x = np.triu_indices(len(indexes_x), k=1)  # Upper triangle indices (no diagonal)
    triu_idx_y = np.triu_indices(len(indexes_y), k=1)

    # Compute means without including diagonal elements
    mean_Kxx = np.mean(Kxx_full[triu_idx_x]) if len(triu_idx_x[0]) > 0 else 0.0
    mean_Kyy = np.mean(Kyy_full[triu_idx_y]) if len(triu_idx_y[0]) > 0 else 0.0
    mean_Kxy = np.mean(Kxy) if Kxy.size > 0 else 0.0  # All values included for Kxy

    # Compute unbiased MMD^2
    MMD2 = mean_Kxx + mean_Kyy - 2 * mean_Kxy

    return float(MMD2)


@jit(nopython=True, fastmath=True, cache=True)
def fast_rbf_kernel(d: np.ndarray, bandwidth: float) -> np.ndarray:
    """Fast RBF kernel computation using Numba."""
    return np.exp(-d / bandwidth)

@jit(nopython=True, fastmath=True, cache=True)
def compute_mmd_numba(
    dist_mat: np.ndarray,
    indexes_x: np.ndarray,
    indexes_y: np.ndarray,
    bandwidth: float
) -> float:
    """
    Fast Maximum Mean Discrepancy (MMD) computation with Numba acceleration.

    Parameters:
        dist_mat (np.ndarray): Pairwise distance matrix (shape: [N, N]).
        indexes_x (np.ndarray): Indices of samples from distribution P (X).
        indexes_y (np.ndarray): Indices of samples from distribution Q (Y).
        bandwidth (float): Bandwidth parameter for the RBF kernel.

    Returns:
        float: MMD squared value.
    """
    # Extract submatrices
    Kxx = fast_rbf_kernel(dist_mat[np.ix_(indexes_x, indexes_x)], bandwidth)
    Kyy = fast_rbf_kernel(dist_mat[np.ix_(indexes_y, indexes_y)], bandwidth)
    Kxy = fast_rbf_kernel(dist_mat[np.ix_(indexes_x, indexes_y)], bandwidth)

    # Compute mean while avoiding the diagonal
    n_x = len(indexes_x)
    n_y = len(indexes_y)

    sum_Kxx, count_Kxx = 0.0, 0
    sum_Kyy, count_Kyy = 0.0, 0
    sum_Kxy = 0.0

    for i in range(n_x):
        for j in range(i + 1, n_x):  # Only upper triangle
            sum_Kxx += Kxx[i, j]
            count_Kxx += 1

    for i in range(n_y):
        for j in range(i + 1, n_y):
            sum_Kyy += Kyy[i, j]
            count_Kyy += 1

    for i in range(n_x):
        for j in range(n_y):
            sum_Kxy += Kxy[i, j]

    mean_Kxx = sum_Kxx / count_Kxx if count_Kxx > 0 else 0.0
    mean_Kyy = sum_Kyy / count_Kyy if count_Kyy > 0 else 0.0
    mean_Kxy = sum_Kxy / (n_x * n_y) if n_x > 0 and n_y > 0 else 0.0

    return mean_Kxx + mean_Kyy - 2 * mean_Kxy


@njit(parallel=True)
def compute_mmd_numba(exp_dist_mat, indexes_x, indexes_y):
    n_x = len(indexes_x)
    n_y = len(indexes_y)

    # Initialize accumulators
    sum_Kxx = 0.0
    sum_Kyy = 0.0
    sum_Kxy = 0.0

    # Compute Kxx (excluding diagonal)
    for i in prange(n_x):
        for j in range(n_x):
            if i != j:
                sum_Kxx += exp_dist_mat[indexes_x[i], indexes_x[j]]

    # Compute Kyy (excluding diagonal)
    for i in prange(n_y):
        for j in range(n_y):
            if i != j:
                sum_Kyy += exp_dist_mat[indexes_y[i], indexes_y[j]]

    # Compute Kxy
    for i in prange(n_x):
        for j in range(n_y):
            sum_Kxy += exp_dist_mat[indexes_x[i], indexes_y[j]]

    # Final unbiased estimators
    mean_Kxx = sum_Kxx / (n_x * (n_x - 1)) if n_x > 1 else 0.0
    mean_Kyy = sum_Kyy / (n_y * (n_y - 1)) if n_y > 1 else 0.0
    mean_Kxy = sum_Kxy / (n_x * n_y)

    return mean_Kxx + mean_Kyy - 2 * mean_Kxy


def resp_to_pattern(responsibilities: np.ndarray) -> np.ndarray:
    """
    Convert a vector of responsibility values into a Responsibility Pattern fingerprint.

    Parameters:
        responsibilities (np.ndarray): Array of responsibility values for a compound.

    Returns:
        np.ndarray: Integer fingerprint vector, where values < 0.01 are set to zero,
                    and others are scaled and truncated to integers between 1 and 10.
    """
    rp = np.floor(10 * responsibilities + 0.9).astype(int)
    rp[responsibilities < 0.01] = 0
    return rp

def get_fingerprint_counts(fingerprint_array: np.ndarray) -> dict[Tuple[int, ...], int]:
    """
    Compute counts of unique fingerprints from an array of fingerprints.
    
    Parameters:
        fingerprint_array (np.ndarray): Array of fingerprints (each row is a pattern).
    
    Returns:
        dict: A dictionary where keys are tuples representing unique patterns,
              and values are the counts of occurrences.
    """
    unique_patterns, counts = np.unique(fingerprint_array, axis=0, return_counts=True)
    return {tuple(pattern): count for pattern, count in zip(unique_patterns, counts)}


def compute_rp_similarity(
    resps_gtm: np.ndarray,
    indices_A: Union[List[int], np.ndarray],
    indices_B: Union[List[int], np.ndarray],
    use_counts: bool = True,
    metric='tanimoto'
) -> float:
    """
    Compute the Tanimoto similarity between two subsets of compounds based on their
    responsibility values.

    Parameters:
        resps_gtm (np.ndarray): Array of responsibility values with shape (N, D),
                                where N is the number of compounds and D is the number of nodes.
        indices_A (List[int] or np.ndarray): Indices of compounds in the first subset (Dataset A).
        indices_B (List[int] or np.ndarray): Indices of compounds in the second subset (Dataset B).
        use_counts (bool): If True, use counts of each unique Responsibility Pattern.
                           If False, use binary presence/absence of each pattern.
        metric (str): Similarity metric to use.

    Returns:
        float: Tanimoto similarity between Dataset A and Dataset B.
    """
    

    # Generate fingerprints for Dataset A
    dataset_A = pd.DataFrame(resps_gtm[indices_A]).apply(lambda x: resp_to_pattern(x.values), axis=1)
    test_array_A = np.vstack(dataset_A.to_numpy())

    # Generate fingerprints for Dataset B
    dataset_B = pd.DataFrame(resps_gtm[indices_B]).apply(lambda x: resp_to_pattern(x.values), axis=1)
    test_array_B = np.vstack(dataset_B.to_numpy())

    # Get fingerprint counts or presence/absence
    counts_A = get_fingerprint_counts(test_array_A)
    counts_B = get_fingerprint_counts(test_array_B)

    # Align the fingerprint space
    all_patterns = set(counts_A.keys()).union(set(counts_B.keys()))
    sorted_patterns = sorted(all_patterns)

    # Build vectors
    if use_counts:
        vector_A = np.array([counts_A.get(pattern, 0) for pattern in sorted_patterns]) / test_array_A.shape[0]
        vector_B = np.array([counts_B.get(pattern, 0) for pattern in sorted_patterns])/ test_array_B.shape[0]
    else:
        vector_A = np.array([1 if pattern in counts_A else 0 for pattern in sorted_patterns])
        vector_B = np.array([1 if pattern in counts_B else 0 for pattern in sorted_patterns])

    vector_A = vector_A.reshape(1, -1).astype(np.float64)
    vector_B = vector_B.reshape(1, -1).astype(np.float64)
    # Compute and return Tanimoto/Euclidean similarity
    if metric == 'tanimoto':
        dist_val = tanimoto_int_similarity_matrix_numba(vector_A, vector_B)
    elif metric == 'euclidean':
        dist_val = euclidean_distance_square_numba(vector_A, vector_B)
    
    return dist_val


def compute_rp_coverage(
    ref_lib: np.ndarray, 
    test_lib: np.ndarray,
    use_weight: bool = True
) -> float:
    """
    Compute coverage or weighted coverage, starting directly from two NumPy arrays of
    responsibilities.

    If use_weight=True, Weighted coverage is:
        sum_{patterns in both} ref_count(pattern) / sum_{all patterns in ref} ref_count(pattern)

    If use_weight=False, Unweighted coverage is:
        (# of patterns in both ref and test) / (# of patterns in ref).

    Parameters
    ----------
    ref_lib : np.ndarray
        Shape (N_ref, D). Each row => responsibilities for one compound in reference set.
    test_lib : np.ndarray
        Shape (N_test, D). Each row => responsibilities for one compound in test set.
    use_weight : bool
        If True => weighted coverage, else unweighted coverage.

    Returns
    -------
    float
        Coverage or weighted coverage in [0,1].
    """
    # 1) Compute dictionaries of pattern -> occurrence_count for ref and test
    counts_ref = get_fingerprint_counts(ref_lib)
    counts_test = get_fingerprint_counts(test_lib)

    # 2) Use dictionary-intersection logic
    if use_weight:
        #
        # Weighted coverage:
        # sum_{p in both} ref_count(p) / sum_{p in ref} ref_count(p)
        #
        total_ref_count = sum(counts_ref.values())  # total # of comps in ref
        if total_ref_count == 0:
            return 0.0
        # Intersection of patterns
        common_patterns = counts_ref.keys() & counts_test.keys()
        # Sum reference counts for these patterns
        coverage_sum = sum(counts_ref[p] for p in common_patterns)
        coverage_value = coverage_sum / total_ref_count

    else:
        #
        # Unweighted coverage:
        # (# of patterns in both) / (# of patterns in ref)
        #
        num_ref_patterns = len(counts_ref)
        if num_ref_patterns == 0:
            return 0.0
        common_patterns = counts_ref.keys() & counts_test.keys()
        coverage_value = len(common_patterns) / num_ref_patterns

    return coverage_value

    
def resp_to_pattern_flexible(responsibilities: np.ndarray,
                             method: str = 'linear',
                             n_bins: int = 3,
                             threshold: float = 0.01,
                             custom_bin_edges: np.ndarray = None,
                             epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert an array of responsibility values into a coarse-grained fingerprint using different methods.
    
    Parameters:
        responsibilities (np.ndarray): Array of responsibility values.
        method (str): The binning method to use. Options are:
                      - 'linear': Uses evenly spaced bins over the data range.
                      - 'quantile': Uses quantile-based bins (each bin has roughly equal numbers of values).
                      - 'log': Applies a log10 transform before linear binning.
        n_bins (int): Number of bins to use (excluding values below threshold).
        threshold (float): Values below this threshold are set to 0.
        custom_bin_edges (np.ndarray, optional): If provided, these bin edges override automatic binning.
        epsilon (float): Small constant added for numerical stability when using log transform.
    
    Returns:
        np.ndarray: An integer fingerprint vector where values below the threshold are 0,
                    and others are assigned to bins 1..n_bins.
    """
    rp = np.zeros_like(responsibilities, dtype=int)
    # Consider only values above the threshold
    mask = responsibilities >= threshold
    if not np.any(mask):
        return rp  # nothing to process
    
    data = responsibilities[mask]
    
    # Optionally apply a log transform
    if method == 'log':
        data = np.log10(data + epsilon)
    
    # Use custom bin edges if provided; otherwise, compute them based on the method.
    if custom_bin_edges is not None:
        bin_edges = custom_bin_edges
    else:
        if method == 'quantile':
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(data, quantiles)
        else:  # linear (and log, which now uses linear binning on log-transformed data)
            bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
    
    # Digitize the data into bins.
    # np.digitize returns indices starting at 1; clip to ensure maximum value falls within n_bins.
    bins = np.digitize(data, bin_edges, right=False)
    bins = np.clip(bins, 1, n_bins)
    
    rp[mask] = bins
    return rp

# A majority of functions were taken from https://github.com/IBM/controlled-peptide-generation


def recrate(preds: torch.Tensor, target: torch.Tensor):
    preds = preds.argmax(-1)
    correct = torch.sum(torch.all(preds == target, dim=-1))
    total = target.shape[0]
    return correct.float() / total


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


def kl_gaussian_sharedmu(mu, logvar):
    """ analytically compute kl divergence N(mu,sigma) with N(mu, I). """
    return torch.mean(0.5 * torch.sum((logvar.exp() - 1 - logvar), 1))


def wae_mmd_gaussianprior(z, method='full_kernel'):
    """ compute MMD with samples from unit gaussian.
    MMD parametrization from cfg loaded here."""
    z_prior = torch.randn_like(z)  # shape and device
    if method == 'full_kernel':
        mmd_kwargs = {'sigma': 7.0, 'kernel': 'gaussian'}
        return mmd_full_kernel(z, z_prior, **mmd_kwargs)
    else:
        mmd_kwargs = {"sigma": 7.0,  # ~ O( sqrt(z_dim) )
                      "kernel": 'gaussian',
                      "rf_dim": 500,  # for method = rf
                      "rf_resample": False}
        return mmd_rf(z, z_prior, **mmd_kwargs)


def mmd_full_kernel(z1, z2, **mmd_kwargs):
    K11 = compute_mmd_kernel(z1, z1, **mmd_kwargs)
    K22 = compute_mmd_kernel(z2, z2, **mmd_kwargs)
    K12 = compute_mmd_kernel(z1, z2, **mmd_kwargs)
    N = z1.size(0)
    assert N == z2.size(0), 'expected matching sizes z1 z2'
    H = K11 + K22 - K12 * 2  # gretton 2012 eq (4)
    H = H - torch.diag(H)  # unbiased: delete diagonal. Makes MMD^2_u negative! (typically)
    loss = 1. / (N * (N - 1)) * H.sum()
    return loss


def mmd_rf(z1, z2, **mmd_kwargs):
    mu1 = compute_mmd_mean_rf(z1, **mmd_kwargs)
    mu2 = compute_mmd_mean_rf(z2, **mmd_kwargs)
    loss = ((mu1 - mu2) ** 2).sum()
    return loss


rf = {}


def compute_mmd_mean_rf(z, sigma, kernel, rf_dim, rf_resample=False):
    # random features approx of gaussian kernel mmd.
    # rf_resample: keep fixed base of RF? or resample RF every time?
    # Then just loss = |mu_real - mu_fake|_H
    global rf
    if kernel == 'gaussian':
        if kernel not in rf or rf_resample:
            # sample rf if it's the first time or we want to resample every time
            rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
            rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)
            rf['gaussian'] = (rf_w, rf_b)
        else:
            rf_w, rf_b = rf['gaussian']
            assert rf_w.shape == (z.shape[1], rf_dim), 'not expecting z dim or rf_dim to change'
        z_rf = compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
    else:  # kernel xxx
        raise ValueError('todo implement rf for kernel ' + kernel)
    mu_rf = z_rf.mean(0, keepdim=False)
    return mu_rf


def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
    z_emb = (z @ rf_w) / sigma + rf_b
    z_emb = torch.cos(z_emb) * (2. / rf_dim) ** 0.5
    return z_emb


def compute_mmd_kernel(x, y, sigma, kernel):
    """ x: (Nxd) y: (Mxd). sigma: kernel width """
    # adapted from https://discuss.pytorch.org/t/error-when-implementing-rbf-kernel-bandwidth-differentiation-in-pytorch/13542
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    xmy = ((x_i - y_j) ** 2).sum(2)
    if kernel == "gaussian":
        K = torch.exp(- xmy / sigma ** 2)
    elif kernel == "laplace":
        K = torch.exp(- torch.sqrt(xmy + (sigma ** 2)))
    elif kernel == "energy":
        K = torch.pow(xmy + (sigma ** 2), -.25)
    return K

# class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)

#     def get_lr(self):
#         lr_factor = self.get_lr_factor(epoch=self.last_epoch)
#         return [base_lr * lr_factor for base_lr in self.base_lrs]

#     def get_lr_factor(self, epoch):
#         lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
#         if epoch <= self.warmup:
#             lr_factor *= epoch * 1.0 / self.warmup
#         return lr_factor
