#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

"""Utilities for plotting eigenvalue density of Hessian matrices."""

from .utils import condition_number

import time
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_eigenvalue_density(
    eigenvalues: NDArray[np.floating],
    weights: NDArray[np.floating],
    ax: Optional[Axes] = None,
    eigen_abs: bool = True
) -> Axes:
    """
    Plot the eigenvalue spectral density on a matplotlib Axes.
    
    This function visualizes the distribution of eigenvalues from the Hessian matrix
    using a smoothed density estimate. The density is computed using Gaussian kernel
    density estimation and plotted on a log scale for better visualization of the
    full range of densities.
    
    Args:
        eigenvalues: Array of eigenvalues, shape (n_runs, n_eigenvalues_per_run)
                    From the output of Hessian.density()
        weights: Array of corresponding weights for density estimation,
                shape (n_runs, n_eigenvalues_per_run). These are typically the
                squared first components of Lanczos eigenvectors
        ax: Matplotlib Axes object to plot on. If None, uses current axes (plt.gca())
        eigen_abs: If True, plot density using absolute values of eigenvalues
        
    Returns:
        The Axes object with the density plot
        
    Example:
        >>> hessian = Hessian(model, criterion, data=batch)
        >>> eigenvalues, weights = hessian.density(iter=100, n_v=1)
        >>> fig, ax = plt.subplots()
        >>> plot_eigenvalue_density(eigenvalues, weights, ax=ax)
        >>> fig.savefig('eigenvalue_density.png')
    """
    # Ensure inputs are numpy arrays to avoid Python list/int comparison errors
    # TODO: check pyhessian density method
    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    # Compute fraction of positive eigenvalues robustly (works for 1D or 2D input)
    ev_flat = eigenvalues.ravel()  # contiguous flattened array
    if ev_flat.size == 0:
        frac_pos = 0.0
    else:
        frac_pos = float(np.sum(ev_flat > 0)) / float(ev_flat.size)

    # If requested, operate on absolute eigenvalues for density estimation
    if eigen_abs:
        eigenvalues_proc = np.abs(eigenvalues)
    else:
        eigenvalues_proc = eigenvalues

    # density_generate expects shape (n_runs, n_eigenvalues_per_run). If a single
    # run (1D array) was provided, expand dims so axis operations inside work.
    if eigenvalues_proc.ndim == 1:
        eigenvalues_for_density = eigenvalues_proc[np.newaxis, :]
    else:
        eigenvalues_for_density = eigenvalues_proc

    if weights.ndim == 1:
        weights_for_density = weights[np.newaxis, :]
    else:
        weights_for_density = weights

    # Generate the density estimate from (possibly transformed) eigenvalues and weights
    density, grids = density_generate(eigenvalues_for_density, weights_for_density)
    assert density.shape == grids.shape, "Density and grid shapes do not match."

    if ax is None:
        ax = plt.gca()

    # Small epsilon to avoid log(0) issues when using log scales
    eps = 1e-12

    if eigen_abs:
        # Only keep strictly positive grid points for log-scale x-axis
        mask = grids > 0
        grids_pos = grids[mask]
        density_pos = density[mask]

        if grids_pos.size == 0:
            # Fallback: nothing positive in grid (very unusual). Plot entire density
            grids_pos = grids
            density_pos = density

        # Clamp density to a tiny positive value to avoid log(0) or negative values
        density_pos = np.maximum(density_pos, eps)

        ax.plot(grids_pos, density_pos)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}$ Absolute Eigenvalue')
        ax.set_ylabel(r'$\log_{10}$ Density')
    else:
        # For signed eigenvalues, show on semilogy to capture a wide dynamic range
        ax.semilogy(grids, density + eps)
        ax.set_xlabel('Eigenvalue')
        ax.set_ylabel(r'$\log_{10}$ Density')
        ax.tick_params(axis='both', which='major')

        # Set x-axis limits based on original eigenvalue range with small padding
        eig_min = np.min(eigenvalues)
        eig_max = np.max(eigenvalues)
        ax.set_xlim(eig_min - 1, eig_max + 1)

    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
    ax.set_title('Eigenvalue Density')
    # add condition number and fraction of positive eigenvalues to graph space
    y_offset = 0.8
    ax.annotate(f'p(lambd>0)={frac_pos:.2f}', xy=(0.05, y_offset + 0.06), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    ax.annotate(f'cond={condition_number(eigenvalues):.2e}', xy=(0.05, y_offset), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    return ax


def density_generate(
    eigenvalues: NDArray[np.floating],
    weights: NDArray[np.floating],
    num_bins: int = 10000,
    sigma_squared: float = 1e-5,
    overhead: float = 0.01
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate smoothed density estimate from eigenvalues using Gaussian kernel density estimation.
    
    This function takes raw eigenvalues and weights from the Stochastic Lanczos Quadrature
    algorithm and produces a smooth density curve. The density is computed by:
    1. Creating a grid of points between min and max eigenvalues
    2. For each grid point, computing a weighted sum of Gaussian kernels centered at each eigenvalue
    3. Averaging across multiple runs and normalizing
    
    Args:
        eigenvalues: Array of eigenvalues, shape (n_runs, n_eigenvalues_per_run)
                    Multiple runs are averaged to reduce stochastic noise
        weights: Array of weights for each eigenvalue, shape (n_runs, n_eigenvalues_per_run)
                These weights come from the Lanczos algorithm and determine the contribution
                of each eigenvalue to the density
        num_bins: Number of grid points for the density curve (higher = smoother but slower)
        sigma_squared: Base variance for Gaussian kernels, scaled by eigenvalue range
        overhead: Small padding added to min/max eigenvalues to ensure full coverage
        
    Returns:
        A tuple containing:
            - density: Array of density values at each grid point, normalized to integrate to 1
            - grids: Array of eigenvalue grid points where density is evaluated
            
    Note:
        The sigma (kernel width) is automatically scaled based on the eigenvalue range to
        provide appropriate smoothing regardless of the scale of eigenvalues.
    """
    # Convert to numpy arrays for easier manipulation
    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    # Compute eigenvalue range with small padding
    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    # Create uniform grid spanning the eigenvalue range
    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    # Scale kernel width based on eigenvalue range (adaptive bandwidth)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    # Compute density for each run
    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            # Evaluate Gaussian kernel at each eigenvalue
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            # Weight by Lanczos weights and sum
            density_output[i, j] = np.sum(tmp_result * weights[i, :])

    # Average density across all runs
    density = np.mean(density_output, axis=0)

    # Normalize density to integrate to 1 (trapezoidal rule)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization

    return density, grids


def gaussian(
    x: NDArray[np.floating],
    x0: float,
    sigma_squared: float
) -> NDArray[np.floating]:
    """
    Evaluate Gaussian (normal) probability density function.
    
    Computes the value of a Gaussian distribution with mean x0 and variance sigma_squared
    at point(s) x. This is used as a kernel function for density estimation.
    
    Args:
        x: Point(s) at which to evaluate the Gaussian (can be array or scalar)
        x0: Mean (center) of the Gaussian distribution
        sigma_squared: Variance (σ²) of the Gaussian distribution
        
    Returns:
        The value(s) of the Gaussian PDF: (1/√(2πσ²)) * exp(-(x-x0)²/(2σ²))
        
    Mathematical formula:
        f(x; μ, σ²) = (1 / √(2πσ²)) * exp(-(x - μ)² / (2σ²))
        
    Note:
        This is a properly normalized probability density function that integrates to 1
    """
    return np.exp(-((x0 - x) ** 2) / (2.0 * sigma_squared)) / np.sqrt(
        2 * np.pi * sigma_squared
    )
