"""Utilities for plotting the Hessian matrix"""

from typing import Optional
import torch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_hessian_heatmap(hess_mat: torch.Tensor, ax: Optional[Axes] = None) -> Axes:
    """Hessian matrix heatmap in log10 scale"""
    # Use log10 scale for intensities
    hess_log = torch.log10(hess_mat.squeeze(0).abs() + 1e-10).cpu().numpy()

    if ax is None:
        ax = plt.gca()

    img = ax.imshow(hess_log, cmap='viridis', aspect='auto')
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Parameter Index")
    # Add legend for color scale
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label(r'$\log_{10}(|H| + 10^{-10})$', rotation=270, labelpad=20)

    return ax


def plot_hessian_sign(hess_mat: torch.Tensor, ax: Optional[Axes] = None) -> Axes:
    """A binary plot that shows the sign of the entries"""
    # TODO: there might be problems with the max-pool, i.e. doesn't make much sense
    # TODO: maybe I can do an histogram, but I won't know where the values are located

    # Reduce the size with max-pooling for visualization
    # H_red = F.max_pool2d(hess_mat.unsqueeze(0), kernel_size=filter_size)
    # Apply the sign function
    hess_sign = hess_mat.squeeze(0).sign().cpu().numpy()

    if ax is None:
        ax = plt.gca()

    # Create a binary colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap([0., 1.])
    new_cmap = ListedColormap(colors)

    img = ax.imshow(hess_sign, cmap=new_cmap, aspect='auto')
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Parameter Index")
    # Add legend for color scale
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label(r'$\mathrm{sign}(H)$', rotation=270, labelpad=20)

    return ax
