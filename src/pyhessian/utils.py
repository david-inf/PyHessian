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

"""Utility functions for Hessian computation in neural networks."""

import time
from typing import Tuple, List, Dict, Union, TYPE_CHECKING
import torch
from torch import nn
from torch import Tensor

if TYPE_CHECKING:
    from .hessian import Hessian

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Configure logging
monitor_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
})
console = Console(theme=monitor_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M",
    handlers=[RichHandler(
        console=console,
        rich_tracebacks=True,
    )],
)

LOG = logging.getLogger("rich")


def group_product(xs: List[Tensor], ys: List[Tensor]) -> Tensor:
    """
    Compute the inner product of two lists of tensors.
    
    This function computes the sum of element-wise products across all corresponding
    tensor pairs in the two lists. Mathematically: sum_i <xs[i], ys[i]>
    
    Args:
        xs: First list of tensors (typically model parameters or gradients)
        ys: Second list of tensors (must have same structure as xs)
        
    Returns:
        A scalar tensor representing the total inner product
        
    Example:
        >>> xs = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
        >>> ys = [torch.tensor([5., 6.]), torch.tensor([7., 8.])]
        >>> group_product(xs, ys)  # (1*5 + 2*6) + (3*7 + 4*8) = 17 + 53 = 70
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params: List[Tensor], update: List[Tensor], alpha: float = 1.0) -> List[Tensor]:
    """
    Perform in-place addition: params = params + update * alpha.
    
    This function adds a scaled update to each parameter tensor in-place.
    Commonly used in optimization algorithms and for vector operations in
    eigenvalue computations.
    
    Args:
        params: List of parameter tensors to be updated (modified in-place)
        update: List of update tensors (same structure as params)
        alpha: Scaling factor for the update (default: 1)
        
    Returns:
        The updated params list (same object as input, modified in-place)
        
    Example:
        >>> params = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
        >>> update = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
        >>> group_add(params, update, alpha=2.0)  # params += 2 * update
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v: List[Tensor]) -> List[Tensor]:
    """
    Normalize a list of vectors to unit length.
    
    Computes the L2 norm across all tensors in the list and divides each element
    by this norm to create a unit vector. The norm is computed as sqrt(sum of all
    squared elements across all tensors).
    
    Args:
        v: List of tensors representing a vector in the parameter space
        
    Returns:
        Normalized list of tensors with unit L2 norm
        
    Note:
        A small epsilon (1e-6) is added to prevent division by zero
    """
    # Compute L2 norm: ||v|| = sqrt(v^T v)
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    # Normalize: v / ||v||
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model: nn.Module) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Extract model parameters and their corresponding gradients.
    
    This function iterates through all model parameters that require gradients
    and collects both the parameters themselves and their gradient values.
    If a parameter has no gradient (e.g., hasn't been used in backward pass yet),
    a zero tensor is used instead.
    
    Args:
        model: The PyTorch model from which to extract parameters and gradients
        
    Returns:
        A tuple containing:
            - params: List of parameter tensors that require gradients
            - grads: List of corresponding gradient tensors (or zeros if grad is None)
            
    Note:
        Parameters with requires_grad=False are skipped
    """
    params: List[Tensor] = []
    grads: List[Tensor] = []

    param: nn.Parameter
    for param in model.parameters():
        # Skip parameters that don't require gradients (frozen layers, etc.)
        if not param.requires_grad:
            continue
        params.append(param)
        # Use zero if gradient hasn't been computed yet, otherwise use the gradient
        # The "+ 0." creates a copy to avoid in-place modification issues
        grads.append(0. if param.grad is None else param.grad + 0.)

    return params, grads


def hessian_vector_product(
        gradsH: List[Tensor],
        params: List[Tensor],
        v: List[Tensor]
    ) -> Tuple[Tensor, ...]:
    """
    Compute the Hessian-vector product H*v using automatic differentiation.
    
    This function computes the product of the Hessian matrix with a vector without
    explicitly constructing the Hessian. It uses the identity:
    H*v = ∇(∇L^T * v) where ∇L is the gradient of the loss.
    
    The computation is efficient because it only requires two backward passes and
    has memory complexity O(n) rather than O(n²) for storing the full Hessian.
    
    Args:
        gradsH: List of gradient tensors (∇L) at the current point
        params: List of model parameter tensors
        v: List of direction vectors (must match structure of params)
        
    Returns:
        A tuple of tensors representing H*v, with the same structure as params
        
    Note:
        - Requires that the gradient computation graph is retained (create_graph=True)
        - Uses retain_graph=True to allow multiple Hessian-vector products
    """
    hv = torch.autograd.grad(
        gradsH,
        params,
        grad_outputs=v,
        only_inputs=True,
        retain_graph=True
    )
    return hv


def orthnormal(w: List[Tensor], v_list: List[List[Tensor]]) -> List[Tensor]:
    """
    Orthonormalize vector w against a list of vectors, then normalize.
    
    This function implements the Gram-Schmidt orthogonalization process:
    1. For each vector v in v_list, subtract the projection of w onto v
    2. After orthogonalization, normalize w to unit length
    
    This is commonly used in iterative eigenvalue algorithms (power iteration,
    Lanczos) to maintain an orthonormal basis.
    
    Args:
        w: The vector to be orthonormalized (list of tensors)
        v_list: List of vectors against which to orthogonalize (list of lists of tensors)
        
    Returns:
        The orthonormalized vector with unit length
        
    Mathematical formulation:
        For each v in v_list: w = w - <w, v> * v
        Then: w = w / ||w||
        
    Example:
        Used to ensure eigenvectors remain orthogonal during power iteration
    """
    # Gram-Schmidt: subtract projection onto each vector in v_list
    for v in v_list:
        # Compute projection: <w, v> and subtract: w = w - <w,v>*v
        w = group_add(w, v, alpha=-group_product(w, v))
    # Normalize the resulting orthogonal vector
    return normalization(w)


def map_param_to_block_name(param_idx: int, hessian_comp: "Hessian") -> Dict[str, Union[str, List[int]]]:
    """
    Given a parameter index, returns back the name of its param block.
    
    This function iterates through the model's named parameters and constructs
    a dict that maps each parameter index to its block name (layer/module name).
    This is useful for interpreting Hessian computations in terms of model structure.
    
    Args:
        param_idx: The parameter index
        hessian_comp: An instance of the Hessian class containing the model
        
    Returns:
        A dict containing the block name and the corresponding parameter index range.
    """
    out = {}  # output
    param_offset = 0  # counter for param block ranges
    for p_name, p in hessian_comp.model.named_parameters():
        # Cycle over param blocks and take the ones with gradient
        if p.requires_grad:
            numel = p.numel()  # number of params in the current block
            if param_offset <= param_idx < param_offset + numel:
                out['p_name'] = p_name
                out['block_range'] = [param_offset, param_offset + numel]
                break
            param_offset += numel

    if not out:
        raise ValueError(f"Parameter index {param_idx} does not correspond to any parameter block.")

    return out


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        """Initialize the AverageMeter with default values."""
        # store metric statistics
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with new value.

        Args:
            val: The value to update with
            n: Weight of the value (default: 1)
        """
        # update statistic with given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_sec_to_hms(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))
