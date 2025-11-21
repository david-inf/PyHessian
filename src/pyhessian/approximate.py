"""Utilities for approximating the Hessian matrix of a nn.Module"""

from .hessian import Hessian
from .utils import AverageMeter, LOG, map_param_to_block_name, convert_sec_to_hms

import time
from argparse import Namespace
from typing import Optional, List
from pathlib import Path
import torch
from torch import nn, Tensor
from tqdm import tqdm


class HessianApproximator:
    def __init__(
        self,
        opts: Namespace,
        hessian_comp: Hessian,
        method: str = 'exact',
    ):
        if getattr(opts, "ckpt", None) is None:
            raise ValueError("Model checkpoint not found in opts.ckpt")

        self.model_ckpt = opts.ckpt
        self.hessian_comp = hessian_comp
        self.method = method

        # Device handling logic
        try:
            self.device = hessian_comp.model.device
            if self.device == torch.device("cpu"):
                LOG.warning("Hessian computation running on CPU, this may be slow.")
        except AttributeError:
            self.device = torch.device("cpu")
            LOG.warning("Hessian computation running on CPU, this may be slow.")

        # Store there the computed matrix
        self.hess_approx: Optional[Tensor] = None

    def run(self) -> Tensor:
        """Run the Hessian matrix approximation."""
        start = time.time()

        if self.method == "exact":
            self.hess_approx = self._exact_hessian()
        else:
            raise ValueError(f"Unknown Hessian approximation method {self.method}")

        if self.hess_approx is None:
            raise RuntimeError("Hessian approximation failed, no matrix computed.")

        LOG.info(f"Hessian matrix approximation completed in "
                 f"{convert_sec_to_hms(time.time() - start)}")
        return self.hess_approx

    def _block_wise_hessian(self) -> Tensor:
        """Compute the Hessian matrix block-wise (for each param block in model)."""
        raise NotImplementedError()

    def _exact_hessian(self) -> Tensor:
        """Compute the Hessian matrix for all the model parameters."""
        approx_start = time.time()
        n_params = sum(p.numel() for p in self.hessian_comp.model.parameters() if p.requires_grad)

        mean_comp_time = AverageMeter()
        hess_cols: List[Tensor] = []     # store here the columns of the Hessian
        hess_cols_names: List[str] = []  # store here the names of the Hessian columns

        for i in tqdm(range(n_params), desc="Hv prods", unit="dim", disable=True):
            # Create i-th standard basis vector
            e_i = torch.zeros(n_params, device=self.device)
            e_i[i] = 1.0
            # Map param i to its name
            name_dict = map_param_to_block_name(i, self.hessian_comp)
            hess_cols_names.append(name_dict['p_name'])

            # Reshape to match parameter structure
            e_i_shaped = []
            idx = 0
            # TODO: use somehow the param name
            p: nn.Parameter
            for p_name, p in self.hessian_comp.model.named_parameters():
                # Cycle over param blocks
                numel = p.numel()  # number of params in the current block
                e_i_shaped.append(e_i[idx:idx+numel].reshape(p.shape))
                idx += numel

            # Compute the Hessian-vector product H * e_i -- delegate to pyhessian
            start = time.time()
            _, hv_prod = self.hessian_comp.dataloader_hv_product(e_i_shaped)
            mean_comp_time.update(time.time() - start)

            # Flatten the product so to have the column
            hv_prod_flat = torch.cat([h.flatten() for h in hv_prod])
            hess_cols.append(hv_prod_flat)

            # Print status
            if i % 1000 == 0:
                approx_runtime = time.time() - approx_start
                LOG.info(f"{self.ckpt}: {i}/{n_params} columns, "
                         f"{mean_comp_time.avg:.2f}s for Hv product, "
                         f"{convert_sec_to_hms(approx_runtime)} total runtime")
                mean_comp_time.reset()

        # Build the matrix
        H_approx = torch.stack(hess_cols, dim=1)
        return H_approx

    def export_matrix(self, output_dir: Path) -> None:
        """Export the computed matrix to a .pt file."""
        if self.hess_approx is None:
            raise ValueError("No Hessian matrix found at self.hess_approx")

        LOG.debug(f"Checking directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / self.model_ckpt
        torch.save(self.hess_approx, data_path)
        LOG.info(f"Dumped Hessian matrix data at [bold purple]{data_path}[/bold purple]")

    def plot_matrix(self) -> None:
        raise NotImplementedError()
