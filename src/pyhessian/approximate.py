"""Utilities for approximating the Hessian matrix of a nn.Module"""

from .hessian import Hessian
from .utils import (
    AverageMeter, LOG, map_param_to_block_name, convert_sec_to_hms,
    hessian_vector_product)

import time
from argparse import Namespace
from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
from torch import nn, Tensor
from tqdm import tqdm


class HessianApproximator:
    def __init__(
        self,
        opts: Namespace,
        hessian_comp: Hessian,
        method_config: Dict[str, Any],
    ):
        if getattr(opts, "ckpt", None) is None:
            raise ValueError("Model checkpoint not found in opts.ckpt")

        self.model_ckpt = opts.ckpt
        self.hessian_comp = hessian_comp
        self.method_config = method_config

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

        method = self.method_config.get("method", "exact")
        LOG.info(f"Computing Hessian matrix with method {method}")
        if method == "exact":
            self.hess_approx = self._exact_hessian()
        elif method == "block-wise":
            raise NotImplementedError()
            self.hess_approx = self._block_wise_hessian()
        elif method == "sampling":
            step = self.method_config.get("step", 10)
            LOG.info(f"Sampling every {step} parameters for Hessian approximation")
            self.hess_approx = self._exact_hessian(step=step)
        else:
            raise ValueError(f"Unknown Hessian approximation method {method}")

        if self.hess_approx is None:
            raise RuntimeError("Hessian approximation failed, no matrix computed.")

        LOG.info(f"Hessian matrix approximation completed in "
                 f"{convert_sec_to_hms(time.time() - start)}")
        return self.hess_approx

    def _block_wise_hessian(self) -> Dict[str, Tensor]:
        """Compute the Hessian matrix block-wise (for each param block in model)."""
        approx_start = time.time()
        mean_comp_time = AverageMeter()

        # Cycle over the param blocks (pytorch default partition)
        blocks = self.hessian_comp.model.named_parameters()
        hess_blocks: Dict[str, Tensor] = {}
        p: nn.Parameter
        for i, (p_name, p) in enumerate(blocks, 1):
            # do the exact approch for each block
            start = time.time()
            # TODO: implement block-wise hessian
            # block_hess_approx = self._exact_hessian([p])
            # hess_blocks[p_name] = block_hess_approx
            mean_comp_time.update(time.time() - start)

            approx_runtime = time.time() - approx_start
            LOG.info(f"{self.model_ckpt}: {p_name} [{i}/{len(blocks)}]"
                     f"{mean_comp_time.avg:.2f}s per block"
                     f"{convert_sec_to_hms(approx_runtime)} current runtime")

        if hess_blocks is None:
            raise RuntimeError("Block Hessian matrices dict is empty :(")

        # TODO: otherwise a tensor with 1e-10 offdiagonal
        return hess_blocks

    def _exact_hessian(self, step: int = 1) -> Tensor:
        """Compute the Hessian matrix for all the model parameters.
        - If step is >1, will work as a sampling"""
        approx_start = time.time()
        log_every = self.method_config.get("log_every", 1000)
        n_params = sum(p.numel() for p in self.hessian_comp.model.parameters() if p.requires_grad)

        LOG.info(f"Computing full Hessian for {n_params} parameters!")
        LOG.info(f"This will require ~{(n_params**2 * 4) / (1024**3):.2f} GB of memory")

        mean_comp_time = AverageMeter()  # store here the average computation time
        hess_cols: List[Tensor] = []     # store here the columns of the Hessian
        hess_cols_names: List[str] = []  # store here the names of the Hessian columns

        for i in tqdm(
            range(0, n_params, step),
            desc="Hv prods", unit="dim", disable=True
        ):
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
            # TODO: remove here based on step
            hv_prod_flat = hv_prod_flat[0::step]
            hess_cols.append(hv_prod_flat)

            # Print status
            if i % log_every == 0:
                approx_runtime = time.time() - approx_start
                LOG.info(f"{self.model_ckpt}: {i}/{n_params} columns, "
                         f"{mean_comp_time.avg:.2f}s per Hv product, "
                         f"{convert_sec_to_hms(approx_runtime)} current runtime")
                mean_comp_time.reset()

        # Build the matrix
        H_approx = torch.stack(hess_cols, dim=1)
        return H_approx

    def export_matrix(self, output_dir: Optional[Path] = Path("."), fname: Optional[str] = None) -> None:
        """Export the computed matrix to a .pt file."""
        if self.hess_approx is None:
            raise ValueError("No Hessian matrix found at self.hess_approx")

        LOG.debug(f"Checking directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / (fname if fname else self.model_ckpt)
        torch.save(self.hess_approx, data_path)
        LOG.info(f"Dumped Hessian matrix data at {data_path}")

    def plot_matrix(self) -> None:
        raise NotImplementedError()
