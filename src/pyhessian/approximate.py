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


def trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

        # TODO: print here the model size

        # Store there the computed matrix
        self.hess_approx_out: Optional[Dict[str, Any]] = None

    # TODO: improve type hints
    def run(self) -> Dict[str, Any]:
        """Run the Hessian matrix approximation."""
        start = time.time()

        method = self.method_config.get("method", "exact")
        LOG.info(f"Computing Hessian matrix with method {method}")
        if method == "exact":
            self.hess_approx_out = self._exact_hessian()
        elif method == "block-wise":
            self.hess_approx_out = self._block_wise_hessian()
        elif method == "sampling":
            step = self.method_config.get("step", 10)
            LOG.info(f"Sampling every {step} parameters for Hessian approximation")
            self.hess_approx_out = self._exact_hessian(step=step)
        else:
            raise ValueError(f"Unknown Hessian approximation method {method}")

        if self.hess_approx_out is None:
            raise RuntimeError("Hessian approximation failed, no matrix computed.")
        if self.hess_approx_out['hess_approx'] is None:
            raise RuntimeError(f"Hessian approximation didn't return a key with Tensor, got {self.hess_approx_out}")

        LOG.info(f"Hessian matrix approximation completed in "
                 f"{convert_sec_to_hms(time.time() - start)}")
        return self.hess_approx_out

    def _block_wise_hessian(self) -> Dict[str, Tensor]:
        """Compute the Hessian matrix block-wise (for each param block in model)."""
        approx_start = time.time()
        mean_comp_time = AverageMeter()

        # NOTE: do grad on all params
        # TODO: handle non-trainable params from beginning?
        for p in self.hessian_comp.model.parameters():
            p.requires_grad = True
        # Cycle over the param blocks (pytorch default partition)
        n_blocks = sum(1 for p in self.hessian_comp.model.parameters() if p.requires_grad)
        block_names = [p_name for p_name, p in self.hessian_comp.model.named_parameters() if p.requires_grad]
        LOG.info(f"Computing block-wise Hessian for {n_blocks} ({block_names}) blocks from {self.model_ckpt}!")

        hess_blocks: Dict[str, Tensor] = {}
        p: nn.Parameter
        for i, (p_name, p) in enumerate(self.hessian_comp.model.named_parameters(), 1):
            # Freeze all model params except the current one
            LOG.info(f"  [block {i}] Freezing model params except ones with {p_name} ({p.numel()} params)...")
            # TODO: handle params with requires_grad=False from beginning
            for model_p_name, model_p in self.hessian_comp.model.named_parameters():
                if p_name not in model_p_name:
                    # - params that aren't the current one
                    # - params with or without grad will be set to False anyway
                    model_p.requires_grad = False
                else:
                    # resets grad at each iteration
                    model_p.requires_grad = True
            LOG.info(f"  [block {i}] Trainable params: {trainable_params(self.hessian_comp.model)}")

            # Launch the brute-force approach
            start = time.time()
            block_hess_approx_out = self._exact_hessian()
            block_hess_approx = block_hess_approx_out['hess_approx']
            hess_blocks[p_name] = block_hess_approx
            mean_comp_time.update(time.time() - start)

            approx_runtime = time.time() - approx_start
            LOG.info(f"  {p_name} [{i}/{n_blocks}] "
                     f"{mean_comp_time.avg:.2f}s per block | "
                     f"{convert_sec_to_hms(approx_runtime)} current runtime | "
                     f"Hessian: {block_hess_approx.size()}")

            # Reset requires grad (NOTE: we assume to do hess over all params)
            # TODO: handle params with requires_grad=False from beginning
            # for model_p_name, model_p in self.hessian_comp.model.named_parameters():
            #     model_p.requires_grad = True

        if hess_blocks is None:
            raise RuntimeError("Block Hessian matrices dict is empty :(")

        LOG.info(f"Assembling block-diagonal Hessian with blocks {hess_blocks.keys()}")
        hess_approx = torch.block_diag(*hess_blocks.values())
        LOG.info(f"Hessian final shape {hess_approx.size()}")

        hess_approx_out = {
            "hess_approx": hess_approx,
            "block_names": hess_blocks.keys()
        }
        return hess_approx_out

    def _exact_hessian(self, step: int = 1) -> Tensor:
        """Compute the Hessian matrix for all the model parameters. Brute-force.
        - If step is >1, will work as a sampling"""
        approx_start = time.time()
        log_every = self.method_config.get("log_every", 1000)

        n_params = trainable_params(self.hessian_comp.model)
        if n_params == 0:
            raise ValueError(f"Number of params with grad is {n_params}")

        LOG.info(f"Computing full Hessian for {n_params} params with grad!")
        LOG.info(f"This will require ~{(n_params**2 * 4) / (1024**3):.2f} GB of memory")

        mean_comp_time = AverageMeter()  # store here the average computation time
        hess_cols: List[Tensor] = []     # store here the columns of the Hessian
        hess_cols_names: List[str] = []  # store here the names of the Hessian columns

        # Cycle over trainable params
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
            # Cycle over trainable param blocks
            p: nn.Parameter
            for p_name, p in self.hessian_comp.model.named_parameters():
                if not p.requires_grad:
                    # n_params already considers params with grad
                    # so we skip the params without grad
                    continue
                numel = p.numel()  # number of params in the current block
                e_i_shaped.append(e_i[idx:idx+numel].reshape(p.shape))
                idx += numel  # with non-trainable this will exceed e_i size

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
                LOG.info(f"{self.model_ckpt}: {i+1}/{n_params} columns, "
                         f"{mean_comp_time.avg:.2f}s per Hv product, "
                         f"{convert_sec_to_hms(approx_runtime)} current runtime")
                mean_comp_time.reset()

        # Build the matrix
        hess_approx = torch.stack(hess_cols, dim=1)
        hess_approx_out = {
            "hess_approx": hess_approx
        }
        return hess_approx_out

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
