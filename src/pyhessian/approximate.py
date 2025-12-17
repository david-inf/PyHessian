"""Utilities for approximating the Hessian matrix of a nn.Module"""

from .hessian import Hessian
from .utils import (
    AverageMeter, LOG, map_param_to_block_name, convert_sec_to_hms)
from .hessian_plot import plot_hessian_heatmap
from .density_plot import plot_eigenvalue_density

import math
import time
from argparse import Namespace
from typing import Optional, List, Dict, Any,  Tuple
from pathlib import Path
import json
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""TODO:
- Check memory-efficiency
  - Never hold the full matrix in memory
- Block-wise:
  - Create folders and dump here each block, then reconstruct on demand
- Aggressive sampling (required for embd and head layers)
"""


class HessianApproximator:
    def __init__(
        self,
        opts: Namespace,
        hessian_comp: Hessian,
        method_config: Dict[str, Any],
        plot_config: Optional[Dict[str, Any]] = None,
    ):
        if getattr(opts, "ckpt", None) is None:
            raise ValueError("Model checkpoint not found in opts.ckpt")

        self.model_ckpt = opts.ckpt  # NOTE: can be annoying, ckpt must be loaded outside
        self.hessian_comp = hessian_comp
        self.method_config = method_config
        self.plot_config = plot_config

        # Output path for Hessian matrix data handling
        if not str(self.model_ckpt).endswith('.pt'):
            self.model_ckpt = f"{self.model_ckpt}.pt"
        self.output_path: Path = self.method_config.get("output_path", Path(".") / self.model_ckpt)
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        # Device handling logic
        try:
            self.device = hessian_comp.device
            if self.device == torch.device("cpu"):
                LOG.warning("Hessian computation running on CPU, this may be slow.")
        except AttributeError:
            self.device = torch.device("cpu")
            LOG.warning("Hessian computation running on CPU, this may be slow.")
        LOG.info(f"Computation will be on {self.device}")

        # Store metadata, optionally the Hessian matrix itself
        self.hess_approx_out: Optional[Dict[str, Any]] = None

    def run(self, plot: bool = True) -> None:
        """Run the Hessian matrix approximation."""
        start = time.time()

        # TODO: would be better to parallelize the for loops
        method = self.method_config.get("method", "exact")
        LOG.info(f"Computing Hessian matrix with method {method}")
        if method == "exact":
            self.hess_approx_out = self._exact_hessian()
        elif method == "block-wise":
            self.hess_approx_out = self._block_wise_hessian()
        elif method == "sampling":
            # Aggressive sampling: stride or uniform random selection (always square matrix)
            step = self.method_config.get("step", None)
            sample_size = self.method_config.get("sample_size", None)
            sampling_mode = self.method_config.get("sampling_mode", "stride" if step else "uniform")
            LOG.info(
                f"Sampling mode={sampling_mode} | step={step} | sample_size={sample_size}"
            )
            self.hess_approx_out = self._exact_hessian(
                step=step if step else 1,
                sampling_mode=sampling_mode,
                sample_size=sample_size,
            )
        else:
            raise ValueError(f"Unknown Hessian approximation method {method}")

        if self.hess_approx_out is None:
            raise RuntimeError("Hessian approximation failed, no matrix computed.")

        LOG.info(f"Hessian matrix approximation completed in "
                 f"{convert_sec_to_hms(time.time() - start)}")
        LOG.info(f"Hessian matrix should be at {self.hess_approx_out['output_path']}")
        LOG.info(f"  shape: {self.hess_approx_out['shape']}"
                f", dtype: {self.hess_approx_out['dtype']}"
                # f", range: [{self.hess_approx_out['range'][0]:.4e}, "
                # f"{self.hess_approx_out['range'][1]:.4e}]"
            )

        if plot:
            LOG.info("Plotting Hessian matrix...")
            self.plot_hessian()

    # TODO: refactor for numpy memmap
    def _block_wise_hessian(self) -> Dict[str, Any]:
        """Compute the Hessian matrix block-wise (for each param block in model)."""
        approx_start = time.time()
        mean_comp_time = AverageMeter()

        # Original params setting (will be reset at the end)
        original_requires_grad = {
            p_name: p.requires_grad
            for p_name, p in self.hessian_comp.model.named_parameters()
        }

        # Blocks that needs Hessian computation
        blocks_with_grad = [
            p_name for p_name, p in self.hessian_comp.model.named_parameters()
            if p.requires_grad
        ]
        n_blocks = len(blocks_with_grad)
        if not blocks_with_grad:
            raise ValueError("No trainable parameters found for block-wise Hessian computation.")

        LOG.info("="*20)
        LOG.info(f"Computing block-wise Hessian for {n_blocks} blocks from {self.model_ckpt}:")
        LOG.info(blocks_with_grad)

        hess_blocks: Dict[str, Tensor] = {}
        # Cycle over param blocks with grad
        p: nn.Parameter
        for i, p_name in enumerate(blocks_with_grad, 1):
            LOG.info(f"  [{p_name} {i}/{n_blocks}] Computing Hessian for param block {p_name}...")

            # Freeze all model params except the current one
            for model_p_name, model_p in self.hessian_comp.model.named_parameters():
                if model_p_name == p_name:
                    model_p.requires_grad = True
                else:
                    model_p.requires_grad = False

            LOG.info(f"  [{p_name} {i}/{n_blocks}] Params with grad: {trainable_params(self.hessian_comp.model)}")

            # Launch the brute-force approach
            start = time.time()
            block_hess_approx_out = self._exact_hessian()
            block_hess_approx = block_hess_approx_out['hess_approx']
            hess_blocks[p_name] = block_hess_approx
            # TODO: need a way to clear memory after computing each block
            # there's no space for keeping all blocks in memory
            # this also applies when plotting the matrix
            # need a way to do that without having the whole matrix in memory
            mean_comp_time.update(time.time() - start)

            approx_runtime = time.time() - approx_start
            LOG.info(f"  [{p_name} {i}/{n_blocks}] Computed block Hessian in "
                     f"{mean_comp_time.avg:.2f}s per block | "
                     f"{convert_sec_to_hms(approx_runtime)} current runtime | "
                     f"Hessian ({p_name}): {block_hess_approx.size()}")

            # Dump partial Hessian (in case of job failure)
            hess_approx = torch.block_diag(*hess_blocks.values())
            self.hess_approx_out = {
                "hess_approx": hess_approx,
                "block_names": hess_blocks.keys()
            }
            self.export_matrix()
            self.plot_hessian()

        # Restore original requires_grad settings
        for p_name, p in self.hessian_comp.model.named_parameters():
            p.requires_grad = original_requires_grad[p_name]

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

    def _exact_hessian(
        self,
        step: int = 1,
        sampling_mode: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute a square Hessian matrix for all the model parameters.
        - If step is >1, will work as a stride sampling
        - sampling_mode: 'stride' (default) or 'uniform'
        - sample_size: number of rows/cols to sample (overrides step if provided)
        """
        approx_start = time.time()
        log_every = self.method_config.get("log_every", 1000)

        n_params = trainable_params(self.hessian_comp.model)
        if n_params == 0:
            raise ValueError(f"Number of params with grad is {n_params}")

        # Build square index set for sampling
        if sampling_mode is None and sample_size is None:
            # Default: stride sampling
            indices = list(range(0, n_params, step))
        else:
            sampling_mode = sampling_mode or "stride"
            if sampling_mode == "uniform":
                rng = np.random.default_rng(self.method_config.get("seed", 42))
                k = sample_size if sample_size is not None else max(1, n_params // max(1, step))
                indices = sorted(rng.choice(n_params, size=min(k, n_params), replace=False).tolist())
            elif sampling_mode == "stride":
                stride = step if sample_size is None else max(1, n_params // max(1, sample_size))
                indices = list(range(0, n_params, max(1, stride)))
            else:
                raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

        n = len(indices)
        streaming = self.method_config.get("streaming", False)
        if streaming:
            # Change extension to .npy for numpy memmap format
            output_path = self.output_path.with_suffix('.npy')
            # Storage dtype for the streamed matrix (controls file size/precision)
            dtype = self.method_config.get("dtype", np.float16)
            LOG.info(
                f"Using streaming Hessian writer to dump matrix to {output_path} "
                f"without keeping it in memory | shape=({n}, {n})."
            )
            hess_writer = HessianWriter(
                n_rows=n, n_cols=n,
                output_path=output_path, dtype=dtype
            )
        else:
            output_path = self.output_path

        mean_comp_time = AverageMeter()  # store here the average computation time
        # Hessian columns storage (if not with data streaming)
        hess_cols: List[Tensor] = [] if not streaming else None
        hess_cols_names: List[str] = []

        # Cycle over sampled indices
        for j, i in enumerate(tqdm(indices, desc="Hv prods", unit="dim", disable=True)):
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
            # Slice rows to match sampled indices (always square)
            # NOTE: The product with all params is still computed
            hv_prod_flat = hv_prod_flat[indices]
            # Store the column
            if streaming and hess_writer is not None:
                hess_writer.write_col(hv_prod_flat)
            else:
                hess_cols.append(hv_prod_flat)

            # Print status
            if (j % max(1, log_every)) == 0:
                approx_runtime = time.time() - approx_start
                LOG.info(f"{self.model_ckpt}: {i+1}/{n_params} columns, "
                         f"{mean_comp_time.avg:.2f}s per Hv product, "
                         f"{convert_sec_to_hms(approx_runtime)} current runtime")
                mean_comp_time.reset()

        if streaming and hess_writer is not None:
            hess_writer.flush()
            hess_writer.close()
            LOG.info(f"Completed streaming Hessian matrix to {output_path}")
            self.hess_approx_out = {
                "hess_approx": None,
                "output_path": output_path,
                "shape": (n, n),
                "range": (None, None),
                "dtype": hess_writer.dtype,
                "param_names": hess_cols_names,
                "streaming": True
            }
            return self.hess_approx_out

        hess_approx = torch.stack(hess_cols, dim=1).detach().cpu()
        self.hess_approx_out = {
            "hess_approx": hess_approx,
            "output_path": output_path,
            "shape": hess_approx.size(),
            "range": (hess_approx.min().item(), hess_approx.max().item()),
            "dtype": hess_approx.dtype,
            "param_names": hess_cols_names
        }
        # Save tensor when not streaming
        self.export_matrix()
        return self.hess_approx_out

    def load_hessian(
        self,
        data_path: Optional[Path] = None,
        shape: Optional[Tuple[int, int]] = None,
        target_size: Optional[int] = None,
        streaming: Optional[bool] = None
    ) -> Tensor:
        """Get the computed Hessian matrix. Memory has precedence over disk loading.
        If streaming is True, load from disk with possible downsampling.
        If streaming is None, auto-detect the format based on the file.

        target_size is only used for streaming mode to limit the size of the loaded matrix.
        """
        # If already in memory, return it
        if self.hess_approx_out is not None:
            LOG.info("Checking for Hessian matrix in memory...")
            hess_approx: Optional[Tensor] = self.hess_approx_out.get("hess_approx", None)
            if hess_approx is not None:
                LOG.info("Hessian matrix found in memory, returning it directly.")
                return hess_approx.cpu()
        LOG.info("Hessian matrix not found in memory, loading from disk...")

        if data_path is None:
            raise ValueError("data_path must be provided to load Hessian from disk.")
        if not data_path.is_file():
            raise FileNotFoundError(f"Hessian matrix file not found at {data_path}.")

        # Auto-detect format if streaming is not specified
        if streaming is None:
            streaming = self._is_memmap_format(data_path)
            LOG.info(f"Auto-detected format: {'memmap (streaming)' if streaming else 'torch pickle'}")

        if not streaming:
            # Simple loading, iff matrix fits in memory
            start = time.time()
            LOG.info(f"Loading full Hessian matrix from {data_path}...")
            hess_approx: Tensor = torch.load(data_path, map_location="cpu", weights_only=False)
            LOG.info(f"Loaded Hessian matrix with shape {hess_approx.size()} from {data_path}"
                     f" in {time.time() - start:.2f} seconds.")
            return hess_approx

        # Load from the output path via streaming mode
        LOG.info(f"Loading streamed Hessian matrix from {data_path}...")
        
        # Read metadata if available (dtype + shape); otherwise infer
        meta = self._read_memmap_meta(data_path)
        mmap_dtype = np.dtype(meta["dtype"]) if meta is not None else np.float16
        if shape is None:
            if meta is not None and "shape" in meta:
                shape = tuple(meta["shape"])  # type: ignore
            else:
                shape = self._infer_memmap_shape(data_path, dtype=mmap_dtype)
            LOG.info(f"Memmap shape: {shape} | dtype: {mmap_dtype.name}")

        mmap = np.memmap(
            data_path, dtype=mmap_dtype, mode='r', shape=shape)

        if target_size is not None:
            stride = max(1, math.ceil(shape[0] / target_size))
            LOG.info(f"Loading Hessian with stride {stride} to limit size to ~{target_size}x{target_size}...")
            hess_approx = torch.from_numpy(mmap[::stride, ::stride].copy())  # small copy for plotting
        else:
            LOG.info(f"Loading full memmap matrix with shape {shape}...")
            hess_approx = torch.from_numpy(mmap[:].copy())
        return hess_approx

    def condition_number(
        self,
        data_path: Optional[Path] = None,
        shape: Optional[Tuple[int, int]] = None,
        target_size: Optional[int] = None,
        streaming: Optional[bool] = None,
        norm_ord: int = 2,
    ) -> float:
        """Compute the condition number of the Hessian matrix.

        Args:
            data_path: Optional path to the Hessian data on disk. If omitted,
                tries to use ``self.hess_approx_out`` metadata.
            shape: Shape hint for streamed (memmap) matrices.
            target_size: Optional stride-based downsampling target for huge matrices.
            streaming: Force memmap loading if True, pickle if False, auto-detect if None.
            norm_ord: Norm order for ``torch.linalg.cond`` (default spectral norm ``p=2``).

        Returns:
            The condition number as a float. Returns ``inf`` if the matrix is singular.
        """
        hess_data_path = data_path or self.output_path
        hess_approx = self.load_hessian(
            data_path=hess_data_path,
            shape=shape,
            target_size=target_size,
            streaming=streaming,
        ).to(dtype=torch.float32)

        if hess_approx.ndim != 2:
            raise ValueError(f"Expected a 2D Hessian matrix, got shape {hess_approx.shape}")

        try:
            cond_val = torch.linalg.cond(hess_approx, p=norm_ord).item()
        except Exception:
            # Fallback via singular values if torch.linalg.cond is unavailable or fails
            svals = torch.linalg.svdvals(hess_approx)
            s_min, s_max = torch.min(svals), torch.max(svals)
            cond_val = float("inf") if s_min == 0 else (s_max / s_min).item()

        LOG.info(f"Condition number (p={norm_ord}) of Hessian at {hess_data_path}: {cond_val:.4e}")
        return cond_val

    def fraction_positive_eigenvalues(
        self,
        data_path: Optional[Path] = None,
        shape: Optional[Tuple[int, int]] = None,
        target_size: Optional[int] = None,
        streaming: Optional[bool] = None,
    ) -> float:
        """Compute the fraction of positive eigenvalues from the Hessian matrix itself.

        This avoids relying on the pyhessian density helper (which may report per-iteration
        densities). For very large matrices, provide ``target_size`` to stride-load a
        downsampled version before eigen-decomposition.
        """
        hess_data_path = data_path or self.output_path
        hess_approx = self.load_hessian(
            data_path=hess_data_path,
            shape=shape,
            target_size=target_size,
            streaming=streaming,
        ).to(dtype=torch.float32)

        if hess_approx.ndim != 2:
            raise ValueError(f"Expected a 2D Hessian matrix, got shape {hess_approx.shape}")

        # Ensure symmetry for eigenvalue computation
        hess_sym = 0.5 * (hess_approx + hess_approx.transpose(0, 1))

        # Use symmetric eigvals; torch.linalg.eigvalsh is faster/stabler for Hermitian matrices
        eigvals = torch.linalg.eigvalsh(hess_sym.cpu())
        if eigvals.numel() == 0:
            return float("nan")

        pos = (eigvals > 0).sum().item()
        frac_pos = pos / eigvals.numel()
        LOG.info(
            f"Fraction of positive eigenvalues from matrix (path={hess_data_path}): {frac_pos:.4f}"
        )
        return frac_pos

    def _is_memmap_format(self, data_path: Path) -> bool:
        """Detect if the file is a memmap (raw binary) or torch pickle format.
        Torch files start with a specific magic number for ZIP archives.
        """
        try:
            with open(data_path, 'rb') as f:
                magic = f.read(2)
                # PyTorch files are ZIP archives, which start with 'PK'
                return magic != b'PK'
        except Exception as e:
            LOG.warning(f"Could not detect file format: {e}. Assuming memmap format.")
            return True

    def _read_memmap_meta(self, data_path: Path) -> Optional[Dict[str, Any]]:
        """Read sidecar metadata for memmap files if present."""
        try:
            meta_path = Path(str(data_path) + '.meta.json')
            if meta_path.is_file():
                with open(meta_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            LOG.warning(f"Failed to read memmap metadata: {e}")
        return None

    def _infer_memmap_shape(self, data_path: Path, dtype: np.dtype = np.float16) -> Tuple[int, int]:
        """Infer the shape of a square memmap file from its size and dtype."""
        file_size = data_path.stat().st_size
        itemsize = np.dtype(dtype).itemsize
        total_elements = file_size // itemsize
        n = int(math.sqrt(total_elements))
        
        if n * n != total_elements:
            raise ValueError(
                f"File size {file_size} bytes does not correspond to a square matrix. "
                f"Total elements: {total_elements}, sqrt: {n}")
        
        return (n, n)

    def export_matrix(self) -> None:
        """
        Export the computed Hessian matrix to a .pt file. This method will be only used after computation.

        Only the Hessian tensor (not the full dictionary) is saved using torch.save().
        To reload, use torch.load(path), which will return the tensor as stored in self.hess_approx_out['hess_approx'].
        If block-wise information or additional keys are needed, modify this method to save the full dictionary.
        """
        # No need for other arguments since we have the hessian in memory
        hess_approx = self.load_hessian()

        LOG.info(f"Exporting Hessian matrix to {self.output_path}...")
        LOG.info(f"  shape={hess_approx.size()}, dtype={hess_approx.dtype}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: dump the dict itself
        torch.save(hess_approx, self.output_path)
        file_size = self.output_path.stat().st_size / (1024 ** 2)  # size in MB
        LOG.info(f"Dumped Hessian matrix data at {self.output_path} (size: {file_size:.2f} MB)")

    def plot_hessian(self) -> Tuple[Axes, Axes, str]:
        """Plot the Hessian matrix by first loading it from disk.

        Returns:
            Tuple[Axes, Axes, str]: The axes for the Hessian heatmap and
            eigenvalue density plot, and the output path of the saved figure.
        """
        # Configs
        if self.plot_config is None:
            LOG.info("No plot config provided, using default settings.")
            self.plot_config = {}
        LOG.info(f"Plot config: {self.plot_config}")

        output_path = self.plot_config.get('output_path', f"./{self.model_ckpt}")
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        # Replace extension with .svg while preserving the parent directory
        # e.g. keep "some/dir/name.pt" -> "some/dir/name.svg"
        output_path = output_path.with_suffix('.svg')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load Hessian matrix from disk
        streaming = self.plot_config.get("streaming", None)  # None = auto-detect
        target_size = self.plot_config.get("target_size", None)
        shape = self.plot_config.get("shape", None)

        hess_data_path: Optional[Path] = self.plot_config.get("hess_data_path", None)
        hess_approx = self.load_hessian(
            data_path=hess_data_path,
            shape=shape,
            target_size=target_size,
            streaming=streaming,
        )
        # Ensure we operate in float32 on CPU for pooling/plotting stability
        hess_approx = hess_approx.to(dtype=torch.float32)

        # TODO: if block-wise, I should pool the blocks alone
        pooling = self.plot_config.get("pooling", True)
        if pooling:
            filter_size = self.plot_config.get("filter_size", 50)
            # Pooling is applied to reduce the resolution of the Hessian matrix for visualization.
            # Reshape Hessian to 4D for pooling: (N=1, C=1, H, W)
            hess_approx_4d = hess_approx.unsqueeze(0).unsqueeze(0)
            hess_approx_pooled = F.max_pool2d(hess_approx_4d, kernel_size=filter_size)
            hess_approx = hess_approx_pooled.squeeze(0).squeeze(0)  # Restore 2D
            LOG.info(f"Pooled Hessian to {hess_approx.size()}")

        # Create figure with appropriate size and width ratios
        axs: List[Axes]
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]}, 
            layout='constrained'
        )

        # Hessian matrix heatmap
        try:
            plot_hessian_heatmap(hess_approx, ax=axs[0])
            # TODO: improve
            axs[0].set_title(Path(self.model_ckpt).stem, fontsize=10, color="black", pad=8)
            # Custom subtitle with parameter names
            # param_names = [name for name, p in self.hessian_comp.model.named_parameters() if p.requires_grad]
            # subtitle = ", ".join(param_names)
            # axs[0].text(0.5, 0.99, subtitle, transform=axs[0].transAxes,
            #             ha="center", va="bottom", fontsize=6, color="gray")
            LOG.info("Added Hessian heatmap")
        except Exception as e:
            import traceback
            LOG.info(e)
            traceback.print_exc()

        plt.savefig(output_path)
        LOG.info(f"Partial plot available at {output_path}")

        # Hessian eigen-spectrum
        try:
            # TODO: for some reasons is slow
            density_eigen, density_weight = self.hessian_comp.density()
            plot_eigenvalue_density(density_eigen, density_weight, ax=axs[1], eigen_abs=True)

            frac_pos = self.fraction_positive_eigenvalues(
                data_path=hess_data_path or self.output_path,
                shape=shape,
                target_size=target_size,
                streaming=streaming,
            )
            cond_val = self.condition_number(
                data_path=hess_data_path or self.output_path,
                shape=shape,
                target_size=target_size,
                streaming=streaming,
            )

            y_offset = 0.8
            axs[1].annotate(
                f'p(lambd>0)={frac_pos:.2f}', xy=(0.05, y_offset + 0.06),
                xycoords='axes fraction', fontsize=10, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
            )
            axs[1].annotate(
                f'cond={cond_val:.2e}', xy=(0.05, y_offset), xycoords='axes fraction',
                fontsize=10, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
            )

            LOG.info("Added density plot")
        except Exception as e:
            import traceback
            LOG.info(e)
            traceback.print_exc()

        plt.savefig(output_path)
        LOG.info(f"Plot of the computed Hessian and its density available at {output_path}")
        return axs[0], axs[1], output_path


class HessianWriter:
    """Helper class for dumping the Hessian matrix in a memory-efficient way.
    Allows to stream Hessian matrix columns to disk without keeping
    them in RAM. Uses `numpy.memmap` to append columns incrementally."""

    def __init__(self, n_rows: int, n_cols: int, output_path: Path, dtype=np.float16):
        self.n_rows = n_rows  # number of rows in the Hessian matrix
        self.n_cols = n_cols  # number of columns in the Hessian matrix
        self.output_path = output_path
        self.dtype = np.dtype(dtype)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a memory-mapped file to store the Hessian matrix
        self.hess_memmap = np.memmap(
            output_path, dtype=self.dtype, mode='w+',
            shape=(n_rows, n_cols)
        )
        self.current_col = 0

        # Write sidecar metadata to aid robust loading later
        try:
            meta = {
                "shape": [n_rows, n_cols],
                "dtype": self.dtype.name,
            }
            meta_path = Path(str(self.output_path) + '.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception as e:
            LOG.warning(f"Could not write memmap metadata: {e}")

    def write_col(self, col_data: Tensor) -> None:
        """Write a column of the Hessian matrix to the memmap file."""
        if self.current_col >= self.n_cols:
            raise IndexError("All columns have already been written to the Hessian matrix.")

        if col_data.numel() != self.n_rows:
            raise ValueError(f"Column data size {col_data.numel()} does not match expected number of rows {self.n_rows}.")

        # Convert Tensor to numpy array and write to memmap
        LOG.debug(f"Writing column {self.current_col} to Hessian memmap...")
        # Map numpy dtype to torch dtype for safe casting
        torch_dtype = torch.float16 if self.dtype == np.float16 else (
            torch.float32 if self.dtype == np.float32 else torch.float32
        )
        col_data_np = col_data.detach().cpu().to(dtype=torch_dtype).numpy()
        self.hess_memmap[:, self.current_col] = col_data_np
        self.current_col += 1

        if self.current_col % 200 == 0:
            LOG.info(f"Wrote column {self.current_col}/{self.n_cols} to Hessian memmap.")

    def flush(self) -> None:
        """Flush (write) changes to disk."""
        self.hess_memmap.flush()
        LOG.info("Flushed Hessian memmap to disk.")

    def close(self) -> None:
        """Close the memmap file."""
        del self.hess_memmap
        LOG.info("Closed Hessian memmap.")
