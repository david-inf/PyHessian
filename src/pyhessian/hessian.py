"""Core of the PyHessian package."""

from typing import Optional, Tuple, List
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import numpy as np

from .utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class Hessian:
    """
    Compute Hessian-related information for neural networks.
    
    This class provides methods to compute:
        i) the top 1 (n) eigenvalue(s) of the neural network Hessian
        ii) the trace of the Hessian matrix
        iii) the estimated eigenvalue density using stochastic Lanczos quadrature
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data: Optional[Tuple[Tensor, Tensor]] = None,
        dataloader: Optional[DataLoader] = None,
        cuda: bool = True
    ) -> None:
        """
        Initialize the Hessian computation class.
        
        Args:
            model: The neural network model for which to compute Hessian information
            criterion: The loss function (e.g., nn.CrossEntropyLoss())
            data: A single batch of data as a tuple (inputs, targets). Mutually exclusive with dataloader
            dataloader: A DataLoader containing multiple batches. Mutually exclusive with data
            cuda: Whether to use CUDA for computation if available
            
        Raises:
            AssertionError: If both data and dataloader are provided, or if neither is provided
        """

        # Ensure we either pass a single batch or a dataloader (mutually exclusive)
        assert (data is not None and dataloader is None) or (
            data is None and dataloader is not None
        ), "Must provide either 'data' or 'dataloader', but not both"

        # Set model to evaluation mode (disables dropout, batch norm training mode, etc.)
        self.model = model.eval()
        self.criterion = criterion

        # Store data source and set flag for computation method
        if data is not None:
            self.data = data
            self.full_dataset = False  # Using single batch
        else:
            self.data = dataloader
            self.full_dataset = True  # Using full dataloader

        # Set computation device
        self.device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'

        # Pre-processing for single batch case to simplify and optimize computation
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            # Move data to GPU if using CUDA
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # For single batch, compute gradients once and reuse them
            # create_graph=True allows computing second-order derivatives
            outputs = self.model(self.inputs)
            loss: Tensor = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # Extract model parameters and their gradients for Hessian computation
        params, gradsH = get_params_grad(self.model)  # a for loop over params
        self.params = params  # Model parameters (weights and biases)
        self.gradsH = gradsH  # Gradients used for Hessian-vector products

    def dataloader_hv_product(self, v: List[torch.Tensor]) -> Tuple[float, List[torch.Tensor]]:
        """
        Compute Hessian-vector product averaged over all batches in the dataloader.
        
        This method iterates through all batches in the dataloader, computes the Hessian-vector
        product for each batch, and returns the weighted average (weighted by batch size).
        
        Args:
            v: A list of tensors (same structure as model parameters) representing the vector
               to multiply with the Hessian
               
        Returns:
            A tuple containing:
                - eigenvalue: The eigenvalue estimate (v^T H v)
                - THv: The accumulated Hessian-vector product (H*v) averaged over all data
        """
        num_data = 0  # Accumulator for total number of data points processed

        # Initialize accumulator for Hessian-vector products (one tensor per parameter)
        THv = [torch.zeros(p.size()).to(self.device) for p in self.params]

        # Iterate through all batches in the dataloader
        for inputs, targets in self.data:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)  # batch size

            # Forward pass and loss computation
            outputs = self.model(inputs)
            loss: Tensor = self.criterion(outputs, targets)
            loss.backward(create_graph=True)  # Need graph for second-order derivatives

            # Get current parameters and gradients
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            
            # Compute Hessian-vector product using autograd
            # This computes ∇(∇L^T v) = H*v where H is the Hessian
            Hv = torch.autograd.grad(
                gradsH, params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=False  # No need to retain graph after this
            )

            # Accumulate weighted Hessian-vector product (weighted by batch size)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        # Normalize by total number of data points to get average
        THv = [THv1 / float(num_data) for THv1 in THv]
        
        # Compute eigenvalue estimate: v^T * H * v
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(
        self, 
        maxIter: int = 100, 
        tol: float = 1e-3, 
        top_n: int = 1
    ) -> Tuple[List[float], List[List[torch.Tensor]]]:
        """
        Compute the top-n eigenvalues of the Hessian using the power iteration method.
        
        This method uses power iteration to iteratively compute the largest eigenvalues
        and their corresponding eigenvectors. Each eigenvalue is computed by finding the
        dominant eigenvector, then subsequent eigenvalues are found by orthogonalizing
        against previously found eigenvectors.
        
        Args:
            maxIter: Maximum number of iterations for computing each eigenvalue
            tol: Relative tolerance for convergence. Stops when |λ_new - λ_old| / (|λ_old| + 1e-6) < tol
            top_n: Number of top eigenvalues to compute (must be >= 1)
            
        Returns:
            A tuple containing:
                - eigenvalues: List of the top-n eigenvalues (sorted from largest to smallest)
                - eigenvectors: List of corresponding eigenvectors (each eigenvector is a list of tensors)
        """

        assert top_n >= 1, "top_n must be at least 1"

        device = self.device

        eigenvalues: List[float] = []
        eigenvectors: List[List[torch.Tensor]] = []

        computed_dim = 0

        # Compute eigenvalues one by one
        while computed_dim < top_n:
            eigenvalue = None
            
            # Initialize with a random vector matching the parameter structure
            v = [torch.randn(p.size()).to(device) for p in self.params]
            v = normalization(v)  # Normalize to unit length

            # Power iteration loop
            for i in range(maxIter):
                # Orthogonalize against previously computed eigenvectors
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                # Compute Hessian-vector product: Hv = H * v
                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                # Normalize the result to get next iteration's vector
                v = normalization(Hv)

                # Check for convergence
                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    # Compute relative change in eigenvalue
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
                        break  # Converged
                    else:
                        eigenvalue = tmp_eigenvalue

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter: int = 100, tol: float = 1e-3) -> List[float]:
        """
        Compute the trace of the Hessian matrix using Hutchinson's stochastic method.
        
        Hutchinson's trace estimator uses the fact that:
        tr(H) = E[v^T H v] where v is a random vector with entries ±1 (Rademacher distribution)
        
        This method computes multiple samples and averages them until convergence.
        
        Args:
            maxIter: Maximum number of Hutchinson iterations (samples)
            tol: Relative tolerance for convergence of the trace estimate
            
        Returns:
            A list of trace estimates from each iteration (useful for monitoring convergence)
        """

        device = self.device
        trace_vhv: List[float] = []  # Store v^T H v for each iteration
        trace = 0.  # Running average of trace estimate

        for i in range(maxIter):
            self.model.zero_grad()
            
            # Generate Rademacher random variables (uniformly ±1)
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # Convert {0, 1} to {-1, 1}
            for v_i in v:
                v_i[v_i == 0] = -1

            # Compute Hessian-vector product
            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
                
            # Compute v^T H v (unbiased trace estimator)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            
            # Check for convergence based on mean of all samples so far
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv  # Converged
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(
        self, 
        iter: int = 100, 
        n_v: int = 1
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Compute the estimated eigenvalue density using Stochastic Lanczos Quadrature (SLQ).
        
        This method approximates the spectral density (distribution of eigenvalues) of the
        Hessian matrix without computing all eigenvalues explicitly. It uses the Lanczos
        algorithm to build a tridiagonal approximation of the Hessian, then computes its
        eigenvalues as a proxy for the true spectrum.
        
        The algorithm:
        1. Start with a random Rademacher vector
        2. Run Lanczos iterations to build a tridiagonal matrix T that approximates H
        3. Compute eigenvalues and eigenvectors of T
        4. Use the first component of eigenvectors as weights for the density
        5. Repeat n_v times with different random vectors and aggregate results
        
        Args:
            iter: Number of Lanczos iterations (determines resolution of density estimate)
            n_v: Number of independent SLQ runs to average over (higher = more accurate)
            
        Returns:
            A tuple containing:
                - eigen_list_full: List of lists, where each inner list contains eigenvalues 
                  from one SLQ run
                - weight_list_full: List of lists, where each inner list contains corresponding
                  weights (squared first eigenvector components) for the density estimate
        """
        eigen_list_full: List[List[float]] = []
        weight_list_full: List[List[float]] = []

        # Run multiple independent Lanczos processes
        for k in range(n_v):
            # Generate initial random Rademacher vector (uniformly ±1)
            v = [
                torch.randint_like(p, high=2, device=self.device)
                for p in self.params
            ]
            # Convert {0, 1} to {-1, 1}
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # Initialize Lanczos algorithm data structures
            v_list = [v]  # Orthonormal basis vectors
            w_list = []  # Working vectors
            alpha_list = []  # Diagonal elements of tridiagonal matrix
            beta_list = []  # Off-diagonal elements of tridiagonal matrix

            ############### Lanczos Algorithm ###############
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(self.device) for p in self.params]

                if i == 0:
                    # First iteration: compute H*v
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    # Diagonal element: α_i = v^T H v
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    # w = H*v - α*v
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    # Subsequent iterations: maintain orthogonality
                    # β_i = ||w||
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())

                    if beta_list[-1] != 0.:
                        # Normalize w and re-orthogonalize to maintain numerical stability
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # If beta is zero, generate a new random vector
                        w = [torch.randn(p.size()).to(self.device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)

                    # Compute H*v for the new basis vector
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    # Diagonal element
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    # Three-term recurrence: w = H*v - α*v - β*v_{i-1}
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            # Build tridiagonal matrix T from Lanczos coefficients
            T = torch.zeros(iter, iter).to(self.device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]  # Diagonal
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]  # Sub-diagonal
                    T[i, i + 1] = beta_list[i]  # Super-diagonal

            # Compute eigendecomposition of tridiagonal matrix
            eigenvalues: Tensor
            eigenvectors: Tensor
            eigenvalues, eigenvectors = torch.linalg.eig(T)

            # Extract real parts (imaginary parts should be ~0 for symmetric matrix)
            eigen_list: Tensor = eigenvalues.real
            # Weights are squared first components of eigenvectors (for density estimation)
            weight_list: Tensor = torch.pow(eigenvectors[0, :], 2)

            # Store results from this SLQ run
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
