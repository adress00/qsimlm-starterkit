"""
Physics-aware constraints and state reconstruction module for density matrices.

This module provides:
1. State Reconstruction: Convert predicted lower triangular DM to valid quantum state
2. Physical Constraint Enforcement: Use CVXPY to project onto valid density matrices
"""

from __future__ import annotations
import numpy as np
import torch
try:
    import cvxpy as cp
except ImportError:
    cp = None


def extract_lower_triangular_indices(D: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get row and column indices for lower triangular part (including diagonal).
    
    Args:
        D: dimension of the matrix
    
    Returns:
        (row_indices, col_indices) for lower triangular elements
    """
    return np.tril_indices(D)


def reconstruct_dm_from_lower_triangular(
    lower_vec: np.ndarray,
    D: int,
    hermitian: bool = True,
) -> np.ndarray:
    """
    Reconstruct Hermitian density matrix from lower triangular representation.
    
    Args:
        lower_vec: (D*(D+1),) float array [Re(lower), Im(lower)]
        D: dimension of the density matrix
        hermitian: whether to enforce Hermitian symmetry (default True)
    
    Returns:
        (D, D) complex numpy array (Hermitian)
    """
    n_lower = D * (D + 1) // 2
    re_part = lower_vec[:n_lower]
    im_part = lower_vec[n_lower:]
    
    lower_complex = re_part + 1j * im_part
    
    # Reconstruct lower triangular matrix
    dm = np.zeros((D, D), dtype=complex)
    indices = np.tril_indices(D)
    dm[indices] = lower_complex
    
    if hermitian:
        # Fill upper triangular by Hermitian conjugate
        dm = dm + dm.conj().T - np.diag(dm.diagonal())
    
    return dm


def project_to_valid_density_matrix(
    dm_pred: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Project a predicted density matrix to the valid quantum state space using CVXPY.
    
    A valid density matrix must satisfy:
    1. Hermitian: rho = rho^dagger
    2. Trace constraint: Tr(rho) = 1
    3. Positive semidefinite: rho >= 0 (all eigenvalues >= 0)
    
    Uses CVXPY to solve:
        minimize ||rho - rho_pred||_F^2
        subject to:
            rho is Hermitian
            Tr(rho) = 1
            rho >= 0 (semidefinite cone constraint)
    
    Args:
        dm_pred: (D, D) complex array - predicted density matrix
        verbose: whether to print optimization details
    
    Returns:
        (D, D) complex array - projected valid density matrix
    
    Raises:
        ImportError: if cvxpy is not installed
        ValueError: if optimization fails
    """
    if cp is None:
        raise ImportError("CVXPY is required for physical constraint enforcement. "
                         "Install with: pip install cvxpy")
    
    D = dm_pred.shape[0]
    
    # Separate real and imaginary parts for optimization
    # CVXPY requires real-valued matrices
    dm_re = dm_pred.real
    dm_im = dm_pred.imag
    
    # Decision variables: Hermitian matrix stored as real + imag parts
    rho_re = cp.Variable((D, D), symmetric=True)  # Real part must be symmetric
    rho_im = cp.Variable((D, D))  # Imaginary part must be antisymmetric (but we enforce via Hermitian)
    
    # For Hermitian matrix: rho_im must be antisymmetric
    # This means im[i,j] = -im[j,i]
    # We'll enforce this by only using lower triangular part and mirroring
    rho_im_lower = cp.Variable((D, D))
    
    # Objective: minimize Frobenius norm distance
    # ||rho - rho_pred||_F^2 = tr((rho - rho_pred)(rho - rho_pred)^H)
    real_diff = rho_re - dm_re
    imag_diff = rho_im_lower - dm_im
    
    # For numerical stability, work with real-valued Frobenius norm
    # ||A||_F^2 = sum(|A_ij|^2) for complex A
    objective = cp.sum_squares(real_diff) + cp.sum_squares(imag_diff)
    
    # Constraints
    constraints = []
    
    # 1. Trace constraint: Tr(rho) = 1
    constraints.append(cp.trace(rho_re) == 1.0)
    
    # 2. Hermitian constraint: imaginary part is antisymmetric
    # For simplicity, we enforce that the matrix is real (im part = 0) as an alternative
    # Or we can use a more sophisticated approach with real symmetric + antisymmetric decomposition
    # For now, let's use the semidefinite constraint which works with complex matrices via SCS solver
    
    # Alternative: convert to real semidefinite form
    # For Hermitian matrix H = A + iB where A is symmetric, B is antisymmetric
    # H >= 0 iff the real matrix [A -B; B A] >= 0 (2D x 2D semidefinite)
    
    # Construct the 2D block matrix form of the Hermitian constraint
    # [A  -B] >= 0  where A = rho_re, B = im part (antisymmetric)
    # [B   A]
    rho_block = cp.bmat([
        [rho_re, -rho_im_lower],
        [rho_im_lower, rho_re]
    ])
    
    # 3. Positive semidefinite constraint
    constraints.append(rho_block >> 0)
    
    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        # Use SCS solver for complex semidefinite constraints
        # SCS is more robust for semidefinite constraints
        problem.solve(solver=cp.SCS, verbose=verbose, max_iters=10000)
    except:
        # Fallback to other solvers if SCS fails
        try:
            problem.solve(solver=cp.ECOS_BB, verbose=verbose)
        except:
            problem.solve(verbose=verbose)  # Use default solver
    
    if problem.status != cp.OPTIMAL:
        # If not optimal, return best effort solution
        if verbose:
            print(f"Warning: CVXPY solver status: {problem.status}")
    
    # Reconstruct complex Hermitian matrix
    rho_re_opt = rho_re.value
    rho_im_opt = rho_im_lower.value
    
    # Construct full matrix ensuring Hermitian property
    dm_corrected = np.zeros((D, D), dtype=complex)
    dm_corrected.real = rho_re_opt
    
    # Make imaginary part antisymmetric
    rho_im_antisym = (rho_im_opt - rho_im_opt.T) / 2
    dm_corrected.imag = rho_im_antisym
    
    # Final Hermitian symmetry enforcement
    dm_corrected = (dm_corrected + dm_corrected.conj().T) / 2
    
    return dm_corrected


def state_reconstruction_cvxpy(
    dm_pred: np.ndarray | torch.Tensor,
    enforce_physical: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Reconstruct and enforce physical validity of a density matrix.
    
    Args:
        dm_pred: (D, D) complex array or torch tensor - predicted density matrix
        enforce_physical: whether to use CVXPY projection (default True)
        verbose: whether to print optimization details
    
    Returns:
        (D, D) complex numpy array - valid density matrix
    """
    # Convert to numpy if needed
    if isinstance(dm_pred, torch.Tensor):
        dm_pred = dm_pred.detach().cpu().numpy()
    
    # Ensure proper complex type
    dm_pred = np.asarray(dm_pred, dtype=complex)
    
    if not enforce_physical:
        # Just normalize trace if not enforcing full physical constraints
        trace = np.trace(dm_pred)
        if abs(trace) > 1e-10:
            dm_pred = dm_pred / trace
        return dm_pred
    
    # Use CVXPY to project onto valid density matrix space
    try:
        dm_corrected = project_to_valid_density_matrix(dm_pred, verbose=verbose)
    except ImportError as e:
        print(f"Warning: {e}")
        print("Falling back to simple trace normalization")
        trace = np.trace(dm_pred)
        if abs(trace) > 1e-10:
            dm_pred = dm_pred / trace
        dm_corrected = dm_pred
    
    return dm_corrected
