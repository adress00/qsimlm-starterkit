from __future__ import annotations
import numpy as np
import torch
from qiskit.quantum_info import state_fidelity, DensityMatrix

def state_fidelity_from_realimag(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    y: (B, 8) = [Re(psi0..psi3), Im(psi0..psi3)]
    fidelity = |<psi_true|psi_pred>|^2
    """
    re_t, im_t = y_true[:, :4], y_true[:, 4:]
    re_p, im_p = y_pred[:, :4], y_pred[:, 4:]

    psi_t = torch.complex(re_t, im_t)
    psi_p = torch.complex(re_p, im_p)

    psi_t = psi_t / (torch.linalg.vector_norm(psi_t, dim=1, keepdim=True) + eps)
    psi_p = psi_p / (torch.linalg.vector_norm(psi_p, dim=1, keepdim=True) + eps)

    overlap = (psi_t.conj() * psi_p).sum(dim=1)
    return (overlap.abs() ** 2).real


def density_matrix_fidelity(
    dm_true: np.ndarray | torch.Tensor,
    dm_pred: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute quantum state fidelity between two density matrices using Qiskit.
    
    Fidelity is defined as: F(rho, sigma) = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2
    
    Args:
        dm_true: (D, D) complex array - true/reference density matrix
        dm_pred: (D, D) complex array - predicted density matrix
    
    Returns:
        float in [0, 1] - fidelity value
    """
    # Convert to numpy if needed
    if isinstance(dm_true, torch.Tensor):
        dm_true = dm_true.detach().cpu().numpy()
    if isinstance(dm_pred, torch.Tensor):
        dm_pred = dm_pred.detach().cpu().numpy()
    
    # Ensure complex type
    dm_true = np.asarray(dm_true, dtype=complex)
    dm_pred = np.asarray(dm_pred, dtype=complex)
    
    # Normalize trace if necessary (for numerical stability)
    tr_true = np.trace(dm_true)
    tr_pred = np.trace(dm_pred)
    
    if abs(tr_true) > 1e-10:
        dm_true = dm_true / tr_true
    if abs(tr_pred) > 1e-10:
        dm_pred = dm_pred / tr_pred
    
    # Use Qiskit's state_fidelity function
    try:
        # Create DensityMatrix objects
        rho_true = DensityMatrix(dm_true)
        rho_pred = DensityMatrix(dm_pred)
        
        # Compute fidelity
        fid = state_fidelity(rho_true, rho_pred)
        fid = float(np.real(fid))  # Ensure real and scalar
        
        return fid
    except Exception as e:
        print(f"Error in state_fidelity computation: {e}")
        # Fallback: simple trace distance
        diff = dm_true - dm_pred
        trace_dist = 0.5 * np.trace(np.linalg.sqrtm(diff.conj().T @ diff))
        return float(1.0 - trace_dist)


def batch_density_matrix_fidelity(
    dm_true: np.ndarray | torch.Tensor,
    dm_pred: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    Compute fidelity for a batch of density matrices.
    
    Args:
        dm_true: (B, D, D) complex array - batch of true density matrices
        dm_pred: (B, D, D) complex array - batch of predicted density matrices
    
    Returns:
        (B,) float array - fidelity values for each sample
    """
    # Convert to numpy if needed
    if isinstance(dm_true, torch.Tensor):
        dm_true = dm_true.detach().cpu().numpy()
    if isinstance(dm_pred, torch.Tensor):
        dm_pred = dm_pred.detach().cpu().numpy()
    
    batch_size = dm_true.shape[0]
    fidelities = np.zeros(batch_size)
    
    for i in range(batch_size):
        fidelities[i] = density_matrix_fidelity(dm_true[i], dm_pred[i])
    
    return fidelities
