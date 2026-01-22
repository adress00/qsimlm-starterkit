from __future__ import annotations
import torch

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

def density_matrix_fidelity(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    y_true, y_pred: (B, 32) flattened density matrices [Re(16), Im(16)].
    Fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2.
    Since we are in PyTorch and might not want to do full matrix sqrt, 
    we can check if there's a simpler form. But generally for mixed states this is it.
    
    Warning: This function does SVD/Eigendecomposition which is expensive on GPU.
    Reference: qiskit.quantum_info.state_fidelity
    """
    B = y_true.size(0)
    
    # Reconstruct (B, 4, 4) complex density matrices
    rho = torch.complex(y_true[:, :16], y_true[:, 16:]).reshape(B, 4, 4)
    sigma = torch.complex(y_pred[:, :16], y_pred[:, 16:]).reshape(B, 4, 4)
    
    # To compute Fidelity, we need sqrt of rho. 
    # Since rho is Hermitian PSD, we can use eigh.
    # rho = U D U^dag -> sqrt(rho) = U sqrt(D) U^dag
    L_rho, U_rho = torch.linalg.eigh(rho)
    
    # Clip negative eigenvalues for numerical stability
    L_rho = torch.clamp(L_rho, min=0.0)
    sqrt_L_rho = torch.sqrt(L_rho)
    
    # Construct sqrt(rho)
    # diag_embed creates a diagonal matrix from the vector of eigenvalues
    # Cast to complex to match U_rho for matmul
    sqrt_L_rho_c = sqrt_L_rho.to(U_rho.dtype)
    sqrt_rho = U_rho @ torch.diag_embed(sqrt_L_rho_c) @ U_rho.mH
    
    # Compute product: sqrt(rho) * sigma * sqrt(rho)
    temp = sqrt_rho @ sigma @ sqrt_rho
    
    # Now we need sqrt of this product
    L_temp, U_temp = torch.linalg.eigh(temp)
    L_temp = torch.clamp(L_temp, min=0.0)
    sqrt_temp = torch.sqrt(L_temp)
    
    # Trace of the result
    tr_val = sqrt_temp.sum(dim=-1) # Sum of eigenvalues is the trace
    
    # Fidelity is square of trace
    fid = (tr_val ** 2).real
    
    return fid
