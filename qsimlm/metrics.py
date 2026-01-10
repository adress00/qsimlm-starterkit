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
