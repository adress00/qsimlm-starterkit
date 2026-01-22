
import numpy as np
import torch
from qsimlm.data import make_dataset, sample_2q_special_params, density_matrix_target, statevector_target, get_noise_model
from qsimlm.metrics import density_matrix_fidelity

def inspect_noisy_data():
    print("Generating noisy dataset (n=100)...")
    X, Y = make_dataset(100, seed=42, noisy=True, noise_prob=0.01)
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # 1. Check if Y contains valid density matrices (trace=1)
    Y_tensor = torch.tensor(Y)
    B = Y.shape[0]
    rho = torch.complex(Y_tensor[:, :16], Y_tensor[:, 16:]).reshape(B, 4, 4)
    
    traces = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Average Trace (Real): {traces.real.mean():.6f}")
    print(f"Average Trace (Imag): {traces.imag.mean():.6f}")
    
    # 2. Check Variance
    mean_std = Y.std(axis=0).mean()
    print(f"Average Standard Deviation across 32 dimensions: {mean_std:.6f}")
    
    # 3. Check Fidelity with Identity/4
    target_I = np.zeros((B, 32), dtype=np.float32)
    target_I[:, 0] = 0.25 # Re(0,0)
    target_I[:, 5] = 0.25 # Re(1,1) -> index 5
    target_I[:, 10] = 0.25 # Re(2,2) -> index 10
    target_I[:, 15] = 0.25 # Re(3,3) -> index 15
    
    target_I_tensor = torch.tensor(target_I)
    fid_with_mixed = density_matrix_fidelity(Y_tensor, target_I_tensor).mean().item()
    print(f"Average Fidelity with Maximally Mixed State (I/4): {fid_with_mixed:.4f}")
    
    # 4. Check Fidelity with ITSELF
    fid_self = density_matrix_fidelity(Y_tensor, Y_tensor).mean().item()
    print(f"Average Self-Fidelity (should be 1.0): {fid_self:.4f}")

    # 5. Compare Noisy vs Noiseless Targets
    print("\nGenerating paired data (Noisy vs Noiseless)...")
    rng = np.random.default_rng(42)
    noise_model = get_noise_model(0.01)
    
    fids = []
    for _ in range(100):
        theta = sample_2q_special_params(rng)
        
        # Get Noiseless Density Matrix
        y_sv = statevector_target(theta)
        psi = y_sv[:4] + 1j * y_sv[4:]
        rho_pure = np.outer(psi, psi.conj())
        rho_pure_flat = rho_pure.flatten()
        y_pure_dm = np.concatenate([rho_pure_flat.real, rho_pure_flat.imag])
        
        # Get Noisy Density Matrix
        y_noisy_dm = density_matrix_target(theta, noise_model)
        
        t_pure = torch.tensor(y_pure_dm, dtype=torch.float32).unsqueeze(0)
        t_noisy = torch.tensor(y_noisy_dm, dtype=torch.float32).unsqueeze(0)
        f = density_matrix_fidelity(t_pure, t_noisy).item()
        fids.append(f)
        
    print(f"Average Fidelity (Noiseless vs Noisy Target): {np.mean(fids):.4f}")
    if np.mean(fids) > 0.9:
        print("Verdict: Noise is small. The target IS learnable.")
    else:
        print("Verdict: Noise is LARGE. Information might be lost.")

if __name__ == "__main__":
    inspect_noisy_data()
