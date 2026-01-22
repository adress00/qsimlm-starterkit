from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def get_noise_model(prob: float) -> NoiseModel:
    """Create a depolarizing noise model with given probability."""
    noise_model = NoiseModel()
    # Add 1-qubit error
    error_1 = depolarizing_error(prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['u', 'u1', 'u2', 'u3'])
    # Add 2-qubit error
    error_2 = depolarizing_error(prob, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def sample_2q_special_params(rng: np.random.Generator) -> np.ndarray:
    """Sample 12 angles for the 2-qubit special circuit, in [-pi, pi]."""
    return rng.uniform(-np.pi, np.pi, size=(12,)).astype(np.float32)

def build_2q_special_circuit(theta: np.ndarray) -> QuantumCircuit:
    """
    2-qubit special circuit (Zhou et al.):
      U(theta1..3) on q0
      U(theta4..6) on q1
      CNOT q0->q1
      U(theta7..9) on q0
      U(theta10..12) on q1
    """
    if theta.shape != (12,):
        raise ValueError(f"theta must have shape (12,), got {theta.shape}")
    qc = QuantumCircuit(2)
    qc.u(float(theta[0]), float(theta[1]), float(theta[2]), 0)
    qc.u(float(theta[3]), float(theta[4]), float(theta[5]), 1)
    qc.cx(0, 1)
    qc.u(float(theta[6]), float(theta[7]), float(theta[8]), 0)
    qc.u(float(theta[9]), float(theta[10]), float(theta[11]), 1)
    return qc

def statevector_target(theta: np.ndarray) -> np.ndarray:
    """
    Target: 2-qubit statevector (4 complex amps) -> 8 floats [Re(4), Im(4)].
    """
    qc = build_2q_special_circuit(theta)
    sv = Statevector.from_instruction(qc).data  # complex (4,)
    y = np.concatenate([sv.real, sv.imag], axis=0).astype(np.float32)  # (8,)
    return y

    return y

def density_matrix_target(theta: np.ndarray, noise_model: NoiseModel) -> np.ndarray:
    """
    Target: 2-qubit density matrix (4x4 complex) -> 32 floats [Re(flat), Im(flat)].
    Uses AerSimulator to simulate the circuit with noise.
    """
    qc = build_2q_special_circuit(theta)
    qc.save_density_matrix()
    
    backend = AerSimulator(noise_model=noise_model)
    # Transpilation is implicitly handled or not needed for basic gates in Aer, 
    # but for safety and noise model matching, we rely on Aer's default basis gates.
    # We run 1 shot just to get the final density matrix state (since we use save_density_matrix)
    # Actually, save_density_matrix works with creating a snapshot. 
    # But AerSimulator().run(qc) returns a result containing the density matrix.
    
    # Efficient way: use density_matrix method directly if available or use DensityMatrix class?
    # Qiskit's DensityMatrix class does NOT support noise models directly in construction.
    # We must use the backend.
    
    job = backend.run(qc, shots=1)
    result = job.result()
    dm = result.data()['density_matrix'] # This is a DensityMatrix object or complex array
    dm_data = np.asarray(dm) # (4, 4) complex
    
    # Flatten: 16 complex numbers
    dm_flat = dm_data.flatten()
    
    # 16 Re + 16 Im = 32 floats
    y = np.concatenate([dm_flat.real, dm_flat.imag], axis=0).astype(np.float32)
    return y

def make_dataset(n: int, seed: int = 0, noisy: bool = False, noise_prob: float = 0.01):
    """
    Returns:
      X: (n, 4, 3)  four U-gates, each has 3 angles
      Y: (n, 8) or (n, 32)
         If noisy=False: 8 floats (statevector)
         If noisy=True: 32 floats (density matrix)
    """
    rng = np.random.default_rng(seed)
    
    out_dim = 32 if noisy else 8
    X = np.zeros((n, 4, 3), dtype=np.float32)
    Y = np.zeros((n, out_dim), dtype=np.float32)

    noise_model = None
    if noisy:
        noise_model = get_noise_model(noise_prob)

    for i in range(n):
        theta = sample_2q_special_params(rng)
        X[i, 0, :] = theta[0:3]
        X[i, 1, :] = theta[3:6]
        X[i, 2, :] = theta[6:9]
        X[i, 3, :] = theta[9:12]
        
        if noisy:
            Y[i, :] = density_matrix_target(theta, noise_model)
        else:
            Y[i, :] = statevector_target(theta)
    return X, Y
