from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

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

def make_dataset(n: int, seed: int = 0):
    """
    Returns:
      X: (n, 4, 3)  four U-gates, each has 3 angles
      Y: (n, 8)     Re/Im of statevector
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4, 3), dtype=np.float32)
    Y = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        theta = sample_2q_special_params(rng)
        X[i, 0, :] = theta[0:3]
        X[i, 1, :] = theta[3:6]
        X[i, 2, :] = theta[6:9]
        X[i, 3, :] = theta[9:12]
        Y[i, :] = statevector_target(theta)
    return X, Y

def make_dataset_noisy(
    n: int,
    seed: int = 0,
    sigma: float = 0.01,
    noise_type: str = "gaussian",
):
    """
    Create a noisy dataset by adding measurement-like noise to the target.

    This function keeps the circuit generation identical to `make_dataset` and
    perturbs the target statevector components. The noise is applied on the
    real-imag concatenated vector of length 8, then consumers can re-normalize
    if needed (our fidelity metric normalizes internally).

    Args:
        n: number of samples
        seed: RNG seed for reproducibility
        sigma: standard deviation for noise
        noise_type: currently only 'gaussian' is supported

    Returns:
        X: (n, 4, 3) gate parameters
        Y_noisy: (n, 8) noisy real/imag statevector components
    """
    if noise_type != "gaussian":
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4, 3), dtype=np.float32)
    Y_noisy = np.zeros((n, 8), dtype=np.float32)

    for i in range(n):
        theta = sample_2q_special_params(rng)
        # Pack parameters into sequence of 4 gates with 3 angles each
        X[i, 0, :] = theta[0:3]
        X[i, 1, :] = theta[3:6]
        X[i, 2, :] = theta[6:9]
        X[i, 3, :] = theta[9:12]

        # Ideal target
        y = statevector_target(theta)

        # Add Gaussian noise to simulate measurement/estimation imperfections
        noise = rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
        Y_noisy[i, :] = y + noise

    return X, Y_noisy


def extract_lower_triangular(dm_array: np.ndarray) -> np.ndarray:
    """
    Extract lower triangular part (including diagonal) of a Hermitian density matrix.
    For a DxD matrix, this gives D*(D+1)/2 complex numbers -> D*(D+1) real numbers (real + imag).
    
    Args:
        dm_array: (D, D) complex numpy array (Hermitian)
    
    Returns:
        (D*(D+1),) float32 array with [Re(lower), Im(lower)]
    """
    D = dm_array.shape[0]
    lower_triangle = np.tril(dm_array)  # (D, D) complex
    # Flatten and separate real/imag
    lower_vec = lower_triangle[np.tril_indices(D)]  # D*(D+1)/2 complex values
    result = np.concatenate([lower_vec.real, lower_vec.imag], axis=0).astype(np.float32)
    return result


def reconstruct_from_lower_triangular(lower_vec: np.ndarray, D: int) -> np.ndarray:
    """
    Reconstruct Hermitian density matrix from its lower triangular part.
    
    Args:
        lower_vec: (D*(D+1),) float array with [Re(lower), Im(lower)]
        D: dimension of the density matrix
    
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
    
    # Fill upper triangular by Hermitian conjugate
    dm = dm + dm.conj().T - np.diag(dm.diagonal())
    
    return dm


def make_dataset_dm_noisy(
    n: int,
    n_qubits: int = 2,
    seed: int = 0,
    depol_error: float = 0.01,
    amp_damp_error: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset with noisy density matrices using Qiskit Aer NoiseModel.
    
    The noise model includes:
    - Depolarizing error (probability depol_error on each qubit gate)
    - Amplitude damping error (probability amp_damp_error on each qubit gate)
    
    Args:
        n: number of samples
        n_qubits: number of qubits (default 2)
        seed: RNG seed for reproducibility
        depol_error: depolarizing error probability
        amp_damp_error: amplitude damping error probability
    
    Returns:
        X: (n, 4, 3) gate parameters
        Y_dm: (n, D*(D+1)) lower triangular density matrix components [Re, Im]
              where D = 2^n_qubits
    """
    rng = np.random.default_rng(seed)
    
    # For 2-qubit: D=4, lower triangular has 10 complex = 20 real values
    # For 3-qubit: D=8, lower triangular has 36 complex = 72 real values
    D = 2 ** n_qubits
    n_lower = D * (D + 1) // 2
    
    X = np.zeros((n, 4, 3), dtype=np.float32)
    Y_dm = np.zeros((n, 2 * n_lower), dtype=np.float32)
    
    # Build noise model
    noise_model = NoiseModel()
    
    # Add depolarizing and amplitude damping errors to single-qubit gates
    single_qubit_gates = ['u', 'x', 'y', 'z', 'h', 's', 't']
    for gate_name in single_qubit_gates:
        error_depol = depolarizing_error(depol_error, 1)
        error_amp = amplitude_damping_error(amp_damp_error)
        combined_error = error_depol.compose(error_amp)
        noise_model.add_all_qubit_quantum_error(combined_error, gate_name)
    
    # Add error to CNOT (2-qubit) gates
    error_2q_depol = depolarizing_error(depol_error * 2, 2)  # roughly 2x for 2-qubit
    noise_model.add_all_qubit_quantum_error(error_2q_depol, 'cx')
    
    # Create simulator with noise
    simulator = AerSimulator(noise_model=noise_model)
    
    for i in range(n):
        theta = sample_2q_special_params(rng)
        
        # Pack parameters
        X[i, 0, :] = theta[0:3]
        X[i, 1, :] = theta[3:6]
        X[i, 2, :] = theta[6:9]
        X[i, 3, :] = theta[9:12]
        
        # Build circuit with measurements (necessary for density matrix extraction)
        qc = build_2q_special_circuit(theta)
        
        # Simulate with noise to get noisy statevector/density matrix
        # Use simulator to get final state as density matrix
        result = simulator.run(qc, shots=1, seed_simulator=int(seed) + i).result()
        
        # Alternative: use operator_simulator to get density matrix directly
        # For 2-qubit system, use save_density_matrix
        qc_with_dm = build_2q_special_circuit(theta)
        qc_with_dm.save_density_matrix()
        result_dm = simulator.run(qc_with_dm, seed_simulator=int(seed) + i).result()
        
        try:
            dm_noisy = result_dm.data(0)['density_matrix']
        except (KeyError, IndexError):
            # Fallback: compute density matrix from statevector
            result_sv = simulator.run(build_2q_special_circuit(theta), seed_simulator=int(seed) + i).result()
            sv = result_sv.get_statevector(0)
            dm_noisy = np.outer(sv, sv.conj())
        
        # Extract lower triangular part (real + imag)
        y_dm = extract_lower_triangular(dm_noisy)
        Y_dm[i, :] = y_dm
    
    return X, Y_dm
