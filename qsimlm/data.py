from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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
