from __future__ import annotations
"""
Optional cross-check between Qiskit and PennyLane.
Requires: pennylane, pennylane-qiskit
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def probs_from_qiskit(qc: QuantumCircuit) -> np.ndarray:
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.probabilities(), dtype=np.float64)

def probs_from_pennylane(qc: QuantumCircuit) -> np.ndarray:
    import pennylane as qml  # local import for optional dep
    n = qc.num_qubits
    dev = qml.device("default.qubit", wires=n)
    qfunc = qml.from_qiskit(qc)

    @qml.qnode(dev)
    def circuit():
        qfunc()
        return qml.probs(wires=list(range(n)))

    return np.asarray(circuit(), dtype=np.float64)

def quick_check():
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0,1)
    p1 = probs_from_qiskit(qc)
    p2 = probs_from_pennylane(qc)
    tv = 0.5 * np.abs(p1 - p2).sum()
    print("TV(Qiskit,PennyLane) =", float(tv))

if __name__ == "__main__":
    quick_check()
