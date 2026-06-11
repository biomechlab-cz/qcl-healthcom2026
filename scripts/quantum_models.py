"""Small Qiskit circuits and local featurizers for ECG-first QML bakeoffs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector


def temporal_tokens(window: np.ndarray, n_tokens: int = 8) -> np.ndarray:
    """Reduce a window to modality-wise temporal means in [-pi, pi]."""
    if window.ndim != 2:
        raise ValueError("window must have shape (samples, channels)")
    chunks = np.array_split(window, n_tokens, axis=0)
    tokens = np.array([chunk.mean(axis=0) for chunk in chunks], dtype=float)
    scale = np.std(tokens, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    tokens = (tokens - np.mean(tokens, axis=0, keepdims=True)) / scale
    return np.clip(tokens, -3.0, 3.0) / 3.0 * np.pi


def build_qrc_circuit(tokens: np.ndarray, n_qubits: int = 6, entanglement: str = "linear") -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    n_channels = tokens.shape[1]
    for step in range(tokens.shape[0]):
        for channel in range(n_channels):
            q0 = (2 * channel) % n_qubits
            q1 = (2 * channel + 1) % n_qubits
            angle = float(tokens[step, channel])
            circuit.rx(angle, q0)
            circuit.rz(angle * angle / np.pi, q1)
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
        if entanglement == "ring" and n_qubits > 2:
            edges.append((n_qubits - 1, 0))
        for left, right in edges:
            circuit.rzz(0.2, left, right)
    return circuit


def build_fusion_circuit(
    features: np.ndarray,
    channels: list[str],
    qubits_per_channel: int = 2,
    reps: int = 1,
    entanglement: str = "minimal",
    mixer_angle: float = 0.0,
) -> QuantumCircuit:
    n_qubits = len(channels) * qubits_per_channel
    circuit = QuantumCircuit(n_qubits)
    values = np.asarray(features, dtype=float)
    if values.size < n_qubits:
        values = np.pad(values, (0, n_qubits - values.size))
    values = np.clip(values[:n_qubits], -3.0, 3.0) / 3.0 * np.pi
    for rep in range(max(1, reps)):
        bandwidth = 1.0 / (rep + 1)
        for q, value in enumerate(values):
            angle = float(value * bandwidth)
            circuit.ry(angle, q)
            circuit.rz(float(angle * angle / np.pi), q)
        for channel in range(len(channels)):
            base = channel * qubits_per_channel
            if qubits_per_channel > 1:
                if entanglement == "full":
                    for offset in range(qubits_per_channel - 1):
                        circuit.cz(base + offset, base + offset + 1)
                else:
                    circuit.cz(base, base + 1)
        for channel in range(len(channels) - 1):
            left = channel * qubits_per_channel
            right = (channel + 1) * qubits_per_channel
            if entanglement == "full":
                for offset in range(qubits_per_channel):
                    circuit.rzz(0.35 * bandwidth, left + offset, right + offset)
            else:
                circuit.rzz(0.35 * bandwidth, left, right)
        if mixer_angle:
            for q in range(n_qubits):
                circuit.rx(float(mixer_angle * bandwidth), q)
    return circuit


def z_expectations(circuit: QuantumCircuit) -> np.ndarray:
    state = Statevector.from_instruction(circuit)
    values = []
    for q in range(circuit.num_qubits):
        label = ["I"] * circuit.num_qubits
        label[circuit.num_qubits - q - 1] = "Z"
        values.append(float(np.real(state.expectation_value(SparsePauliOp.from_list([("".join(label), 1.0)])))))
    return np.asarray(values, dtype=float)


def z_zz_expectations(circuit: QuantumCircuit) -> np.ndarray:
    state = Statevector.from_instruction(circuit)
    values = []
    for q in range(circuit.num_qubits):
        label = ["I"] * circuit.num_qubits
        label[circuit.num_qubits - q - 1] = "Z"
        values.append(float(np.real(state.expectation_value(SparsePauliOp.from_list([("".join(label), 1.0)])))))
    for q in range(circuit.num_qubits - 1):
        label = ["I"] * circuit.num_qubits
        label[circuit.num_qubits - q - 1] = "Z"
        label[circuit.num_qubits - q - 2] = "Z"
        values.append(float(np.real(state.expectation_value(SparsePauliOp.from_list([("".join(label), 1.0)])))))
    return np.asarray(values, dtype=float)


def circuit_summary(circuit: QuantumCircuit) -> dict[str, int]:
    counts = circuit.count_ops()
    two_qubit = sum(int(counts.get(name, 0)) for name in ["cx", "cz", "rzz", "ecr"])
    transpiled = transpile(circuit, optimization_level=1)
    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "two_qubit_gates": two_qubit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Feature parquet/csv for a representative fusion circuit")
    parser.add_argument("--output", type=Path, default=Path("results/quantum_circuit_summary.csv"))
    args = parser.parse_args()

    rows = []
    demo_tokens = np.zeros((8, 1))
    rows.append({"model": "qrc_ecg", **circuit_summary(build_qrc_circuit(demo_tokens, n_qubits=6))})

    if args.features:
        frame = pd.read_csv(args.features) if args.features.suffix == ".csv" else pd.read_parquet(args.features)
        numeric = frame.select_dtypes(include=[np.number]).drop(columns=["label", "window_idx"], errors="ignore")
        channels = ["ECG", "EDA"] if "signal_set" in frame and (frame["signal_set"] == "ecg_eda").any() else ["ECG"]
        rows.append(
            {
                "model": "fusion_" + "_".join(channels).lower(),
                **circuit_summary(build_fusion_circuit(numeric.iloc[0].to_numpy(), channels=channels)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
