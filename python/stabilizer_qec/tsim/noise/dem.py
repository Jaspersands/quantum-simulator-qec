# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/noise/dem.py; Apache License 2.0;
# see LICENSE in the tsim directory), modified for stabilizer_qec: stim is stabilizer_qec, and the
# circuit is this package's (a tsim.Circuit read as Stim reads Tsim's tagged form of it).
"""Detector error model generation for QEC decoder integration."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import stabilizer_qec as sq


def get_detector_error_model(
    circuit: Any,
    *,
    allow_non_deterministic_observables: bool = True,
    decompose_errors: bool = False,
    flatten_loops: bool = False,
    allow_gauge_detectors: bool = False,
    approximate_disjoint_errors: Any = False,
    ignore_decomposition_failures: bool = False,
    block_decomposition_from_introducing_remnant_edges: bool = False,
) -> "sq.DetectorErrorModel":
    """The circuit's detector error model, as Tsim's ``Circuit.detector_error_model``: unlike
    Stim's, it allows non-deterministic observables (a logical observable a non-Clifford gate
    leaves random) when ``allow_non_deterministic_observables`` is set, by converting the
    observables into detectors, building the model with ``allow_gauge_detectors=True``, and
    converting those detectors back into observables; gauge errors on observables alone
    (``error(0.5) L0``) are dropped. With ``decompose_errors`` that conversion is not supported.

    A ``tsim.Circuit``'s model is Tsim's: Stim's analysis of Tsim's tagged form, which reads T
    as ``S``, a rotation or ``U3`` as ``I`` and a Pauli-product rotation as ``SPP`` (so
    ``R 0``, ``H 0``, ``T 0``, ``H 0``, ``M 0`` has a random observable, ``error(0.5) D0 L0``);
    a ``stabilizer_qec.Circuit``'s reads ``I[T]`` and the rotations as the identity.

    WARNING: if the circuit has distance one (an error flips an observable alone), the model is
    wrong, as Tsim's is.
    """
    if allow_non_deterministic_observables and decompose_errors:
        raise ValueError("Decomposition of error mechanisms is not supported when allowing non-deterministic observables.")
    kwargs = dict(
        allow_gauge_detectors=allow_gauge_detectors,
        decompose_errors=decompose_errors,
        flatten_loops=flatten_loops,
        approximate_disjoint_errors=approximate_disjoint_errors,
        ignore_decomposition_failures=ignore_decomposition_failures,
        block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
    )
    from stabilizer_qec.tsim._native import stim_reading
    from stabilizer_qec.tsim.circuit import Circuit as TsimCircuit

    if isinstance(circuit, TsimCircuit):
        # Tsim's model is Stim's of its tagged circuit: T read as S, a rotation as I, a
        # Pauli-product rotation as SPP.
        circuit = sq.Circuit(stim_reading(circuit._stim_exact_text()))
    circuit = circuit if isinstance(circuit, sq.Circuit) else sq.Circuit(circuit)
    if not allow_non_deterministic_observables:
        # Stim's model (a tsim.Circuit's own method is this one's).
        return sq.Circuit.detector_error_model(circuit, **kwargs)

    obs: dict = defaultdict(list)
    new_circuit = sq.Circuit()

    # NOTE: stim allows multiple OBSERVABLE_INCLUDE instruction with the same index.
    # We will combine them into a single OBSERVABLE_INCLUDE instructions and
    # push them to the end of the circuit. This requires updating the rec[] indices.
    for instruction in circuit.flattened():
        assert not isinstance(instruction, sq.CircuitRepeatBlock)
        if sq.gate_data(instruction.name).produces_measurements:
            # Ask the circuit how many records this exact instruction appended.
            probe = sq.Circuit()
            probe.append(instruction)
            num_meas = probe.num_measurements
            for idx in obs:
                obs[idx] = [t - num_meas for t in obs[idx]]

        if instruction.name == "OBSERVABLE_INCLUDE":
            assert len(instruction.gate_args_copy()) == 1
            idx = int(instruction.gate_args_copy()[0])
            obs[idx].extend(t.value for t in instruction.targets_copy())
        else:
            new_circuit.append_operation(instruction.name, instruction.targets_copy(), instruction.gate_args_copy(), tag=instruction.tag)

    # obs combines all OBSERVABLE_INCLUDE instructions. We now add them to the end
    # of the flattened circuit as DETECTOR instructions.
    num_detectors = circuit.num_detectors
    mapping: dict = {}
    for idx, targets in obs.items():
        new_circuit.append_operation("DETECTOR", [sq.target_rec(t) for t in targets])
        mapping[num_detectors] = idx
        num_detectors += 1

    kwargs["allow_gauge_detectors"] = True
    dem = new_circuit.detector_error_model(**kwargs)
    new_dem = sq.DetectorErrorModel()
    for instruction in dem:
        assert not isinstance(instruction, sq.DemRepeatBlock)
        new_targets = []
        new_type = instruction.type
        for t in instruction.targets_copy():
            if isinstance(t, sq.DemTarget) and t.is_relative_detector_id() and t.val in mapping:
                new_targets.append(sq.target_logical_observable_id(mapping[t.val]))
                if instruction.type == "detector":
                    new_type = "logical_observable"
            else:
                new_targets.append(t)
        new_instruction = sq.DemInstruction(new_type, instruction.args_copy(), new_targets)
        # Remove gauge statements that only affect logical observables (e.g., "error(0.5) L0").
        if instruction.args_copy() == [0.5]:
            if all(isinstance(t, sq.DemTarget) and t.is_logical_observable_id() for t in new_targets):
                continue
        new_dem.append(new_instruction)

    if new_dem.num_observables != circuit.num_observables:
        raise ValueError(
            "Failed to compute detector error model. "
            "The number of observables changed after conversion. "
            "This indicates that stim has interpreted logical observables as gauges "
            f"and removed them. Error model:\n {new_dem!s}"
        )
    return new_dem
