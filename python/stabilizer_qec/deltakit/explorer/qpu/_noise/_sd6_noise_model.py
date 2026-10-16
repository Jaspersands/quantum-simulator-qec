# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""
This module includes implementation of a standard depolarising noise model, SD6.
This noise model sets all error probabilities to a single value p.
This model can be used for theory and simulation but is not
representative of real-world experiments.
"""

from dataclasses import dataclass

from stabilizer_qec.deltakit.circuit import after_reset_flip_probability, measurement_noise_profile
from stabilizer_qec.deltakit.circuit.gates import TWO_QUBIT_GATES, OneQubitCliffordGate
from stabilizer_qec.deltakit.circuit.noise_channels import Depolarise1, Depolarise2

from stabilizer_qec.deltakit.explorer.qpu._noise._noise_parameters import NoiseParameters


@dataclass(kw_only=True)
class SD6Noise(NoiseParameters):
    """
    The standard depolarising noise model as seen in QEC literature, e.g.:
    Table 2 in https://arxiv.org/abs/2108.10457
    This model sets all one- and two-qubit depolarising error probabilities to p,
    as well as setting all measurement flip probabilities to p.
    This model can be used for theory and simulation but is not
    representative of other models seen in real hardware devices.

    Parameters
    ----------
    p: float, optional
        Rate of all depolarising errors and measurement flips. By default, 0.0.
    """

    name = "sd6_noise"

    p: float = 0

    def __post_init__(self):
        depolarise1_generator = Depolarise1.generator_from_prob(self.p)
        depolarise2_generator = Depolarise2.generator_from_prob(self.p)

        self.gate_noise.append(
            lambda noise_context: depolarise1_generator(
                noise_context.gate_layer_qubits(
                    (*tuple(TWO_QUBIT_GATES), OneQubitCliffordGate), gate_qubit_count=1
                )
            )
        )

        self.gate_noise.append(
            lambda noise_context: depolarise2_generator(
                noise_context.gate_layer_qubits(
                    tuple(TWO_QUBIT_GATES), gate_qubit_count=2
                )
            )
        )

        self.idle_noise = lambda qubit, _t=0.0: Depolarise1(
            qubit=qubit, probability=self.p
        )

        self.reset_noise = after_reset_flip_probability(self.p)

        self.measurement_flip = measurement_noise_profile(self.p)

    def __str__(self) -> str:
        return f"{self.name}_{self.p:.0e}"
