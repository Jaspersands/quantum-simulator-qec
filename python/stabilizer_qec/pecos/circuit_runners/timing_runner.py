#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #
# modified for stabilizer_qec: pecos is stabilizer_qec.pecos; total_time times run's gates. (PECOS's TimingRunner
# timed run_gates, which its run, Standard's, never calls: total_time stayed 0.0.)

from time import perf_counter as default_timer
from .standard import Standard


class TimingRunner(Standard):
    """
    This class represents a standard model for running quantum circuits and adding in errors.
    """

    def __init__(self, seed=None, timer=None):
        """

        Args:
            seed:
        """

        super().__init__(seed)

        self.total_time = 0.0
        self.num_gates = 0

        if timer is None:
            self.timer = default_timer
        else:
            self.timer = timer

    def run(self, state, circuit, error_gen=None, error_params=None, error_circuits=None, output=None):
        """Standard's run, the state's running of each circuit (the errors' too) timed, as run_gates times gates.

        (stabilizer_qec: PECOS's ran the state's run_circuit, untimed.)
        """
        return Standard.run(_TimedState(self, state), circuit, error_gen, error_params, error_circuits, output)

    def reset_time(self):
        """
        Used to clear the time data in `total_time`.

        Returns:

        """
        self.total_time = 0.0
        self.num_gates = 0

    def run_gates(self, state, gates, removed_locations=None):
        """
        Directly apply a collection of quantum gates to a state.

        Args:
            state:
            gates:
            removed_locations:

        Returns:

        """

        timer = self.timer

        if removed_locations is None:
            removed_locations = set([])

        gate_results = {}
        for symbol, physical_gate_locations, gate_kwargs in gates.items():

            ti = timer()
            results = state.run_gate(symbol, physical_gate_locations - removed_locations, **gate_kwargs)
            tf = timer()
            gate_results.update(results or {})  # (stabilizer_qec: every gate's results, as run_circuit's; PECOS's kept the last's)
            self.total_time += tf - ti
            self.num_gates += len(physical_gate_locations - removed_locations)

        return gate_results


class _TimedState:
    """(stabilizer_qec) The state, its run_circuit timed into the runner's total_time and num_gates. (Its own
    run_circuit, which PauliFaultProp's, for one, makes more than its gates.)"""

    def __init__(self, runner, state):
        self._runner = runner
        self._state = state

    def run_circuit(self, circuit, removed_locations=None):
        runner = self._runner
        removed = removed_locations or set()
        ti = runner.timer()
        results = self._state.run_circuit(circuit, removed_locations=removed_locations)
        tf = runner.timer()
        runner.total_time += tf - ti
        runner.num_gates += sum(len(locations - removed) for _, locations, _ in circuit.items())
        return results

    def __getattr__(self, name):
        return getattr(self._state, name)
