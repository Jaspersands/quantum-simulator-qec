# Ported from the QDK (microsoft/qdk 1.33.1, qdk/estimator/_estimator.py: the parameter classes; Copyright (c)
# Microsoft Corporation, MIT License; see LICENSE-qdk in this directory), modified for stabilizer_qec: written
# for Python 3.9; `QECScheme`'s names are on `stabilizer_qec.estimate.azure.QECScheme`, the port's scheme model.
# Their dictionaries, checks, messages and warning are the QDK's (tests/estimate_port/qdk_params.json).
"""The QDK's parameter classes for its resource estimator (``qdk.estimator``): a job's qubit model, QEC scheme,
distillation units, constraints and error budget as objects, each written by ``as_dict()`` as the camelCase
dict the estimate reads, checked as the QDK checks it (a value out of range is a ValueError, a missing field of
a custom model a LookupError, each with the QDK's message). `EstimatorParams` is a whole job, or a batch of
``num_items`` items (at most 1000) whose unset sections come from the top level. Pass one to
`LogicalCounts.estimate` or `azure_estimate`; the classes are importable from `stabilizer_qec.estimate.azure`.

>>> from stabilizer_qec.estimate.azure import EstimatorQubitParams, MeasurementErrorRate, QubitParams
>>> EstimatorQubitParams(name=QubitParams.MAJ_NS_E4, one_qubit_measurement_error_rate=MeasurementErrorRate(1e-4, 2e-4)).as_dict()
{'name': 'qubit_maj_ns_e4', 'oneQubitMeasurementErrorRate': {'process': 0.0001, 'readout': 0.0002}}
>>> EstimatorQubitParams(instruction_set="GateBased").as_dict()
Traceback (most recent call last):
...
LookupError: one_qubit_measurement_time must be set
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class AutoValidatingParams:
    """
    A helper class for target parameters.

    It has a function as_dict that automatically extracts a dictionary from
    the class' fields.  They are added to the result dictionary if their value
    is not None, the key is automatically transformed from Python snake case
    to camel case, and if validate is True and if the field has a validation
    function, the field is validated beforehand.
    """

    def as_dict(self, validate: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for name, field in self.__dataclass_fields__.items():
            field_value = self.__getattribute__(name)
            if field_value is not None:
                # validate field?
                if validate and "validate" in field.metadata:
                    func = field.metadata["validate"]
                    # check for indirect call (like in @staticmethod)
                    if hasattr(func, "__func__"):
                        func = func.__func__
                    func(name, field_value)

                # translate field name to camel case
                s = re.sub(r"(_|-)+", " ", name).title().replace(" ", "")
                attribute = "".join([s[0].lower(), s[1:]])
                result[attribute] = field_value

        if validate:
            self.post_validation(result)

        return result

    def post_validation(self, result: Dict[str, Any]) -> None:
        """
        A function that is called after all individual fields have been
        validated, but before the result is returned.

        Here result is the current dictionary.
        """
        pass


def validating_field(validation_func: Any, default: Any = None) -> Any:
    """
    A helper method to declare field for an AutoValidatingParams data class.
    """
    return field(default=default, metadata={"validate": validation_func})


class QubitParams:
    """
    Predefined qubit model name constants for use with :class:`EstimatorQubitParams`.

    Pass one of these string constants as the ``name`` field to select a built-in
    qubit model for resource estimation.
    """

    GATE_US_E3 = "qubit_gate_us_e3"
    GATE_US_E4 = "qubit_gate_us_e4"
    GATE_NS_E3 = "qubit_gate_ns_e3"
    GATE_NS_E4 = "qubit_gate_ns_e4"
    MAJ_NS_E4 = "qubit_maj_ns_e4"
    MAJ_NS_E6 = "qubit_maj_ns_e6"


def _check_error_rate(name: str, value: float) -> None:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _check_error_rate_or_process_and_readout(name: str, value: Optional[Union[float, "MeasurementErrorRate"]]) -> None:
    if value is None:
        return

    if isinstance(value, float):
        _check_error_rate(name, value)
        return

    if not isinstance(value, MeasurementErrorRate):
        raise ValueError(f"{name} must be either a float or " "MeasurementErrorRate with two fields: 'process' and 'readout'")


def check_time(name: str, value: str) -> None:
    pat = r"^(\+?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(s|ms|μs|µs|us|ns)$"
    if re.match(pat, value) is None:
        raise ValueError(f"{name} is not a valid time string; use a suffix s, ms, us, or ns")


@dataclass
class MeasurementErrorRate(AutoValidatingParams):
    """
    Measurement error rate specified separately for process and readout errors.

    Used as the value of ``one_qubit_measurement_error_rate`` or
    ``two_qubit_joint_measurement_error_rate`` in :class:`EstimatorQubitParams`
    when process and readout error rates differ.

    :param process: Error rate during the measurement process. Must be in ``(0, 1)``.
    :param readout: Error rate during readout. Must be in ``(0, 1)``.
    """

    process: float = field(metadata={"validate": _check_error_rate})
    readout: float = field(metadata={"validate": _check_error_rate})


@dataclass
class EstimatorQubitParams(AutoValidatingParams):
    """
    Physical qubit parameters for resource estimation.

    Specify a built-in qubit model by setting ``name`` to one of the
    :class:`QubitParams` constants, or fully define a custom model by setting
    ``instruction_set`` and all relevant timing and error-rate fields.
    """

    @staticmethod
    def check_instruction_set(name: str, value: str) -> None:
        if value not in ["gate-based", "gate_based", "GateBased", "gateBased", "Majorana", "majorana"]:
            raise ValueError(f"{name} must be GateBased or Majorana")

    name: Optional[str] = None
    instruction_set: Optional[str] = validating_field(check_instruction_set)
    one_qubit_measurement_time: Optional[str] = validating_field(check_time)
    two_qubit_joint_measurement_time: Optional[str] = validating_field(check_time)
    one_qubit_gate_time: Optional[str] = validating_field(check_time)
    two_qubit_gate_time: Optional[str] = validating_field(check_time)
    t_gate_time: Optional[str] = validating_field(check_time)
    one_qubit_measurement_error_rate: Union[None, float, MeasurementErrorRate] = validating_field(_check_error_rate_or_process_and_readout)
    two_qubit_joint_measurement_error_rate: Union[None, float, MeasurementErrorRate] = validating_field(_check_error_rate_or_process_and_readout)
    one_qubit_gate_error_rate: Optional[float] = validating_field(_check_error_rate)
    two_qubit_gate_error_rate: Optional[float] = validating_field(_check_error_rate)
    t_gate_error_rate: Optional[float] = validating_field(_check_error_rate)
    idle_error_rate: Optional[float] = validating_field(_check_error_rate)

    _default_models = [QubitParams.GATE_US_E3, QubitParams.GATE_US_E4, QubitParams.GATE_NS_E3, QubitParams.GATE_NS_E4, QubitParams.MAJ_NS_E4, QubitParams.MAJ_NS_E6]
    _gate_based = ["gate-based", "gate_based", "GateBased", "gateBased"]
    _maj_based = ["Majorana", "majorana"]

    def post_validation(self, result: Dict[str, Any]) -> None:
        # check whether all fields have been specified in case a custom qubit
        # model is specified
        custom = result != {} and (self.name is None or self.name not in self._default_models)

        # no further validation needed for non-custom models
        if not custom:
            return

        # instruction set must be set
        if self.instruction_set is None:
            raise LookupError("instruction_set must be set for custom qubit parameters")

        # NOTE at this point, we know that instruction set must have valid
        # value
        if self.one_qubit_measurement_time is None:
            raise LookupError("one_qubit_measurement_time must be set")
        if self.one_qubit_measurement_error_rate is None:
            raise LookupError("one_qubit_measurement_error_rate must be set")

        # this only needs to be checked for gate based qubits
        if self.instruction_set in self._gate_based:
            if self.one_qubit_gate_time is None:
                raise LookupError("one_qubit_gate_time must be set")

    def as_dict(self, validate: bool = True) -> Dict[str, Any]:
        qubit_params = super().as_dict(validate)
        if len(qubit_params) != 0:
            if isinstance(self.one_qubit_measurement_error_rate, MeasurementErrorRate):
                qubit_params["oneQubitMeasurementErrorRate"] = self.one_qubit_measurement_error_rate.as_dict(validate)

            if isinstance(self.two_qubit_joint_measurement_error_rate, MeasurementErrorRate):
                qubit_params["twoQubitJointMeasurementErrorRate"] = self.two_qubit_joint_measurement_error_rate.as_dict(validate)

        return qubit_params


@dataclass
class EstimatorQecScheme(AutoValidatingParams):
    """
    Quantum error correction scheme parameters for resource estimation.

    Specify a built-in scheme by setting ``name`` to one of the
    :class:`QECScheme` constants, or define a custom scheme by setting
    the threshold and code-distance parameters directly.
    """

    name: Optional[str] = None
    error_correction_threshold: Optional[float] = validating_field(_check_error_rate)
    crossing_prefactor: Optional[float] = None
    distance_coefficient_power: Optional[int] = None
    logical_cycle_time: Optional[str] = None
    physical_qubits_per_logical_qubit: Optional[str] = None
    max_code_distance: Optional[int] = None


@dataclass
class ProtocolSpecificDistillationUnitSpecification(AutoValidatingParams):
    """
    Protocol-specific specification for a magic-state distillation unit.

    Defines the number of physical qubits and the duration (in logical cycle
    time units) for one round of distillation under a specific QEC code.
    """

    num_unit_qubits: Optional[int] = None
    duration_in_qubit_cycle_time: Optional[int] = None

    def post_validation(self, result: Dict[str, Any]) -> None:
        if self.num_unit_qubits is None:
            raise LookupError("num_unit_qubits must be set")

        if self.duration_in_qubit_cycle_time is None:
            raise LookupError("duration_in_qubit_cycle_time must be set")


@dataclass
class DistillationUnitSpecification(AutoValidatingParams):
    """
    Specification for a magic-state distillation unit.

    Either select a built-in unit by setting ``name``, or define a custom unit
    by providing ``num_input_ts``, ``num_output_ts``, ``failure_probability_formula``,
    ``output_error_rate_formula``, and optionally physical and logical qubit
    specifications.
    """

    name: Optional[str] = None
    display_name: Optional[str] = None
    num_input_ts: Optional[int] = None
    num_output_ts: Optional[int] = None
    failure_probability_formula: Optional[str] = None
    output_error_rate_formula: Optional[str] = None
    physical_qubit_specification: Optional[ProtocolSpecificDistillationUnitSpecification] = None
    logical_qubit_specification: Optional[ProtocolSpecificDistillationUnitSpecification] = None
    logical_qubit_specification_first_round_override: Optional[ProtocolSpecificDistillationUnitSpecification] = None

    def has_custom_specification(self):
        return (
            self.display_name is not None
            or self.num_input_ts is not None
            or self.num_output_ts is not None
            or self.failure_probability_formula is not None
            or self.output_error_rate_formula is not None
            or self.physical_qubit_specification is not None
            or self.logical_qubit_specification is not None
            or self.logical_qubit_specification_first_round_override is not None
        )

    def has_predefined_name(self):
        return self.name is not None

    def post_validation(self, result: Dict[str, Any]) -> None:
        if not self.has_custom_specification() and not self.has_predefined_name():
            raise LookupError("name must be set or custom specification must be provided")

        if self.has_custom_specification() and self.has_predefined_name():
            raise LookupError("If predefined name is provided, " "custom specification is not allowed. " "Either remove name or remove all other " "specification of the distillation unit")

        if self.has_predefined_name():
            return  # all other validation is on the server side

        if self.num_input_ts is None:
            raise LookupError("num_input_ts must be set")

        if self.num_output_ts is None:
            raise LookupError("num_output_ts must be set")

        if self.failure_probability_formula is None:
            raise LookupError("failure_probability_formula must be set")

        if self.output_error_rate_formula is None:
            raise LookupError("output_error_rate_formula must be set")

        if self.physical_qubit_specification is not None:
            self.physical_qubit_specification.post_validation(result)

        if self.logical_qubit_specification is not None:
            self.logical_qubit_specification.post_validation(result)

        if self.logical_qubit_specification_first_round_override is not None:
            self.logical_qubit_specification_first_round_override.post_validation(result)

    def as_dict(self, validate: bool = True) -> Dict[str, Any]:
        specification_dict = super().as_dict(validate)
        if len(specification_dict) != 0:
            if self.physical_qubit_specification is not None:
                physical_qubit_specification_dict = self.physical_qubit_specification.as_dict(validate)
                if len(physical_qubit_specification_dict) != 0:
                    specification_dict["physicalQubitSpecification"] = physical_qubit_specification_dict

            if self.logical_qubit_specification is not None:
                logical_qubit_specification_dict = self.logical_qubit_specification.as_dict(validate)
                if len(logical_qubit_specification_dict) != 0:
                    specification_dict["logicalQubitSpecification"] = logical_qubit_specification_dict

            if self.logical_qubit_specification_first_round_override is not None:
                logical_qubit_specification_first_round_override_dict = self.logical_qubit_specification_first_round_override.as_dict(validate)
                if len(logical_qubit_specification_first_round_override_dict) != 0:
                    specification_dict["logicalQubitSpecificationFirstRoundOverride"] = logical_qubit_specification_first_round_override_dict

        return specification_dict


@dataclass
class ErrorBudgetPartition(AutoValidatingParams):
    """
    Partition of the total error budget across algorithm components.

    The three fields must sum to the overall error budget. Defaults to equal
    thirds of ``0.001`` (i.e. each component gets ``~3.33e-4``).

    :param logical: Budget allocated to logical errors in the algorithm.
    :param t_states: Budget allocated to T-state distillation errors.
    :param rotations: Budget allocated to rotation synthesis errors.
    """

    logical: float = 0.001 / 3
    t_states: float = 0.001 / 3
    rotations: float = 0.001 / 3


@dataclass
class EstimatorConstraints(AutoValidatingParams):
    """
    Optional runtime and resource constraints for resource estimation.

    At most one of ``max_duration`` or ``max_physical_qubits`` may be set
    simultaneously.
    """

    @staticmethod
    def at_least_one(name: str, value: Union[float, int]) -> None:
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    logical_depth_factor: Optional[float] = validating_field(at_least_one)
    max_t_factories: Optional[int] = validating_field(at_least_one)
    max_duration: Optional[int] = validating_field(check_time)
    max_physical_qubits: Optional[int] = validating_field(at_least_one)

    def post_validation(self, result: Dict[str, Any]) -> None:
        if self.max_duration is not None and self.max_physical_qubits is not None:
            raise LookupError("Both duration and number of physical qubits constraints are provided, but only one is allowed at a time.")


class EstimatorInputParamsItem:
    """
    Input parameters for a single resource estimation job.

    Contains qubit model, QEC scheme, distillation unit specifications, constraints,
    and error budget settings. Used directly for single-point estimation or as a
    base class for batching via :class:`EstimatorParams`.
    """

    def __init__(self) -> None:
        super().__init__()

        self.qubit_params: EstimatorQubitParams = EstimatorQubitParams()
        self.qec_scheme: EstimatorQecScheme = EstimatorQecScheme()
        self.distillation_unit_specifications: List[DistillationUnitSpecification] = []
        self.constraints: EstimatorConstraints = EstimatorConstraints()
        self.error_budget: Optional[Union[float, ErrorBudgetPartition]] = None
        self.estimate_type: Optional[str] = None

    def as_dict(self, validate: bool = True, additional_params: Optional["EstimatorInputParamsItem"] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        qubit_params = self.qubit_params.as_dict(validate)
        if len(qubit_params) != 0:
            result["qubitParams"] = qubit_params
        elif additional_params and hasattr(additional_params, "qubit_params"):
            qubit_params = additional_params.qubit_params.as_dict(validate)
            if len(qubit_params) != 0:
                result["qubitParams"] = qubit_params

        qec_scheme = self.qec_scheme.as_dict(validate)
        if len(qec_scheme) != 0:
            result["qecScheme"] = qec_scheme
        elif additional_params and hasattr(additional_params, "qec_scheme"):
            qec_scheme = additional_params.qec_scheme.as_dict(validate)
            if len(qec_scheme) != 0:
                result["qecScheme"] = qec_scheme

        for specification in self.distillation_unit_specifications:
            specification_dict = specification.as_dict(validate)
            if len(specification_dict) != 0:
                if result.get("distillationUnitSpecifications") is None:
                    result["distillationUnitSpecifications"] = []

                result["distillationUnitSpecifications"].append(specification_dict)
        # As the QDK: the top level's units are added only to an item that has its own.
        if result.get("distillationUnitSpecifications") is not None and additional_params is not None and hasattr(additional_params, "distillation_unit_specifications"):
            for specification in additional_params.distillation_unit_specifications:
                specification_dict = specification.as_dict(validate)
                if len(specification_dict) != 0:
                    if result.get("distillationUnitSpecifications") is None:
                        result["distillationUnitSpecifications"] = []

                    result["distillationUnitSpecifications"].append(specification_dict)

        constraints = self.constraints.as_dict(validate)
        if len(constraints) != 0:
            result["constraints"] = constraints
        elif additional_params and hasattr(additional_params, "constraints"):
            constraints = additional_params.constraints.as_dict(validate)
            if len(constraints) != 0:
                result["constraints"] = constraints

        if self.error_budget is not None:
            if isinstance(self.error_budget, float) or isinstance(self.error_budget, int):
                if validate and (self.error_budget <= 0 or self.error_budget >= 1):
                    message = "error_budget must be value between 0 and 1"
                    raise ValueError(message)
                result["errorBudget"] = self.error_budget
            elif isinstance(self.error_budget, ErrorBudgetPartition):
                result["errorBudget"] = self.error_budget.as_dict(validate)
        elif additional_params and hasattr(additional_params, "error_budget"):
            if isinstance(additional_params.error_budget, float) or isinstance(additional_params.error_budget, int):
                if validate and (additional_params.error_budget <= 0 or additional_params.error_budget >= 1):
                    message = "error_budget must be value between 0 and 1"
                    raise ValueError(message)
                result["errorBudget"] = additional_params.error_budget
            elif isinstance(additional_params.error_budget, ErrorBudgetPartition):
                result["errorBudget"] = additional_params.error_budget.as_dict(validate)

        if self.estimate_type is not None:
            if self.estimate_type not in ["frontier", "singlePoint"]:
                raise ValueError("estimate_type must be either 'frontier' or 'singlePoint'")
            result["estimateType"] = self.estimate_type

        return result


class EstimatorParams(EstimatorInputParamsItem):
    """
    Top-level input parameters for the Microsoft Resource Estimator.

    Extends :class:`EstimatorInputParamsItem` with support for batching: pass
    ``num_items`` to create a batching job where each item can override the
    top-level parameters.

    :param num_items: Number of batching items. If ``None``, creates a
        single-point estimation job.
    :type num_items: int
    """

    MAX_NUM_ITEMS: int = 1000

    def __init__(self, num_items: Optional[int] = None):
        warnings.warn(
            "This version of QRE is deprecated and will be removed in a future release. Please use the new version of QRE in qdk.qre. Refer to aka.ms/qdk.QREv3 for more information.",
            DeprecationWarning,
            stacklevel=2,
        )

        EstimatorInputParamsItem.__init__(self)

        if num_items is not None:
            self.has_items = True
            if num_items <= 0 or num_items > self.MAX_NUM_ITEMS:
                raise ValueError("num_items must be a positive value less or equal to " f"{self.MAX_NUM_ITEMS}")
            self._items = [EstimatorInputParamsItem() for _ in range(num_items)]
        else:
            self.has_items = False

    @property
    def items(self) -> List:
        if self.has_items:
            return self._items
        else:
            raise Exception("Cannot access items in a non-batching job, call " "make_params with num_items parameter")

    def as_dict(self, validate: bool = True, additional_params: Optional["EstimatorInputParamsItem"] = None) -> Dict[str, Any]:
        """
        Constructs a dictionary from the input params.

        For batching jobs, top-level entries are merged into item entries.
        Item entries have priority in case they are specified.
        """

        # initialize result and set type hint
        result: Dict[str, Any] = EstimatorInputParamsItem.as_dict(self, validate)

        if self.has_items:
            result["items"] = [item.as_dict(validate, self) for item in self._items]
            # In case of batching, no need to stop if failing an item
            result["resumeAfterFailedItem"] = True

        return result
