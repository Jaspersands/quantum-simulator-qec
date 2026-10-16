# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Module explores how the error suppression factor (lambda)
builds a connection between QuOps and code distance.

References
----------
- https://doi.org/10.48550/arXiv.2408.13687
"""

from collections.abc import Callable

from stabilizer_qec.deltakit.explorer.analysis._binomial_fit import ConfidenceInterval
from stabilizer_qec.deltakit.explorer.analysis._lambda import LambdaData


def _equal_or_less_brute_force_search(
    func: Callable[[int], float],
    target: float,
    minimum: int,
    maximum: int,
    step: int,
) -> int | None:
    """Brute force search or the solution at on an interval.

    func(...) is (unfortunately) not always a monotonic function.
    The function searches for smallest D from [min, max],
    with func(D) <= target.

    Parameters
    ----------
        func (Callable): a descending function.
        target (float): a value to search for.
        minimum (int): minimum tested value of D.
        maximum (int): maximum tested value of D.
        step (int): next value to consider is D+step.

    Returns
    -------
        int | None:
            a value, which safisfies the search, or None.
    """
    while minimum < maximum:
        value = func(minimum)
        if value <= target:
            return minimum
        minimum += step
    return None


def _calculate_lep(
    lambda0: float, lambda_: float, distance: int, num_rounds: int
) -> float:
    """Returns the probability of observing a logical error on a code of fixed
    distance after a number of rounds.

    It uses the formula in Section VI.B of Supplementary Information in
    https://doi.org/10.48550/arXiv.2408.13687 which is the sum of the probabilities
    of all ways of there being an odd number of errors in fixed number of rounds.
    """
    lep_per_round = lambda0 * lambda_ ** (-(distance + 1) / 2)
    # At `lep_per_round` << 1 this is be approximated as `lep_per_round * num_rounds`
    return 0.5 * (1 - (1 - 2 * lep_per_round) ** num_rounds)


def predict_quops_at_distance(lambda0: float, lambda_: float, distance: int) -> float:
    """Returns the number of QuOps, given distance.

    This uses the definition that the number of QuOps achievable is 1 / pL, where
    pL is the probability of a logical error occurring in distance-D, D-round block.

    Parameters
    ----------
    lambda0 (float):
        constant factor of lambda fit.
    lambda_ (float):
        Error suppression factor.
    distance (int):
        The distance at which to calculate the number of QuOps.
    """
    if distance % 2 == 0:
        msg = (
            "This method gives correct estimation only at odd distances. "
            f"Distance provided: {distance}"
        )
        raise ValueError(msg)
    return 1.0 / _calculate_lep(lambda0, lambda_, distance, distance)


def predict_quops_interval(
    lambda_data: LambdaData, distance: int
) -> ConfidenceInterval:
    """Predict the number of QuOps with a confidence interval.

    QuOps (quantum operations) is the number of logical operations achievable
    before a logical error occurs, taken as ``1 / pL`` for a distance-D, D-round
    block (see :func:`predict_quops_at_distance`). This propagates the asymmetric
    bounds on Λ and Λ₀ from an asymmetric Lambda fit (see
    :func:`calculate_lambda_asymmetric`) into that estimate. The number of QuOps
    grows with Λ and shrinks with Λ₀, so the widest QuOps comes from the largest Λ
    together with the smallest Λ₀.

    Args:
        lambda_data: A fit carrying the asymmetric ``lambda_interval`` and
            ``lambda0_interval`` bounds (see :attr:`LambdaData.has_asymmetric_bounds`).
        distance: The odd distance at which to evaluate the number of QuOps.

    Returns:
        The best estimate and its lower and upper bounds.

    Raises:
        ValueError: If ``lambda_data`` does not carry asymmetric bounds.
    """
    if not lambda_data.has_asymmetric_bounds:
        msg = (
            "lambda_data must carry asymmetric bounds. Use "
            "calculate_lambda_asymmetric to produce them."
        )
        raise ValueError(msg)
    # has_asymmetric_bounds guarantees both intervals are set.
    assert lambda_data.lambda_interval is not None
    assert lambda_data.lambda0_interval is not None
    best = predict_quops_at_distance(lambda_data.lambda0, lambda_data.lambda_, distance)
    low = predict_quops_at_distance(
        lambda_data.lambda0_interval.high, lambda_data.lambda_interval.low, distance
    )
    high = predict_quops_at_distance(
        lambda_data.lambda0_interval.low, lambda_data.lambda_interval.high, distance
    )
    return ConfidenceInterval(low=low, best=best, high=high)


def predict_distance_for_quops(
    lambda0: float,
    lambda_: float,
    num_quops: float,
    max_distance: int = 999,
) -> int:
    """Returns the nearest odd distance that achieves the desired
    number of QuOps.

    Uses the definition that the number of QuOps achievable at a
    particular distance is 1 / pL, where pL is the probability of a logical error
    occurring during a distance-D, D-round memory experiment
    without state preparation or measurement error.

    Parameters
    ----------
    lambda0 (float):
        constant factor of lambda fit.
    lambda_ (float):
        Error suppression factor.
    num_quops (int):
        Number of desired QuOps, must be a positive integer greater than 2.
    max_distance (int):
        maximum distance to consider. Default is 999.

    Raises
    ------
    ValueError
        - if solution is not found;
        - num_quops < 2;
        - lambda_ <= 1.0
    """

    if num_quops < 2:
        msg = "Number of QuOps should be at least 2"
        raise ValueError(msg)

    if lambda_ <= 1.0:
        msg = "Lambda should be greater than 1 to ensure error suppression"
        raise ValueError(msg)

    required_lep = 1.0 / num_quops
    distance = _equal_or_less_brute_force_search(
        lambda x: _calculate_lep(lambda0, lambda_, x, x),
        required_lep,
        minimum=1,
        maximum=max_distance,
        step=2,  # seek for odd solutions
    )
    if distance is None:
        text = (
            f"Could not find a solution between [1, {max_distance}] "
            "for LEP(distance) < 1 / QuOps. Increase max_distance."
        )
        raise ValueError(text)
    return distance
