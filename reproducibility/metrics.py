from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalProfile:
    persistence: float
    thgee: float
    reemergence: float
    reactivation_windows: tuple[int, ...]


def _validate_trace(trace: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=float)

    if values.ndim != 1:
        raise ValueError("The engagement trace must be one-dimensional.")

    if values.size == 0:
        raise ValueError("The engagement trace cannot be empty.")

    if np.any(~np.isfinite(values)):
        raise ValueError("The engagement trace contains invalid values.")

    if np.any(values < 0):
        raise ValueError("Engagement values cannot be negative.")

    return values


def geometric_persistence(trace: list[float] | np.ndarray) -> float:
    """
    Shifted geometric mean.

    The result is invariant to the order of the windows.
    """
    values = _validate_trace(trace)
    return float(np.exp(np.mean(np.log1p(values))) - 1.0)


def recency_weights(number_of_windows: int, gamma: float) -> np.ndarray:
    if number_of_windows < 1:
        raise ValueError("number_of_windows must be at least 1.")

    if gamma < 0:
        raise ValueError("gamma cannot be negative.")

    positions = np.arange(number_of_windows)
    age = number_of_windows - 1 - positions

    raw_weights = np.exp(-gamma * age)
    return raw_weights / raw_weights.sum()


def thgee(
    trace: list[float] | np.ndarray,
    gamma: float,
) -> float:
    """
    Recency-weighted shifted geometric mean.
    """
    values = _validate_trace(trace)
    weights = recency_weights(len(values), gamma)

    return float(
        np.exp(np.sum(weights * np.log1p(values))) - 1.0
    )


def reemergence_index(
    trace: list[float] | np.ndarray,
    activity_threshold: float,
    dormancy_windows: int,
    epsilon: float = 1e-12,
) -> tuple[float, tuple[int, ...]]:
    """
    Measures the share of logged engagement occurring at valid
    reactivation points.

    A valid reactivation requires:
    1. activity before dormancy;
    2. at least dormancy_windows consecutive inactive windows;
    3. activity in the current window.
    """
    values = _validate_trace(trace)

    if activity_threshold < 0:
        raise ValueError("activity_threshold cannot be negative.")

    if dormancy_windows < 1:
        raise ValueError("dormancy_windows must be at least 1.")

    active = values >= activity_threshold
    reactivation_positions: list[int] = []

    for current in range(dormancy_windows + 1, len(values)):
        dormant_start = current - dormancy_windows
        prior_end = dormant_start

        current_is_active = bool(active[current])
        dormant_period = bool(np.all(~active[dormant_start:current]))
        active_before_dormancy = bool(np.any(active[:prior_end]))

        if current_is_active and dormant_period and active_before_dormancy:
            reactivation_positions.append(current)

    total_logged_engagement = float(np.log1p(values).sum())

    reactivation_engagement = float(
        sum(np.log1p(values[position]) for position in reactivation_positions)
    )

    score = reactivation_engagement / (total_logged_engagement + epsilon)

    return float(score), tuple(reactivation_positions)


def calculate_profile(
    trace: list[float] | np.ndarray,
    gamma: float,
    activity_threshold: float,
    dormancy_windows: int,
) -> TemporalProfile:
    persistence_score = geometric_persistence(trace)
    thgee_score = thgee(trace, gamma)

    reemergence_score, positions = reemergence_index(
        trace,
        activity_threshold=activity_threshold,
        dormancy_windows=dormancy_windows,
    )

    return TemporalProfile(
        persistence=persistence_score,
        thgee=thgee_score,
        reemergence=reemergence_score,
        reactivation_windows=positions,
    )