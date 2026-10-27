"""Minimal notebook-first implementation of the Remark 3 experiments.

The public API is class based and follows the experimental workflow:

``StudySetup`` -> ``ReferenceSampling`` -> either
``DownsamplingExperiment`` or ``EqualLengthExperiment`` ->
``DistributionComparison`` -> ``ModelRecovery``.

Importing this module does not run experiments, draw figures, or write files.
The notebook owns all loops, tables, and visualizations.
The empirical edge must be supplied explicitly as ``edge=(n_ic, n_points)``
when constructing ``StudySetup``.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.stats import qmc


# The repository modules use bare intra-directory imports.  Load the versions
# beside this file in a temporary context, then restore the notebook's import
# state so another same-named Basis.py does not leak into this experiment.
_MODULE_DIR = Path(__file__).resolve().parent
_LOCAL_NAMES = (
    "Base_test",
    "dae_finder",
    "Definitions",
    "Basis",
    "Multi",
    "Comparison",
    "PolyConvert",
    "Recover_Model",
)
_MISSING = object()
_old_modules = {name: sys.modules.get(name, _MISSING) for name in _LOCAL_NAMES}
_old_path = list(sys.path)
if str(_MODULE_DIR) in sys.path:
    sys.path.remove(str(_MODULE_DIR))
sys.path.insert(0, str(_MODULE_DIR))
for _name in _LOCAL_NAMES:
    sys.modules.pop(_name, None)
try:
    import Base_test as _base_test
    import Recover_Model as _recover_model
finally:
    for _name in _LOCAL_NAMES:
        sys.modules.pop(_name, None)
    for _name, _module in _old_modules.items():
        if _module is not _MISSING:
            sys.modules[_name] = _module
    sys.path[:] = _old_path

CRN = _base_test.CRN
Lotka_Volterra = _base_test.Lotka_Volterra
Recover_Model_sindy = _recover_model.Recover_Model_sindy
OrthogonalLibrary = _recover_model.OrthogonalLibrary


Exponent = tuple[int, ...]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    state_names: tuple[str, ...]
    internal_state_names: tuple[str, ...]
    parameters: Mapping[str, float]
    original_initial: tuple[float, ...]
    original_times: tuple[float, ...]
    true_support: tuple[frozenset[Exponent], ...]


@dataclass(frozen=True)
class StudyConfig:
    model: str
    distribution: str
    basis: str
    edge: tuple[int, int]
    reference_end: float
    degree: int
    threshold: float
    after_threshold: float
    joint_bins: int
    kl_pseudocount: float
    kl_null_repeats: int
    noise_level: float
    reference_note: str

    @property
    def reference_n_ic(self) -> int:
        return self.edge[0]

    @property
    def reference_n_points(self) -> int:
        return self.edge[1]

    @property
    def reference_n_observations(self) -> int:
        """Effective observation count in the empirical edge/reference."""
        n_ic, n_points = self.edge
        return n_ic * (n_points - 2)

    @property
    def dt(self) -> float:
        return self.reference_end / (self.edge[1] - 1)


@dataclass(frozen=True)
class Study:
    study_id: str
    model: ModelSpec
    config: StudyConfig
    original: pd.DataFrame
    original_norm: np.ndarray
    lower: pd.Series
    upper: pd.Series


@dataclass
class TrajectoryBatch:
    x_raw: np.ndarray
    x_norm: np.ndarray
    xdot_norm: np.ndarray
    trajectory_id: np.ndarray
    local_time: np.ndarray

    def subset(self, indices: Sequence[int]) -> "TrajectoryBatch":
        index = np.asarray(indices, dtype=int)
        return TrajectoryBatch(
            x_raw=self.x_raw[index],
            x_norm=self.x_norm[index],
            xdot_norm=self.xdot_norm[index],
            trajectory_id=self.trajectory_id[index],
            local_time=self.local_time[index],
        )


@dataclass
class ExperimentRun:
    run_id: str
    kind: str
    study_id: str
    source_reference_id: str
    seed: int
    n_initial_conditions: int
    batch: TrajectoryBatch
    target_n_observations: int | None = None
    initial_raw: np.ndarray | None = None
    initial_norm: np.ndarray | None = None
    allocation: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    selection_seed: int | None = None

    @property
    def n_observations(self) -> int:
        return int(len(self.batch.x_norm))

    @property
    def actual_n_observations(self) -> int:
        return self.n_observations

    @property
    def n_observations_delta(self) -> int | None:
        if self.target_n_observations is None:
            return None
        return self.n_observations - int(self.target_n_observations)

    def to_frame(self, study: Study) -> pd.DataFrame:
        _require_same_study(study, self)
        frame = pd.DataFrame(
            {
                "run_id": self.run_id,
                "target_n_observations": self.target_n_observations,
                "actual_n_observations": self.n_observations,
                "n_observations_delta": self.n_observations_delta,
                "trajectory_id": self.batch.trajectory_id,
                "local_time": self.batch.local_time,
            }
        )
        for column, state in enumerate(study.model.state_names):
            frame[f"{state}_raw"] = self.batch.x_raw[:, column]
            frame[f"{state}_norm"] = self.batch.x_norm[:, column]
            frame[f"d{state}_norm_dt"] = self.batch.xdot_norm[:, column]
        return frame

    def design_frame(self, study: Study) -> pd.DataFrame:
        _require_same_study(study, self)
        if self.allocation is None:
            if self.source_indices is None:
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "run_id": self.run_id,
                    "target_n_observations": self.target_n_observations,
                    "actual_n_observations": self.n_observations,
                    "n_observations_delta": self.n_observations_delta,
                    "selected_reference_index": self.source_indices,
                }
            )

        records: list[dict[str, object]] = []
        for trajectory_id, retained in enumerate(self.allocation):
            record: dict[str, object] = {
                "run_id": self.run_id,
                "trajectory_id": trajectory_id,
                "target_n_observations": self.target_n_observations,
                "actual_n_observations": self.n_observations,
                "n_observations_delta": self.n_observations_delta,
                "retained_points": int(retained),
                "n_step": int(retained + 2),
                "end_time": float((retained + 1) * study.config.dt),
            }
            if self.initial_raw is not None:
                for column, state in enumerate(study.model.state_names):
                    record[f"initial_{state}_raw"] = self.initial_raw[
                        trajectory_id, column
                    ]
                    record[f"initial_{state}_norm"] = self.initial_norm[
                        trajectory_id, column
                    ]
            records.append(record)
        return pd.DataFrame(records)


@dataclass
class DistributionResult:
    run_id: str
    study_id: str
    reference_id: str
    metrics: dict[str, float | str]
    case_probability: np.ndarray
    reference_probability: np.ndarray
    null_values: np.ndarray


@dataclass
class RecoveryResult:
    run_id: str
    study_id: str
    n_observations: int
    target_n_observations: int | None
    n_observations_delta: int | None
    recovered_model: Any
    model_expression: pd.DataFrame
    coefficients: Mapping[str, Mapping[str, float]]
    predicted_support: tuple[frozenset[Exponent], ...]
    metrics: dict[str, float]
    terms: pd.DataFrame

    @property
    def actual_n_observations(self) -> int:
        return self.n_observations


@dataclass(frozen=True)
class _PairDefaults:
    end_time: float
    threshold: float
    after_threshold: float
    joint_bins: int
    note: str


_PAIR_DEFAULTS = {
    ("lv", "uniform"): _PairDefaults(
        end_time=0.05,
        threshold=0.01,
        after_threshold=0.08,
        joint_bins=3,
        note="LV/uniform non-edge defaults; edge supplied by the caller.",
    ),
    ("crn", "arcsine"): _PairDefaults(
        end_time=0.01,
        threshold=0.1,
        after_threshold=0.1,
        joint_bins=2,
        note="CRN/arcsine non-edge defaults; edge supplied by the caller.",
    ),
}

_BASIS = {"uniform": "Legendre", "arcsine": "Chebyshev"}


def _model_spec(key: str) -> ModelSpec:
    if key == "lv":
        return ModelSpec(
            key="lv",
            display_name="Lotka--Volterra",
            state_names=("x1", "x2", "x3"),
            internal_state_names=("x", "y1", "y2"),
            parameters={
                "r": 0.2,
                "K": 1.7,
                "a1": 0.1,
                "a2": 0.2,
                "m1": 0.1,
                "m2": 0.1,
                "epsilon1": 1.0,
                "epsilon2": 2.0,
            },
            original_initial=(1.7, 1.0, 1.7),
            original_times=tuple(np.linspace(0.0, 20.0, 21)),
            true_support=(
                frozenset({(1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1)}),
                frozenset({(0, 1, 0), (1, 1, 0)}),
                frozenset({(0, 0, 1), (1, 0, 1)}),
            ),
        )

    if key == "crn":
        grid = np.array(
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 47, 51],
            dtype=float,
        )
        interaction = (1, 1, 0, 0)
        complex_term = (0, 0, 1, 0)
        return ModelSpec(
            key="crn",
            display_name="Chemical reaction network",
            # x1=S, x2=E, x3=ES, x4=P.
            state_names=("x1", "x2", "x3", "x4"),
            internal_state_names=("S", "E", "ES", "P"),
            parameters={"k": 2.12456, "kr": 1.10124, "kcat": 1.5093},
            original_initial=(2.5, 2.0, 1.0, 0.0),
            original_times=tuple((10.0 / 51.0) * grid),
            true_support=(
                frozenset({interaction, complex_term}),
                frozenset({interaction, complex_term}),
                frozenset({interaction, complex_term}),
                frozenset({complex_term}),
            ),
        )
    raise ValueError(f"Unsupported model: {key!r}")


def _span(lower: pd.Series, upper: pd.Series) -> np.ndarray:
    span = upper.to_numpy(dtype=float) - lower.to_numpy(dtype=float)
    if np.any(~np.isfinite(span)) or np.any(span <= 0):
        raise ValueError("Every normalization bound must be finite and satisfy U>L.")
    return span


def _normalize(x: np.ndarray, lower: pd.Series, upper: pd.Series) -> np.ndarray:
    return 2.0 * (np.asarray(x, float) - lower.to_numpy(float)) / _span(lower, upper) - 1.0


def _denormalize(z: np.ndarray, lower: pd.Series, upper: pd.Series) -> np.ndarray:
    return lower.to_numpy(float) + 0.5 * (np.asarray(z, float) + 1.0) * _span(
        lower, upper
    )


def _simulate_physical(
    model: ModelSpec,
    initial: Sequence[float],
    times: Sequence[float],
) -> np.ndarray:
    time = np.asarray(times, dtype=float)
    if len(time) < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("A trajectory needs at least three increasing time points.")

    if model.key == "lv":
        simulator = Lotka_Volterra(
            params=dict(model.parameters),
            t_span=(float(time[0]), float(time[-1])),
            t_eval=time,
            z0=list(np.asarray(initial, float)),
            noise_level=None,
        )
    else:
        simulator = CRN(
            k_rates=dict(model.parameters),
            init_cond=list(np.asarray(initial, float)),
            solvedT=time,
            noise_level=None,
        )

    values = simulator.data_sim.loc[
        :, list(model.internal_state_names)
    ].to_numpy(float)
    expected = (len(time), len(model.state_names))
    if values.shape != expected or np.any(~np.isfinite(values)):
        raise RuntimeError(f"Simulator returned invalid data: {values.shape}, expected {expected}.")
    return values


def _original_data(
    model: ModelSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    time = np.asarray(model.original_times, float)
    values = _simulate_physical(model, model.original_initial, time)
    frame = pd.DataFrame(values, columns=model.state_names)
    frame.insert(0, "time", time)
    lower = frame.loc[:, model.state_names].min()
    upper = frame.loc[:, model.state_names].max()
    normalized = _normalize(values, lower, upper)
    return frame, lower, upper, normalized


def _require_same_study(study: Study, run: ExperimentRun) -> None:
    if run.study_id != study.study_id:
        raise ValueError("This run was created from a different study configuration.")


def _require_reference(study: Study, reference: ExperimentRun) -> None:
    _require_same_study(study, reference)
    if reference.kind != "reference":
        raise ValueError("Expected a reference produced by ReferenceSampling.sample().")
    if reference.source_reference_id != reference.run_id:
        raise ValueError("The reference identity is inconsistent.")


def _sobol_u01(n: int, dimension: int, seed: int) -> np.ndarray:
    if n <= 0 or seed < 0:
        raise ValueError("n must be positive and seed must be non-negative.")
    power = int(math.ceil(math.log2(max(2, n))))
    points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)[:n]
    shift = np.random.default_rng(seed + 12345).random(dimension)
    return (points + shift) % 1.0


def _initial_conditions(
    study: Study,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    u = _sobol_u01(n, len(study.model.state_names), seed)
    if study.config.distribution == "uniform":
        z = 2.0 * u - 1.0
    else:
        z = np.cos(np.pi * u)
    # Clip only sampled ICs away from exact polynomial endpoints.  Simulated
    # trajectory points are never clipped; overflow is retained for KL.
    z = np.clip(z, -1.0 + 1e-7, 1.0 - 1e-7)
    return _denormalize(z, study.lower, study.upper), z


def _nearest_equal_allocation(
    target_n_observations: int,
    n_ic: int,
) -> np.ndarray:
    """Use equal trajectory lengths at the closest feasible total sample size.

    If the lower and upper multiples of ``n_ic`` are equally close to the
    target, the lower (conservative) total is selected.
    """
    if target_n_observations <= 0 or n_ic <= 0:
        raise ValueError("target_n_observations and n_ic must be positive.")

    quotient, remainder = divmod(target_n_observations, n_ic)
    retained_per_trajectory = max(
        1,
        quotient if 2 * remainder <= n_ic else quotient + 1,
    )
    allocation = np.full(n_ic, retained_per_trajectory, dtype=int)
    if (
        np.any(allocation < 1)
        or allocation.min() != allocation.max()
        or allocation.sum() != n_ic * retained_per_trajectory
    ):
        raise AssertionError("Equal-length trajectory allocation failed.")
    return allocation


def _simulate_batch(
    study: Study,
    initial_conditions: np.ndarray,
    allocation: Sequence[int],
    seed: int,
) -> TrajectoryBatch:
    initial = np.asarray(initial_conditions, float)
    retained_counts = np.asarray(allocation, int)
    if len(initial) != len(retained_counts) or np.any(retained_counts < 1):
        raise ValueError("Each initial condition needs a positive retained count.")

    raw_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    derivative_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    differentiator = ps.FiniteDifference(order=2)

    for trajectory_id, (x0, retained) in enumerate(zip(initial, retained_counts)):
        times = np.arange(int(retained) + 2, dtype=float) * study.config.dt
        raw = _simulate_physical(study.model, x0, times)
        if study.config.noise_level > 0:
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, trajectory_id, 104729])
            )
            raw = raw + rng.normal(0.0, np.abs(raw) * study.config.noise_level)
        normalized = _normalize(raw, study.lower, study.upper)
        derivative = np.asarray(differentiator(normalized, times), float)

        # Differentiate the complete local trajectory, then discard its own
        # two endpoint estimates.
        keep = slice(1, -1)
        raw_parts.append(raw[keep])
        norm_parts.append(normalized[keep])
        derivative_parts.append(derivative[keep])
        trajectory_parts.append(np.full(int(retained), trajectory_id, int))
        time_parts.append(times[keep])

    batch = TrajectoryBatch(
        x_raw=np.vstack(raw_parts),
        x_norm=np.vstack(norm_parts),
        xdot_norm=np.vstack(derivative_parts),
        trajectory_id=np.concatenate(trajectory_parts),
        local_time=np.concatenate(time_parts),
    )
    if len(batch.x_norm) != int(retained_counts.sum()):
        raise AssertionError("The simulated batch violated its allocation total.")
    if np.any(~np.isfinite(batch.x_norm)) or np.any(~np.isfinite(batch.xdot_norm)):
        raise RuntimeError("The simulated states or derivatives contain non-finite values.")
    return batch

def _target_cdf(z: np.ndarray, distribution: str) -> np.ndarray:
    if distribution == "uniform":
        return 0.5 * (z + 1.0) 
    return 0.5 + np.arcsin(np.clip(z, -1.0, 1.0)) / np.pi


def _joint_codes(z: np.ndarray, distribution: str, bins: int) -> tuple[np.ndarray, int]:
    values = np.asarray(z, float)
    if values.ndim != 2:
        raise ValueError("State data must be a two-dimensional array.")
    dimension = values.shape[1]
    inside = np.all((values >= -1.0) & (values <= 1.0), axis=1)
    n_inside_cells = bins**dimension
    codes = np.full(len(values), n_inside_cells, int)  # one joint overflow cell
    if np.any(inside):
        cell = np.floor(_target_cdf(values[inside], distribution) * bins).astype(int)
        cell = np.clip(cell, 0, bins - 1)
        codes[inside] = np.ravel_multi_index(cell.T, (bins,) * dimension)
    return codes, n_inside_cells + 1


def _probability(codes: np.ndarray, n_cells: int, alpha: float) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_cells).astype(float)
    return (counts + alpha) / (counts.sum() + alpha * n_cells)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def joint_kl_theoretical_target_to_sample(
    z: np.ndarray,
    distribution: str,
    bins: int,
    pseudocount: float = 0.5,
) -> float:
    """Estimate joint ``D_KL(P_target || P_sample)`` on a fixed grid.

    The normalized state coordinates are first mapped through the requested
    target CDF.  Under the ideal independent product target (uniform or
    arcsine), the transformed coordinates are jointly uniform on ``[0, 1]^d``;
    consequently every cell of the equal-width joint grid has probability
    ``1 / bins**d``.

    The empirical sample probabilities receive ``pseudocount`` smoothing, so
    every denominator cell in ``P_target / P_sample`` is positive. Values
    outside ``[-1, 1]`` are clipped to the target support before binning; their
    frequency remains available separately through
    ``DistributionComparison.outside_fraction``.
    """
    values = np.asarray(z, dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("State data must be a nonempty two-dimensional array.")
    if np.any(~np.isfinite(values)):
        raise ValueError("State data contain non-finite values.")
    if distribution not in _BASIS:
        raise ValueError("distribution must be 'uniform' or 'arcsine'.")
    if (
        isinstance(bins, (bool, np.bool_))
        or not isinstance(bins, (int, np.integer))
        or int(bins) < 2
    ):
        raise ValueError("bins must be an integer of at least two.")
    alpha = float(pseudocount)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("pseudocount must be positive and finite.")

    target_quantiles = np.clip(
        _target_cdf(np.clip(values, -1.0, 1.0), distribution),
        0.0,
        1.0,
    )
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    counts = np.histogramdd(
        target_quantiles,
        bins=[edges] * values.shape[1],
    )[0].ravel()
    empirical = (counts + alpha) / (counts.sum() + alpha * len(counts))
    theoretical = np.full(len(counts), 1.0 / len(counts), dtype=float)
    value = _kl(theoretical, empirical)
    if value < -1e-12 or not np.isfinite(value):
        raise FloatingPointError(f"Invalid theoretical-target KL value: {value}")
    return max(0.0, value)


def _null_kl(
    reference_codes: np.ndarray,
    n_case: int,
    n_cells: int,
    alpha: float,
    repeats: int,
    seed: int,
    without_replacement: bool,
) -> np.ndarray:
    if repeats <= 0:
        return np.array([], float)
    if without_replacement and n_case > len(reference_codes):
        raise ValueError("Cannot draw a larger null subset without replacement.")
    rng = np.random.default_rng(seed)
    reference_probability = _probability(reference_codes, n_cells, alpha)
    values = np.empty(repeats, float)
    for repeat in range(repeats):
        indices = rng.choice(
            len(reference_codes),
            size=n_case,
            replace=not without_replacement,
        )
        case_probability = _probability(reference_codes[indices], n_cells, alpha)
        values[repeat] = _kl(reference_probability, case_probability)
    return values


def _term_to_exponent(term: str, state_names: Sequence[str]) -> Exponent:
    text = str(term).strip().replace("*", " ")
    if text in {"", "1"}:
        return (0,) * len(state_names)
    powers = {state: 0 for state in state_names}
    for token in text.split():
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\^(\d+))?", token)
        if match is None or match.group(1) not in powers:
            raise ValueError(f"Cannot parse recovered term {term!r}.")
        powers[match.group(1)] += int(match.group(2) or 1)
    return tuple(powers[state] for state in state_names)


def _exponent_to_term(exponent: Exponent, state_names: Sequence[str]) -> str:
    factors = []
    for state, power in zip(state_names, exponent):
        if power == 1:
            factors.append(state)
        elif power > 1:
            factors.append(f"{state}^{power}")
    return " ".join(factors) if factors else "1"


def _support_metrics(
    predicted: Sequence[set[Exponent]],
    truth: Sequence[frozenset[Exponent]],
) -> dict[str, float]:
    if len(predicted) != len(truth):
        raise ValueError("Predicted and true models have different equation counts.")
    tp = fp = fn = wrong_equations = 0
    for predicted_terms, true_terms in zip(predicted, truth):
        predicted_set = set(predicted_terms)
        true_set = set(true_terms)
        tp += len(predicted_set & true_set)
        fp += len(predicted_set - true_set)
        fn += len(true_set - predicted_set)
        wrong_equations += int(predicted_set != true_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_terms": float(tp),
        "false_positive_terms": float(fp),
        "false_negative_terms": float(fn),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "wrong_equations": float(wrong_equations),
        "exact_support": float(wrong_equations == 0),
    }

# Check properties such as "joint_bins", since they are not needed once the way to compute KL changes.  The caller must ensure that the study configuration is valid before calling this function.
class StudySetup:
    """Step 1: prepare the model, original data, and fixed normalization bounds."""

    def __init__(
        self,
        model: str = "lv",
        distribution: str = "uniform",
        *,
        edge: tuple[int, int],
        basis: str | None = None,
        reference_end: float | None = None,
        degree: int = 5,
        threshold: float | None = None,
        after_threshold: float | None = None,
        joint_bins: int | None = None,
        kl_pseudocount: float = 0.5,
        kl_null_repeats: int = 500,
        noise_level: float = 0.0,
    ) -> None:
        self.model_key = model.lower()
        self.distribution = distribution.lower()
        self.edge = edge
        self.basis = basis
        self.reference_end = reference_end
        self.degree = degree
        self.threshold = threshold
        self.after_threshold = after_threshold
        self.joint_bins = joint_bins
        self.kl_pseudocount = kl_pseudocount
        self.kl_null_repeats = kl_null_repeats
        self.noise_level = noise_level

    def prepare(self) -> Study:
        if self.distribution not in _BASIS:
            raise ValueError("distribution must be 'uniform' or 'arcsine'.")
        defaults = _PAIR_DEFAULTS.get((self.model_key, self.distribution))

        try:
            edge_values = tuple(self.edge)
        except TypeError as error:
            raise ValueError("edge must be a pair (n_ic, n_points).") from error
        if len(edge_values) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in edge_values
        ):
            raise ValueError("edge must contain exactly two integers: (n_ic, n_points).")
        edge = (int(edge_values[0]), int(edge_values[1]))

        def choose(value: object | None, attribute: str, name: str) -> object:
            if value is not None:
                return value
            if defaults is None:
                raise ValueError(
                    f"No built-in non-edge defaults for "
                    f"({self.model_key}, {self.distribution}); "
                    f"provide {name}."
                )
            return getattr(defaults, attribute)

        config = StudyConfig(
            model=self.model_key,
            distribution=self.distribution,
            basis=self.basis or _BASIS[self.distribution],
            edge=edge,
            reference_end=float(
                choose(self.reference_end, "end_time", "reference_end")
            ),
            degree=int(self.degree),
            threshold=float(choose(self.threshold, "threshold", "threshold")),
            after_threshold=float(
                choose(self.after_threshold, "after_threshold", "after_threshold")
            ),
            joint_bins=int(
                choose(self.joint_bins, "joint_bins", "joint_bins")
            ),
            kl_pseudocount=float(self.kl_pseudocount),
            kl_null_repeats=int(self.kl_null_repeats),
            noise_level=float(self.noise_level),
            reference_note=(
                defaults.note
                if defaults is not None
                else "User-specified non-edge defaults and edge."
            ),
        )
        if config.basis not in {"Legendre", "Chebyshev"}:
            raise ValueError("basis must be 'Legendre' or 'Chebyshev'.")
        finite_values = (
            config.reference_end,
            config.threshold,
            config.after_threshold,
            config.kl_pseudocount,
            config.noise_level,
        )
        if (
            not all(np.isfinite(value) for value in finite_values)
            or config.reference_n_ic < 1
            or config.reference_n_points < 3
            or config.reference_end <= 0
            or config.degree < 1
            or config.threshold < 0
            or config.after_threshold < 0
            or config.joint_bins < 2
            or config.kl_pseudocount <= 0
            or config.kl_null_repeats < 0
            or config.noise_level < 0
        ):
            raise ValueError("Study parameters are outside their valid ranges.")

        model = _model_spec(config.model)
        original, lower, upper, original_norm = _original_data(model)
        study_id = repr(
            (
                config,
                tuple(lower.to_numpy(float)),
                tuple(upper.to_numpy(float)),
            )
        )
        return Study(
            study_id=study_id,
            model=model,
            config=config,
            original=original,
            original_norm=original_norm,
            lower=lower,
            upper=upper,
        )


class ReferenceSampling:
    """Step 2: generate data for the caller-supplied empirical edge."""

    def __init__(self, study: Study, seed: int = 0) -> None:
        self.study = study
        self.seed = int(seed)

    def sample(self) -> ExperimentRun:
        config = self.study.config
        initial_raw, initial_norm = _initial_conditions(
            self.study, config.reference_n_ic, self.seed
        )
        allocation = np.full(
            config.reference_n_ic,
            config.reference_n_points - 2,
            dtype=int,
        )
        batch = _simulate_batch(self.study, initial_raw, allocation, self.seed)
        run_id = (
            f"{config.model}_{config.distribution}_"
            f"edge{config.reference_n_ic}x{config.reference_n_points}_"
            f"seed{self.seed:04d}_reference"
        )
        return ExperimentRun(
            run_id=run_id,
            kind="reference",
            study_id=self.study.study_id,
            source_reference_id=run_id,
            seed=self.seed,
            n_initial_conditions=config.reference_n_ic,
            batch=batch,
            target_n_observations=config.reference_n_observations,
            initial_raw=initial_raw,
            initial_norm=initial_norm,
            allocation=allocation,
        )

class DownsamplingExperiment:
    """Step 3A: build random or equal-length-candidate-matched subsets."""

    def __init__(self, study: Study, reference: ExperimentRun) -> None:
        _require_reference(study, reference)
        self.study = study
        self.reference = reference

    def sample(
        self,
        n_observations: int,
        *,
        repeat: int = 0,
        seed: int | None = None,
    ) -> ExperimentRun:
        n = int(n_observations)
        if int(repeat) < 0:
            raise ValueError("repeat must be non-negative.")
        if not 0 < n < self.reference.n_observations:
            raise ValueError("n_observations must lie strictly below the reference size.")
        if seed is None:
            seed = int(
                np.random.SeedSequence(
                    [self.reference.seed, n, int(repeat), 1]
                ).generate_state(1)[0]
            )
        if int(seed) < 0:
            raise ValueError("seed must be non-negative.")
        indices = np.random.default_rng(seed).choice(
            self.reference.n_observations,
            n,
            replace=False,
        )
        return ExperimentRun(
            run_id=(
                f"{self.reference.run_id}_downsample_n{n}_"
                f"r{int(repeat):03d}_s{seed}"
            ),
            kind="downsample",
            study_id=self.study.study_id,
            source_reference_id=self.reference.run_id,
            seed=self.reference.seed,
            n_initial_conditions=self.reference.n_initial_conditions,
            batch=self.reference.batch.subset(indices),
            target_n_observations=n,
            source_indices=np.asarray(indices, int),
            selection_seed=seed,
        )

    def matched_sample(
        self,
        candidate: ExperimentRun,
        *,
        repeat: int = 0,
        seed: int | None = None,
    ) -> ExperimentRun:
        """Return the reference control matched to an equal-length run's actual N.

        If the candidate has fewer observations than the reference, this
        delegates to :meth:`sample` and retains exactly the candidate's actual
        number of observations.  If the counts are equal, the reference itself
        is already the matched control.  A candidate larger than the reference
        cannot be constructed by downsampling and is rejected explicitly.

        Matching controls only the observation count.  Candidates with the
        same actual count and repeat intentionally reuse the same reference
        subset, even if their IC counts differ.
        """
        _require_same_study(self.study, candidate)
        if candidate.kind not in {"fewer_longer", "more_shorter"}:
            raise ValueError(
                "matched_sample expects an equal-length run produced by "
                "EqualLengthExperiment.simulate()."
            )
        if candidate.source_reference_id != self.reference.run_id:
            raise ValueError(
                "The equal-length candidate came from a different reference."
            )
        if int(repeat) < 0:
            raise ValueError("repeat must be non-negative.")
        if seed is not None and int(seed) < 0:
            raise ValueError("seed must be non-negative.")

        candidate_size = candidate.n_observations
        reference_size = self.reference.n_observations
        if candidate_size < reference_size:
            return self.sample(
                n_observations=candidate_size,
                repeat=repeat,
                seed=seed,
            )
        if candidate_size == reference_size:
            return self.reference
        raise ValueError(
            f"The candidate has {candidate_size} observations but the reference "
            f"has only {reference_size}; a matched control cannot be created "
            "by downsampling."
        )


class EqualLengthExperiment:
    """Step 3B: vary IC count using equal-length trajectories near reference N.

    The default remains the original fewer-IC/longer-trajectory experiment.
    Callers may explicitly enable more-IC/shorter-trajectory candidates; those
    candidates extend the same seeded Sobol sequence used by the reference.
    """

    def __init__(self, study: Study, reference: ExperimentRun) -> None:
        _require_reference(study, reference)
        if reference.initial_raw is None or reference.initial_norm is None:
            raise ValueError("The reference does not retain its IC pool.")
        self.study = study
        self.reference = reference

    def simulate(
        self,
        n_ic: int,
        *,
        allow_more_initial_conditions: bool = False,
    ) -> ExperimentRun:
        n_ic = int(n_ic)
        reference_n_ic = self.reference.n_initial_conditions
        if n_ic <= 0:
            raise ValueError("n_ic must be positive.")
        if n_ic == reference_n_ic:
            raise ValueError(
                "n_ic equals the reference IC count; use the reference run "
                "itself as the baseline."
            )
        if n_ic > reference_n_ic and not allow_more_initial_conditions:
            raise ValueError(
                "n_ic is larger than the reference IC count. Pass "
                "allow_more_initial_conditions=True to run the explicit "
                "more-IC/shorter-trajectory design."
            )
        target_n_observations = self.reference.n_observations
        if n_ic > target_n_observations:
            raise ValueError(
                "n_ic cannot exceed the reference observation budget because "
                "each trajectory must retain at least one observation."
            )
        allocation = _nearest_equal_allocation(
            target_n_observations,
            n_ic,
        )
        retained_per_trajectory = int(allocation[0])
        actual_n_observations = int(allocation.sum())
        if n_ic < reference_n_ic:
            experiment_kind = "fewer_longer"
            run_id_design = "equal_length"
            initial_raw = self.reference.initial_raw[:n_ic].copy()
            initial_norm = self.reference.initial_norm[:n_ic].copy()
        else:
            experiment_kind = "more_shorter"
            run_id_design = experiment_kind
            initial_raw, initial_norm = _initial_conditions(
                self.study,
                n_ic,
                self.reference.seed,
            )
            # Sobol sampling is nested for a fixed seed: extending the pool
            # must preserve every IC already present in the reference.
            if not np.allclose(
                initial_norm[:reference_n_ic],
                self.reference.initial_norm,
                rtol=0.0,
                atol=1e-14,
            ):
                raise AssertionError(
                    "The extended Sobol IC pool does not preserve the "
                    "reference prefix."
                )
        batch = _simulate_batch(self.study, initial_raw, allocation, self.reference.seed)
        return ExperimentRun(
            run_id=(
                f"{self.reference.run_id}_{run_id_design}_ic{n_ic}_"
                f"k{retained_per_trajectory}_N{actual_n_observations}_"
                f"target{target_n_observations}"
            ),
            kind=experiment_kind,
            study_id=self.study.study_id,
            source_reference_id=self.reference.run_id,
            seed=self.reference.seed,
            n_initial_conditions=n_ic,
            batch=batch,
            target_n_observations=target_n_observations,
            initial_raw=initial_raw,
            initial_norm=initial_norm,
            allocation=allocation,
        )


class DistributionComparison:
    """Step 4: compute ``D_KL(P_reference || P_case)`` on a joint grid."""

    def __init__(self, study: Study, reference: ExperimentRun) -> None:
        _require_reference(study, reference)
        self.study = study
        self.reference = reference

    #way to calculate KL divergence
    def _calculate(
        self,
        run_id: str,
        case_z: np.ndarray,
        target_n_observations: int | None,
        *,
        null_method: str,
        null_seed: int,
    ) -> DistributionResult:
        config = self.study.config
        case_codes, n_cells = _joint_codes(
            case_z, config.distribution, config.joint_bins
        )
        reference_codes, reference_cells = _joint_codes(
            self.reference.batch.x_norm,
            config.distribution,
            config.joint_bins,
        )
        if n_cells != reference_cells:
            raise AssertionError("Case and reference do not share a KL grid.")
        case_probability = _probability(
            case_codes, n_cells, config.kl_pseudocount
        )
        reference_probability = _probability(
            reference_codes, n_cells, config.kl_pseudocount
        )
        observed = _kl(reference_probability, case_probability)

        if null_method == "self":
            null_values = np.array([], float)
            null_median = null_q025 = null_q975 = 0.0
            excess = 0.0
            pvalue = 1.0
        else:
            without_replacement = null_method == "without_replacement"
            null_values = _null_kl(
                reference_codes,
                len(case_codes),
                n_cells,
                config.kl_pseudocount,
                config.kl_null_repeats,
                null_seed,
                without_replacement,
            )
            if len(null_values):
                null_median = float(np.median(null_values))
                null_q025, null_q975 = map(
                    float, np.quantile(null_values, [0.025, 0.975])
                )
                excess = observed - null_median
                pvalue = float(
                    (1 + np.count_nonzero(null_values >= observed))
                    / (len(null_values) + 1)
                )
            else:
                null_median = null_q025 = null_q975 = excess = pvalue = np.nan

        actual_n_observations = int(len(case_z))
        if target_n_observations is None:
            target_n_observations = actual_n_observations
        target_n_observations = int(target_n_observations)
        outside = np.any((case_z < -1.0) | (case_z > 1.0), axis=1)
        metrics: dict[str, float | str] = {
            "n_case": float(actual_n_observations),
            "target_n_case": float(target_n_observations),
            "n_case_minus_target": float(
                actual_n_observations - target_n_observations
            ),
            "n_reference": float(self.reference.n_observations),
            "n_case_minus_reference": float(
                actual_n_observations - self.reference.n_observations
            ),
            "joint_kl_reference_to_case": observed,
            "outside_fraction": float(np.mean(outside)),
            "kl_null_method": null_method,
            "kl_null_median": null_median,
            "kl_null_q025": null_q025,
            "kl_null_q975": null_q975,
            "kl_excess": excess,
            "kl_empirical_pvalue": pvalue,
        }
        return DistributionResult(
            run_id=run_id,
            study_id=self.study.study_id,
            reference_id=self.reference.run_id,
            metrics=metrics,
            case_probability=case_probability,
            reference_probability=reference_probability,
            null_values=null_values,
        )

    def compare(
        self,
        candidate: ExperimentRun,
        *,
        null_seed: int | None = None,
        allow_unpaired: bool = False,
    ) -> DistributionResult:
        _require_same_study(self.study, candidate)
        same_reference = (
            candidate.kind == "reference"
            and candidate.run_id == self.reference.run_id
        )
        paired = same_reference or candidate.source_reference_id == self.reference.run_id
        if not paired and not allow_unpaired:
            raise ValueError(
                "The candidate came from a different reference; set "
                "allow_unpaired=True only for an intentional cross-reference comparison."
            )

        if same_reference:
            null_method = "self"
        elif candidate.kind == "downsample":
            null_method = "without_replacement"
        else:
            null_method = "bootstrap"
        if null_seed is None:
            null_seed = int(
                np.random.SeedSequence(
                    [self.reference.seed, candidate.seed, candidate.n_observations, 500009]
                ).generate_state(1)[0]
            )
        return self._calculate(
            candidate.run_id,
            candidate.batch.x_norm,
            candidate.target_n_observations,
            null_method=null_method,
            null_seed=int(null_seed),
        )

    def compare_original(self, *, null_seed: int | None = None) -> DistributionResult:
        if len(self.study.original_norm) > self.reference.n_observations:
            raise ValueError("Original data exceed the reference size for subset calibration.")
        if null_seed is None:
            null_seed = self.reference.seed + 400003
        return self._calculate(
            f"{self.reference.run_id}_original",
            self.study.original_norm,
            len(self.study.original_norm),
            null_method="without_replacement",
            null_seed=int(null_seed),
        )


def l2_scaled_library_diagnostics(
    study: Study,
    run: ExperimentRun,
) -> dict[str, float | int | bool | str]:
    """Condition diagnostics for the full library used by model recovery.

    The library is evaluated on the run's retained normalized state rows,
    exactly as in :class:`ModelRecovery`.  The current Remark 3 studies use
    Legendre or Chebyshev libraries; both include the constant feature because
    ``Recover_Model_sindy`` fits those orthogonal libraries with
    ``include_bias=True``.  Each completed library column is then divided by
    its Euclidean norm before the matrix 2-norm condition number is computed.

    Scaling is diagnostic only.  It does not modify ``run`` and it is not fed
    back into SINDy fitting.
    """
    _require_same_study(study, run)
    states = np.asarray(run.batch.x_norm, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("The retained normalized states must be a non-empty 2D array.")
    if states.shape[1] != len(study.model.state_names):
        raise ValueError("The retained state dimension does not match the study model.")
    if not np.all(np.isfinite(states)):
        raise ValueError("The retained normalized states contain non-finite values.")

    library = OrthogonalLibrary(
        degree=study.config.degree,
        method=study.config.basis,
        include_bias=True,
    )
    # Use the same list-of-trajectories API used by PySINDy for the fitted
    # orthogonal feature library.  Here the pooled retained rows constitute
    # the design matrix whose conditioning is being diagnosed.
    library.fit([states])
    theta = np.asarray(library.transform([states])[0], dtype=float)
    if theta.ndim != 2 or theta.shape[0] != states.shape[0]:
        raise AssertionError("The transformed library has an unexpected shape.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("The function library contains non-finite values.")

    column_norms = np.linalg.norm(theta, ord=2, axis=0)
    invalid = (~np.isfinite(column_norms)) | (column_norms <= 0.0)
    if np.any(invalid):
        bad_columns = np.flatnonzero(invalid).tolist()
        raise ValueError(f"Cannot L2-scale zero/non-finite library columns: {bad_columns}")
    theta_l2 = theta / column_norms
    if not np.allclose(
        np.linalg.norm(theta_l2, ord=2, axis=0),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("Library-column L2 normalization failed.")

    condition_number = float(np.linalg.cond(theta_l2, p=2))
    if np.isnan(condition_number) or condition_number < 1.0:
        raise ValueError("The L2-scaled library condition number is invalid.")
    log10_condition = (
        float(np.log10(condition_number))
        if np.isfinite(condition_number)
        else float("inf")
    )
    numerical_rank = int(np.linalg.matrix_rank(theta_l2))
    return {
        "library_basis": study.config.basis,
        "library_degree": int(study.config.degree),
        "library_includes_bias": True,
        "library_column_scaling": "unit_l2",
        "library_n_rows": int(theta_l2.shape[0]),
        "library_n_columns": int(theta_l2.shape[1]),
        "library_numerical_rank": numerical_rank,
        "library_full_column_rank": bool(numerical_rank == theta_l2.shape[1]),
        "library_column_nullity": int(theta_l2.shape[1] - numerical_rank),
        "library_is_wide": bool(theta_l2.shape[1] > theta_l2.shape[0]),
        "library_l2_condition_number": condition_number,
        "library_l2_log10_condition_number": log10_condition,
    }


class ModelRecovery:
    """Step 5: fit SINDy and score equation-term support."""

    def __init__(self, study: Study) -> None:
        self.study = study

    def fit(
        self,
        run: ExperimentRun,
        *,
        suppress_stdout: bool = True,
    ) -> RecoveryResult:
        _require_same_study(self.study, run)
        states = self.study.model.state_names
        data = pd.DataFrame(run.batch.x_norm, columns=states)
        # Xdot is explicit, so this pseudo-time only satisfies the wrapper's
        # monotonic-index requirement; no derivative crosses trajectories.
        data.insert(0, "time", np.arange(1, len(data) + 1, dtype=float))
        derivatives = pd.DataFrame(
            run.batch.xdot_norm,
            columns=[f"d{state}/dt" for state in states],
        )
        output_context = (
            contextlib.redirect_stdout(io.StringIO())
            if suppress_stdout
            else contextlib.nullcontext()
        )
        with output_context:
            recovered = Recover_Model_sindy(
                data=data,
                differential_order=2,
                poly_degree=self.study.config.degree,
                method=self.study.config.basis,
                threshold=self.study.config.threshold,
                Xdot=derivatives,
                debug=False,
                after_threshold=self.study.config.after_threshold,
                L=self.study.lower,
                U=self.study.upper,
                denormalize=True,
            )
            coefficient_map = recovered.get_final_coefficients(as_dataframe=False)

        predicted: list[set[Exponent]] = []
        coefficient_by_equation: list[dict[Exponent, float]] = []
        for state in states:
            terms: dict[Exponent, float] = {}
            for term, coefficient in coefficient_map.get(state, {}).items():
                coefficient = float(coefficient)
                if abs(coefficient) <= 1e-14:
                    continue
                terms[_term_to_exponent(term, states)] = coefficient
            predicted.append(set(terms))
            coefficient_by_equation.append(terms)

        metrics = _support_metrics(predicted, self.study.model.true_support)
        term_rows: list[dict[str, object]] = []
        for equation_index, state in enumerate(states):
            truth = set(self.study.model.true_support[equation_index])
            found = set(predicted[equation_index])
            for exponent in sorted(truth | found):
                status = (
                    "TP"
                    if exponent in truth and exponent in found
                    else "FP"
                    if exponent in found
                    else "FN"
                )
                term_rows.append(
                    {
                        "run_id": run.run_id,
                        "equation": f"d{state}/dt",
                        "exponent": " ".join(map(str, exponent)),
                        "term": _exponent_to_term(exponent, states),
                        "coefficient": coefficient_by_equation[equation_index].get(
                            exponent, np.nan
                        ),
                        "status": status,
                    }
                )

        return RecoveryResult(
            run_id=run.run_id,
            study_id=self.study.study_id,
            n_observations=run.n_observations,
            target_n_observations=run.target_n_observations,
            n_observations_delta=run.n_observations_delta,
            recovered_model=recovered,
            model_expression=recovered.model_expression.copy(),
            coefficients=coefficient_map,
            predicted_support=tuple(frozenset(terms) for terms in predicted),
            metrics=metrics,
            terms=pd.DataFrame(term_rows),
        )


__all__ = [
    "StudySetup",
    "ReferenceSampling",
    "DownsamplingExperiment",
    "EqualLengthExperiment",
    "DistributionComparison",
    "ModelRecovery",
    "Study",
    "ExperimentRun",
    "DistributionResult",
    "RecoveryResult",
    "joint_kl_theoretical_target_to_sample",
    "l2_scaled_library_diagnostics",
]
