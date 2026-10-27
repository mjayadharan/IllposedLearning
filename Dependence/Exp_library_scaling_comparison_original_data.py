"""Compare candidate-library scaling strategies for LV and CRN.

This script reconstructs the two experiments used by ``Plots-1_modify.ipynb``
and ``Plots_2.ipynb``:

1. the 2-norm condition number of each full polynomial library; and
2. the number of two-term and three-term combinations whose largest test-set
   R-squared is at least a chosen threshold.

For every library matrix Theta, two *column-scaling* modes are compared:

``existing``
    The intended existing pipeline: monomials are evaluated on physical state
    values, while Legendre and Chebyshev polynomials are evaluated on states
    mapped to [-1, 1].  A CRN notebook bug that fitted on normalized states but
    transformed raw states is deliberately fixed here.
``unit_l2``
    Divide each library column by its Euclidean (L2) norm.

The two requested outputs each contain four panels (LV/CRN by the two scaling
modes), making changes in absolute values and trends directly comparable.
By default the figures are shown and nothing is written to disk.  Use
``--save`` to save figures and tidy CSV files.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


# Keep Matplotlib and other package caches writable in headless/managed runs.
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "sindy_library_scaling_cache"
(RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


CONDITION_DEGREES = (2, 3, 4, 5)
COMBINATION_DEGREES = (1, 2, 3, 4, 5)
COMBINATION_SIZES = (2, 3)

BASIS_ORDER = ("Monomial", "Legendre", "Chebyshev")
BASIS_COLORS = {
    "Monomial": "#9B85AD",
    "Legendre": "#1F78B4",
    "Chebyshev": "#FF7F00",
}
BASIS_MARKERS = {"Monomial": "o", "Legendre": "s", "Chebyshev": "^"}

SCALING_ORDER = ("existing", "unit_l2")
SCALING_LABELS = {
    "existing": "No column scaling",
    "unit_l2": "Library-column L2 = 1",
}

SYSTEM_LABELS = {"lv": "Lotka–Volterra", "crn": "Chemical reaction network"}

CONDITION_YLABEL = r"$\log_{10}$(Cond Number)"
COMBINATION_YLABEL = r"$\log_{10}$(# ill-posed)"

PANEL_FIGURE_ALIASES = {
    "condition": "condition",
    "figure1": "condition",
    "1": "condition",
    "combinations": "combinations",
    "combination": "combinations",
    "figure2": "combinations",
    "2": "combinations",
}
PANEL_SCALING_ALIASES = {
    "existing": "existing",
    "l2": "unit_l2",
    "unit_l2": "unit_l2",
}


@dataclass(frozen=True)
class SystemData:
    key: str
    states: pd.DataFrame

    @property
    def label(self) -> str:
        return SYSTEM_LABELS[self.key]


def parse_panel_spec(value: str) -> tuple[str, str, str]:
    """Parse FIGURE:SYSTEM:SCALING for one standalone subplot."""
    parts = [part.strip().lower() for part in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "panel selection must be FIGURE:SYSTEM:SCALING, for example "
            "condition:lv:l2"
        )
    figure_token, system, scaling_token = parts
    figure = PANEL_FIGURE_ALIASES.get(figure_token)
    scaling = PANEL_SCALING_ALIASES.get(scaling_token)
    if figure is None:
        raise argparse.ArgumentTypeError(
            "FIGURE must be condition/figure1 or combinations/figure2"
        )
    if system not in SYSTEM_LABELS:
        raise argparse.ArgumentTypeError("SYSTEM must be lv or crn")
    if scaling is None:
        raise argparse.ArgumentTypeError(
            "SCALING must be existing or l2/unit_l2"
        )
    return figure, system, scaling


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no column scaling and unit-L2 column scaling "
            "for LV/CRN polynomial-library condition numbers and R2 counts."
        )
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.95,
        help="Ill-conditioning threshold for maximum test R2 (default: 0.95).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.33,
        help="Test fraction used by the existing combination analysis (default: 0.33).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=27,
        help="Train/test split seed used by the existing analysis (default: 27).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the two figures and their tidy CSV data.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open figure windows; normally used together with --save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "library_scaling_comparison",
        help="Output directory used with --save.",
    )
    parser.add_argument(
        "--save-panel",
        action="append",
        type=parse_panel_spec,
        default=[],
        metavar="FIGURE:SYSTEM:SCALING",
        help=(
            "Save one standalone subplot; repeat this option to select several. "
            "Examples: condition:lv:l2, combinations:crn:std."
        ),
    )
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for standalone subplots (default: "
            "<output-dir>/selected_panels)."
        ),
    )
    parser.add_argument(
        "--panel-format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="File format for standalone subplots (default: png).",
    )
    args = parser.parse_args(argv)
    if not (0.0 < args.r2_threshold <= 1.0):
        parser.error("--r2-threshold must lie in (0, 1].")
    if not (0.0 < args.test_size < 1.0):
        parser.error("--test-size must lie in (0, 1).")
    if args.random_state < 0:
        parser.error("--random-state must be non-negative.")
    if args.panel_output_dir is not None and not args.save_panel:
        parser.error("--panel-output-dir requires at least one --save-panel.")
    if args.no_show and not (args.save or args.save_panel):
        parser.error(
            "--no-show would hide both outputs; also pass --save or --save-panel."
        )
    return args


def simulate_systems() -> tuple[SystemData, SystemData]:
    """Reproduce the noiseless LV and CRN state data used by the notebooks."""
    from Base_test import CRN as CRNSimulator
    from Base_test import Lotka_Volterra

    lv_params = {
        "r": 0.2,
        "K": 1.7,
        "a1": 0.1,
        "a2": 0.2,
        "m1": 0.1,
        "m2": 0.1,
        "epsilon1": 1.0,
        "epsilon2": 2.0,
    }
    lv_times = np.linspace(0.0, 20.0, 21)
    lv_sim = Lotka_Volterra(
        lv_params,
        (0.0, 20.0),
        lv_times,
        [1.7, 1.0, 1.7],
        noise_level=None,
    ).data_sim
    lv_states = (
        lv_sim.rename(columns={"x": "x1", "y1": "x2", "y2": "x3"})
        .loc[:, ["x1", "x2", "x3"]]
        .copy()
    )

    crn_rates = {"k": 2.12456, "kr": 1.10124, "kcat": 1.5093}
    crn_time_indices = np.array(
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 47, 51],
        dtype=float,
    )
    crn_times = (10.0 / 51.0) * crn_time_indices
    crn_sim = CRNSimulator(
        crn_rates,
        [2.5, 2.0, 1.0, 0.0],
        crn_times,
        noise_level=None,
    ).data_sim
    crn_states = (
        crn_sim.rename(columns={"S": "x1", "E": "x2", "ES": "x3", "P": "x4"})
        .loc[:, ["x1", "x2", "x3", "x4"]]
        .copy()
    )

    for key, states in (("lv", lv_states), ("crn", crn_states)):
        values = states.to_numpy(dtype=float)
        if values.ndim != 2 or len(states) < 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid simulated state data for {key}.")
        if np.any(states.std(axis=0, ddof=1).to_numpy() <= 0.0):
            raise ValueError(f"A {key} state is constant and cannot be scaled.")

    return SystemData("lv", lv_states), SystemData("crn", crn_states)


def minmax_to_minus_one_one(states: pd.DataFrame) -> pd.DataFrame:
    lower = states.min(axis=0)
    span = states.max(axis=0) - lower
    if np.any(~np.isfinite(span.to_numpy())) or np.any(span.to_numpy() <= 0.0):
        raise ValueError("Every state must have a positive finite min-max span.")
    return 2.0 * (states - lower) / span - 1.0


def build_library(states: pd.DataFrame, basis: str, degree: int) -> pd.DataFrame:
    """Build one total-degree library, using identical data for fit/transform."""
    from Basis import OrthogonalLibrary
    from dae_finder import PolyFeatureMatrix

    if basis == "Monomial":
        library = PolyFeatureMatrix(degree).fit_transform(states)
        library = library.drop(columns=["1"])
    elif basis in ("Legendre", "Chebyshev"):
        transformer = OrthogonalLibrary(
            degree=degree,
            method=basis,
            include_bias=False,
            include_interaction=(degree > 1),
        )
        # The same values are intentionally supplied to fit and transform.
        transformer.fit(states)
        library = transformer.transform_to_dataframe(
            states, input_feature_names=list(states.columns)
        )
    else:
        raise ValueError(f"Unsupported basis: {basis!r}")

    expected_terms = math.comb(states.shape[1] + degree, degree) - 1
    if library.shape != (len(states), expected_terms):
        raise AssertionError(
            f"{basis} degree {degree}: expected {(len(states), expected_terms)}, "
            f"got {library.shape}."
        )
    if not np.all(np.isfinite(library.to_numpy(dtype=float))):
        raise ValueError(f"{basis} degree {degree} produced non-finite values.")
    return library


def intended_existing_library(
    states_raw: pd.DataFrame, basis: str, degree: int
) -> pd.DataFrame:
    """Match the intended current notebook convention and fix its CRN bug."""
    evaluation_states = (
        states_raw if basis == "Monomial" else minmax_to_minus_one_one(states_raw)
    )
    return build_library(evaluation_states, basis, degree)


def scale_library(library: pd.DataFrame, mode: str) -> pd.DataFrame:
    values = library.to_numpy(dtype=float)
    if mode == "existing":
        scaled = values.copy()
    elif mode == "unit_l2":
        divisors = np.linalg.norm(values, ord=2, axis=0)
        if np.any(~np.isfinite(divisors)) or np.any(divisors <= 0.0):
            raise ValueError("Cannot unit-L2 scale a zero/non-finite library column.")
        scaled = values / divisors
        if not np.allclose(
            np.linalg.norm(scaled, ord=2, axis=0), 1.0, rtol=1e-10, atol=1e-12
        ):
            raise AssertionError("Unit-L2 scaling failed.")
    else:
        raise ValueError(f"Unknown scaling mode: {mode!r}")

    if not np.all(np.isfinite(scaled)):
        raise ValueError(f"Scaling mode {mode!r} produced non-finite values.")
    return pd.DataFrame(scaled, index=library.index, columns=library.columns)


@lru_cache(maxsize=None)
def combination_index(n_features: int, size: int) -> np.ndarray:
    n_combinations = math.comb(n_features, size)
    flat = np.fromiter(
        (index for combo in combinations(range(n_features), size) for index in combo),
        dtype=np.int32,
        count=n_combinations * size,
    )
    return flat.reshape(n_combinations, size)


def _correlation_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    centered = x - mean
    std = centered.std(axis=0, ddof=0)
    safe_std = np.clip(std, 1e-15, None)
    z = centered / safe_std
    correlation = (z.T @ z) / len(z)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0), mean, std


def _test_r2_direction(
    correlation: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    predictor_1: np.ndarray,
    predictor_2: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized counterpart of Multi.Regression_analysis_advanced."""
    if predictor_2 is None:
        corr = correlation[target, predictor_1]
        beta_1 = corr * std[target] / (std[predictor_1] + 1e-15)
        intercept = mean[target] - beta_1 * mean[predictor_1]
        prediction = intercept + test[:, predictor_1] * beta_1
    else:
        a = correlation[target, predictor_1]
        b = correlation[target, predictor_2]
        c = correlation[predictor_1, predictor_2]
        denominator = np.maximum(1.0 - c * c, 1e-15)
        beta_1_std = (a - b * c) / denominator
        beta_2_std = (b - a * c) / denominator
        beta_1 = beta_1_std * std[target] / (std[predictor_1] + 1e-15)
        beta_2 = beta_2_std * std[target] / (std[predictor_2] + 1e-15)
        intercept = (
            mean[target]
            - beta_1 * mean[predictor_1]
            - beta_2 * mean[predictor_2]
        )
        prediction = (
            intercept
            + test[:, predictor_1] * beta_1
            + test[:, predictor_2] * beta_2
        )

    observed = test[:, target]
    residual = observed - prediction
    ss_residual = np.sum(residual * residual, axis=0, dtype=np.float64)
    centered_observed = observed - observed.mean(axis=0)
    ss_total = np.sum(centered_observed * centered_observed, axis=0, dtype=np.float64)
    return 1.0 - ss_residual / (ss_total + 1e-15)


def count_ill_conditioned_combinations(
    library: pd.DataFrame,
    combination_size: int,
    threshold: float,
    test_size: float,
    random_state: int,
    chunk_size: int = 50_000,
) -> int:
    """Count combinations using the notebook's split/test-R2 definition."""
    values = library.to_numpy(dtype=float)
    # Match the existing analyzer: library columns are z-scored before the
    # fixed train/test split.  This makes the R2 count invariant to either
    # nonzero external column scaling, which is an expected result.
    column_std = values.std(axis=0, ddof=1)
    if np.any(column_std <= 0.0) or np.any(~np.isfinite(column_std)):
        raise ValueError("R2 analysis requires nonconstant finite library columns.")
    processed = (values - values.mean(axis=0)) / column_std

    indices = np.arange(len(processed))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train = processed[train_index]
    test = processed[test_index]
    correlation, mean, std = _correlation_parameters(train)

    combos = combination_index(processed.shape[1], combination_size)
    count = 0
    for start in range(0, len(combos), chunk_size):
        block = combos[start : start + chunk_size]
        best_r2 = np.full(len(block), -np.inf, dtype=float)
        if combination_size == 2:
            i, j = block[:, 0], block[:, 1]
            best_r2 = np.maximum(
                best_r2, _test_r2_direction(correlation, mean, std, test, i, j)
            )
            best_r2 = np.maximum(
                best_r2, _test_r2_direction(correlation, mean, std, test, j, i)
            )
        elif combination_size == 3:
            i, j, k = block[:, 0], block[:, 1], block[:, 2]
            best_r2 = np.maximum(
                best_r2,
                _test_r2_direction(correlation, mean, std, test, i, j, k),
            )
            best_r2 = np.maximum(
                best_r2,
                _test_r2_direction(correlation, mean, std, test, j, i, k),
            )
            best_r2 = np.maximum(
                best_r2,
                _test_r2_direction(correlation, mean, std, test, k, i, j),
            )
        else:
            raise ValueError("Only two-term and three-term combinations are supported.")
        count += int(np.count_nonzero(best_r2 >= threshold))
    return count


def analyze(
    systems: Sequence[SystemData],
    r2_threshold: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    all_degrees = sorted(set(CONDITION_DEGREES) | set(COMBINATION_DEGREES))

    for system in systems:
        print(
            f"Analyzing {system.label}: {len(system.states)} observations, "
            f"{system.states.shape[1]} states",
            flush=True,
        )
        for basis in BASIS_ORDER:
            for degree in all_degrees:
                print(f"  {basis:10s} degree {degree}", flush=True)
                existing = intended_existing_library(system.states, basis, degree)
                scaled_libraries = {
                    mode: scale_library(existing, mode) for mode in SCALING_ORDER
                }

                if degree in CONDITION_DEGREES:
                    for mode, library in scaled_libraries.items():
                        matrix = library.to_numpy(dtype=float)
                        condition_number = float(np.linalg.cond(matrix, p=2))
                        condition_rows.append(
                            {
                                "system": system.key,
                                "system_label": system.label,
                                "basis": basis,
                                "degree": degree,
                                "scaling": mode,
                                "scaling_label": SCALING_LABELS[mode],
                                "n_observations": matrix.shape[0],
                                "n_library_terms": matrix.shape[1],
                                "matrix_rank": int(np.linalg.matrix_rank(matrix)),
                                "condition_number": condition_number,
                                "log10_condition_number": (
                                    math.log10(condition_number)
                                    if condition_number > 0.0
                                    else -np.inf
                                ),
                            }
                        )

                if degree in COMBINATION_DEGREES:
                    counts_by_mode: dict[str, dict[int, int]] = {}
                    for mode, library in scaled_libraries.items():
                        counts_by_mode[mode] = {}
                        for size in COMBINATION_SIZES:
                            count = count_ill_conditioned_combinations(
                                library,
                                combination_size=size,
                                threshold=r2_threshold,
                                test_size=test_size,
                                random_state=random_state,
                            )
                            counts_by_mode[mode][size] = count
                            combination_rows.append(
                                {
                                    "system": system.key,
                                    "system_label": system.label,
                                    "basis": basis,
                                    "degree": degree,
                                    "scaling": mode,
                                    "scaling_label": SCALING_LABELS[mode],
                                    "combination_size": size,
                                    "r2_threshold": r2_threshold,
                                    "test_size": test_size,
                                    "random_state": random_state,
                                    "n_observations": len(library),
                                    "n_library_terms": library.shape[1],
                                    "n_total_combinations": math.comb(library.shape[1], size),
                                    "n_ill_conditioned": count,
                                    "log10_n_ill_conditioned": (
                                        math.log10(count) if count > 0 else np.nan
                                    ),
                                }
                            )

                    baseline_counts = counts_by_mode["existing"]
                    for mode in ("unit_l2",):
                        if counts_by_mode[mode] != baseline_counts:
                            warnings.warn(
                                f"R2 count changed after {mode} scaling for "
                                f"{system.key}/{basis}/degree {degree}; inspect values "
                                "near the threshold.",
                                RuntimeWarning,
                            )

    return pd.DataFrame(condition_rows), pd.DataFrame(combination_rows)


def configure_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "svg.fonttype": "none",
        }
    )


def plot_condition_numbers(condition: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8.0),
        sharex=True,
        # Use one absolute y-axis for both systems. Matplotlib therefore
        # autoscales from the union of the LV and CRN values: the former LV
        # range is widened enough to retain every CRN curve, while all six
        # panels keep identical limits and tick locations.
        sharey=True,
        constrained_layout=False,
        dpi = 96,
    )
    for row_index, system in enumerate(("lv", "crn")):
        for column_index, mode in enumerate(SCALING_ORDER):
            axis = axes[row_index, column_index]
            panel = condition[
                (condition["system"] == system) & (condition["scaling"] == mode)
            ]
            for basis in BASIS_ORDER:
                curve = panel[panel["basis"] == basis].sort_values("degree")
                axis.plot(
                    curve["degree"],
                    curve["log10_condition_number"],
                    color=BASIS_COLORS[basis],
                    marker=BASIS_MARKERS[basis],
                    linewidth=1.8,
                    markersize=6,
                    label=basis,
                )
            axis.set_title(f"{SYSTEM_LABELS[system]}\n{SCALING_LABELS[mode]}")
            axis.set_xticks(CONDITION_DEGREES)
            axis.set_xlabel("Library degree")
            if column_index == 0:
                axis.set_ylabel(CONDITION_YLABEL)
                #axis.set_ylabel(r"$\log_{10}\,\kappa_2(\Theta)$")
            axis.set_yticks([4, 8, 12, 16, 20])
            axis.grid(True, alpha=0.3)
            #style_black_axis_frame(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Full polynomial-library condition numbers", fontsize=15, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.83, wspace=0.08, hspace=0.40)
    return fig


def plot_combination_counts(combination_counts: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8.0),
        sharex=True,
        # Use one y-axis scale for both LV and CRN.  Matplotlib determines
        # the shared limits from the union of all four panels, so the model
        # with the wider value range sets the range without clipping the
        # other model.
        sharey=True,
        constrained_layout=False,
        dpi = 96,
    )
    line_styles = {2: "--", 3: "-"}
    for row_index, system in enumerate(("lv", "crn")):
        for column_index, mode in enumerate(SCALING_ORDER):
            axis = axes[row_index, column_index]
            panel = combination_counts[
                (combination_counts["system"] == system)
                & (combination_counts["scaling"] == mode)
            ]
            for basis in BASIS_ORDER:
                for combination_size in COMBINATION_SIZES:
                    curve = panel[
                        (panel["basis"] == basis)
                        & (panel["combination_size"] == combination_size)
                    ].sort_values("degree")
                    raw_counts = curve["n_ill_conditioned"].to_numpy(float)
                    # Keep the old plot's visible zero-count convention while
                    # retaining the mathematically correct NaN log in the CSV.
                    plotted_log_count = np.where(
                        raw_counts > 0.0, np.log10(np.maximum(raw_counts, 1.0)), 0.0
                    )
                    axis.plot(
                        curve["degree"],
                        plotted_log_count,
                        color=BASIS_COLORS[basis],
                        marker=BASIS_MARKERS[basis],
                        linestyle=line_styles[combination_size],
                        linewidth=1.7,
                        markersize=5.5,
                    )
            axis.set_title(f"{SYSTEM_LABELS[system]}\n{SCALING_LABELS[mode]}")
            axis.set_xticks(COMBINATION_DEGREES)
            axis.set_xlabel("Library degree")
            if column_index == 0:
                axis.set_ylabel(COMBINATION_YLABEL)
                #axis.set_ylabel(r"$\log_{10}$(number of ill-conditioned combinations)")
            axis.grid(True, alpha=0.3)
            #style_black_axis_frame(axis)

    basis_handles = [
        Line2D(
            [0],
            [0],
            color=BASIS_COLORS[basis],
            marker=BASIS_MARKERS[basis],
            linewidth=1.8,
            label=basis,
        )
        for basis in BASIS_ORDER
    ]
    size_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="2-term"),
        Line2D([0], [0], color="black", linestyle="-", label="3-term"),
    ]
    # Keep basis identity (color/marker) separate from combination size
    # (line style), matching the manuscript-style layout.
    fig.legend(
        basis_handles,
        [handle.get_label() for handle in basis_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=False,
    )
    for axis in axes.flat:
        axis.legend(
            size_handles,
            [handle.get_label() for handle in size_handles],
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
        )
    threshold = combination_counts["r2_threshold"].iloc[0]
    fig.suptitle(
        f"Prevalence of ill-conditioning (maximum test $R^2 \\geq {threshold:g}$)",
        fontsize=15,
        y=0.975,
    )
    fig.text(
        0.5,
        0.012,
        "Display convention: zero combinations are plotted at 0; the CSV retains the raw zero counts.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.88, wspace=0.08, hspace=0.40)
    return fig


def _source_axis(figure, system: str, scaling: str):
    """Return the corresponding axis from a 2-by-2 comparison figure."""
    row = ("lv", "crn").index(system)
    column = SCALING_ORDER.index(scaling)
    axes = np.asarray(figure.axes, dtype=object)
    if axes.size != 4:
        raise AssertionError("Expected exactly four axes in the comparison figure.")
    return axes.reshape(2, 2)[row, column]


def _match_source_axis_limits(target_axis, source_axis) -> None:
    """Preserve the limits and ticks used in the corresponding full figure."""
    target_axis.set_xlim(source_axis.get_xlim())
    target_axis.set_ylim(source_axis.get_ylim())
    target_axis.set_xticks(source_axis.get_xticks())
    target_axis.set_yticks(source_axis.get_yticks())


def _save_condition_panel(
    condition: pd.DataFrame,
    source_figure,
    system: str,
    scaling: str,
    output_path: Path,
) -> None:
    """Redraw one Figure-1 panel as a self-contained figure."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    panel = condition[
        (condition["system"] == system) & (condition["scaling"] == scaling)
    ]
    for basis in BASIS_ORDER:
        curve = panel[panel["basis"] == basis].sort_values("degree")
        axis.plot(
            curve["degree"],
            curve["log10_condition_number"],
            color=BASIS_COLORS[basis],
            marker=BASIS_MARKERS[basis],
            linewidth=1.8,
            markersize=6,
            label=basis,
        )
    axis.set(
        xlabel="Library degree",
        ylabel=CONDITION_YLABEL,
        #ylabel=r"$\log_{10}\,\kappa_2(\Theta)$",
        title=f"{SYSTEM_LABELS[system]}\n{SCALING_LABELS[scaling]}",
    )
    axis.grid(True, alpha=0.3)
    _match_source_axis_limits(
        axis,
        _source_axis(source_figure, system, scaling),
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.25, top=0.84)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _save_combination_panel(
    combination_counts: pd.DataFrame,
    source_figure,
    system: str,
    scaling: str,
    output_path: Path,
) -> None:
    """Redraw one Figure-2 panel as a self-contained figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    panel = combination_counts[
        (combination_counts["system"] == system)
        & (combination_counts["scaling"] == scaling)
    ]
    line_styles = {2: "--", 3: "-"}
    for basis in BASIS_ORDER:
        for combination_size in COMBINATION_SIZES:
            curve = panel[
                (panel["basis"] == basis)
                & (panel["combination_size"] == combination_size)
            ].sort_values("degree")
            raw_counts = curve["n_ill_conditioned"].to_numpy(float)
            plotted_log_count = np.where(
                raw_counts > 0.0,
                np.log10(np.maximum(raw_counts, 1.0)),
                0.0,
            )
            axis.plot(
                curve["degree"],
                plotted_log_count,
                color=BASIS_COLORS[basis],
                marker=BASIS_MARKERS[basis],
                linestyle=line_styles[combination_size],
                linewidth=1.7,
                markersize=5.5,
            )
    threshold = float(combination_counts["r2_threshold"].iloc[0])
    axis.set(
        xlabel="Library degree",
        ylabel=COMBINATION_YLABEL,
        title=(
            f"{SYSTEM_LABELS[system]}\n{SCALING_LABELS[scaling]} "
            rf"($R^2 \geq {threshold:g}$)"
        ),
    )
    axis.grid(True, alpha=0.3)
    _match_source_axis_limits(
        axis,
        _source_axis(source_figure, system, scaling),
    )
    basis_handles = [
        Line2D(
            [0],
            [0],
            color=BASIS_COLORS[basis],
            marker=BASIS_MARKERS[basis],
            linewidth=1.8,
            label=basis,
        )
        for basis in BASIS_ORDER
    ]
    size_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="2-term"),
        Line2D([0], [0], color="black", linestyle="-", label="3-term"),
    ]
    size_legend = axis.legend(
        size_handles,
        [handle.get_label() for handle in size_handles],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )
    axis.add_artist(size_legend)
    axis.legend(
        basis_handles,
        [handle.get_label() for handle in basis_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.30, top=0.82)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_selected_panels(
    selections: Sequence[tuple[str, str, str]],
    output_dir: Path,
    file_format: str,
    condition: pd.DataFrame,
    combination_counts: pd.DataFrame,
    condition_figure,
    combination_figure,
) -> list[Path]:
    """Save requested panels and return their concrete output paths."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    # Preserve command-line order while avoiding accidental duplicate files.
    unique_selections = list(dict.fromkeys(selections))
    for figure_kind, system, scaling in unique_selections:
        output_path = output_dir / f"{figure_kind}_{system}_{scaling}.{file_format}"
        if figure_kind == "condition":
            _save_condition_panel(
                condition,
                condition_figure,
                system,
                scaling,
                output_path,
            )
        elif figure_kind == "combinations":
            _save_combination_panel(
                combination_counts,
                combination_figure,
                system,
                scaling,
                output_path,
            )
        else:  # Defensive guard for direct API calls bypassing argparse.
            raise ValueError(f"Unknown figure kind: {figure_kind}")
        saved_paths.append(output_path)
    return saved_paths


def save_outputs(
    output_dir: Path,
    condition: pd.DataFrame,
    combination_counts: pd.DataFrame,
    condition_figure,
    combination_figure,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition.to_csv(output_dir / "full_library_condition_numbers.csv", index=False)
    combination_counts.to_csv(
        output_dir / "ill_conditioned_combination_counts.csv", index=False
    )
    condition_figure.savefig(
        output_dir / "full_library_condition_scaling_comparison.png",
        bbox_inches="tight",
    )
    condition_figure.savefig(
        output_dir / "full_library_condition_scaling_comparison.pdf",
        bbox_inches="tight",
    )
    condition_figure.savefig(
        output_dir / "full_library_condition_scaling_comparison.svg",
        bbox_inches="tight",
    )
    combination_figure.savefig(
        output_dir / "ill_conditioned_combinations_scaling_comparison.png",
        bbox_inches="tight",
    )
    combination_figure.savefig(
        output_dir / "ill_conditioned_combinations_scaling_comparison.pdf",
        bbox_inches="tight",
    )
    combination_figure.savefig(
        output_dir / "ill_conditioned_combinations_scaling_comparison.svg",
        bbox_inches="tight",
    )
    print(f"Saved results to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    start = time.perf_counter()
    configure_plot_style()
    systems = simulate_systems()
    condition, combination_counts = analyze(
        systems,
        r2_threshold=args.r2_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Regression checks against values saved in Plots-1_modify.ipynb apply
    # only to the notebook's original threshold and split configuration.
    if (
        math.isclose(args.r2_threshold, 0.95)
        and math.isclose(args.test_size, 0.33)
        and args.random_state == 27
    ):
        known_crn_monomial_two = [1, 11, 38, 101, 235]
        known_crn_monomial_three = [4, 147, 1469, 8823, 39154]
        for size, expected in (
            (2, known_crn_monomial_two),
            (3, known_crn_monomial_three),
        ):
            observed = (
                combination_counts[
                    (combination_counts["system"] == "crn")
                    & (combination_counts["basis"] == "Monomial")
                    & (combination_counts["scaling"] == "existing")
                    & (combination_counts["combination_size"] == size)
                ]
                .sort_values("degree")["n_ill_conditioned"]
                .tolist()
            )
            if observed != expected:
                raise AssertionError(
                    f"CRN Monomial {size}-term counts no longer reproduce the notebook: "
                    f"observed {observed}, expected {expected}."
                )

    condition_figure = plot_condition_numbers(condition)
    combination_figure = plot_combination_counts(combination_counts)

    print("\nCondition-number summary:")
    print(
        condition.pivot_table(
            index=["system", "basis", "degree"],
            columns="scaling",
            values="log10_condition_number",
        ).round(4).to_string()
    )
    print("\nIll-conditioned-combination counts are scale invariant:")
    invariance = (
        combination_counts.pivot_table(
            index=["system", "basis", "degree", "combination_size"],
            columns="scaling",
            values="n_ill_conditioned",
        )
        .reset_index()
    )
    print(
        "  ",
        bool(
            (
                invariance[["existing", "unit_l2"]]
                .nunique(axis=1)
                .eq(1)
            ).all()
        ),
    )

    if args.save:
        save_outputs(
            args.output_dir,
            condition,
            combination_counts,
            condition_figure,
            combination_figure,
        )

    if args.save_panel:
        panel_output_dir = (
            args.panel_output_dir
            if args.panel_output_dir is not None
            else args.output_dir / "selected_panels"
        )
        saved_panel_paths = save_selected_panels(
            args.save_panel,
            panel_output_dir,
            args.panel_format,
            condition,
            combination_counts,
            condition_figure,
            combination_figure,
        )
        for path in saved_panel_paths:
            print(f"Saved standalone panel to {path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(condition_figure)
        plt.close(combination_figure)

    print(f"Completed in {time.perf_counter() - start:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
