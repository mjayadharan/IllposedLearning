"""Compare L2 column scaling of libraries built from Plots_2 sampled states.

This reproduces the notebook's degree-5 Uniform and Arcsine sampling pipeline
for the Lotka–Volterra (LV) and chemical reaction network (CRN) models.  For
each model/distribution, the same sampled state rows are used to construct
Monomial, Legendre and Chebyshev libraries.  The two condition numbers for each
library differ only in whether its columns are divided by their L2 norms.

The sampling pipeline already applies an affine state map using the original
trajectory's per-state min/max bounds.  Short simulated trajectories can cross
these bounds, so some inputs can lie slightly outside [-1, 1]; those counts
are recorded, and no additional mapping or clipping is applied here.
Simulation trajectory spreadsheets are temporary; ``--save`` writes only the
numerical results, sample diagnostics and figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "sindy_sampled_scaling_cache"
(RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEGREE = 5
BASIS_ORDER = ("Monomial", "Legendre", "Chebyshev")
BASIS_COLORS = {"Monomial": "#9B85AD", "Legendre": "#1F78B4", "Chebyshev": "#FF7F00"}
SCALING_ORDER = ("existing", "unit_l2")
SCALING_LABELS = {"existing": "No column scaling", "unit_l2": "Library-column L2 = 1"}
SYSTEM_LABELS = {"lv": "Lotka–Volterra", "crn": "Chemical reaction network"}
DISTRIBUTIONS = ("Uniform", "Arcsine")
DISTRIBUTION_COLORS = {"Uniform": "#1F78B4", "Arcsine": "#FF7F00"}
DISTRIBUTION_MARKERS = {"Uniform": "o", "Arcsine": "^"}
LV_PARAMS = {
    "r": 0.2, "K": 1.7, "a1": 0.1, "a2": 0.2,
    "m1": 0.1, "m2": 0.1, "epsilon1": 1.0, "epsilon2": 2.0,
}
CRN_RATES = {"k": 2.12456, "kr": 1.10124, "kcat": 1.5093}
LV_TIMES = np.linspace(0.0, 20.0, 21)
CRN_TIMES = (10.0 / 51.0) * np.array(
    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 47, 51], dtype=float
)
SAMPLE_SPECS = {
    ("lv", "Uniform"): {"n_initial_conditions": 100, "end": 0.05},
    ("lv", "Arcsine"): {"n_initial_conditions": 100, "end": 0.05},
    ("crn", "Uniform"): {"n_initial_conditions": 42, "end": 0.01},
    ("crn", "Arcsine"): {"n_initial_conditions": 50, "end": 0.01},
}
STATE_COLUMNS = {"lv": ("x1", "x2", "x3"), "crn": ("x1", "x2", "x3", "x4")}


def parse_panel_spec(value: str) -> tuple[str, str]:
    """Accept the old condition:SYSTEM:SCALING standalone-panel syntax."""
    parts = value.lower().split(":")
    if len(parts) != 3 or parts[0] not in ("condition", "figure1", "1"):
        raise argparse.ArgumentTypeError("Use condition:lv:l2, for example.")
    system = parts[1]
    scaling = {"existing": "existing", "l2": "unit_l2", "unit_l2": "unit_l2"}.get(parts[2])
    if system not in SYSTEM_LABELS or scaling is None:
        raise argparse.ArgumentTypeError("SYSTEM must be lv/crn; SCALING must be existing/l2.")
    return system, scaling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save the figure and CSV/JSON data.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a figure window.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "sampled_library_scaling_comparison")
    parser.add_argument("--save-panel", action="append", type=parse_panel_spec, default=[],
                        metavar="condition:SYSTEM:SCALING")
    parser.add_argument("--panel-output-dir", type=Path, default=None)
    parser.add_argument("--panel-format", choices=("svg", "pdf", "png"), default="svg")
    args = parser.parse_args()
    if args.no_show and not (args.save or args.save_panel):
        parser.error("--no-show requires --save or --save-panel.")
    if args.panel_output_dir is not None and not args.save_panel:
        parser.error("--panel-output-dir requires --save-panel.")
    return args


def baseline_data() -> dict[str, pd.DataFrame]:
    """Noiseless representative trajectories and time grids from Plots_2."""
    from Base_test import CRN, Lotka_Volterra

    lv = Lotka_Volterra(LV_PARAMS, (0.0, 20.0), LV_TIMES,
                        [1.7, 1.0, 1.7], noise_level=None).data_sim
    lv = lv.rename(columns={"x": "x1", "y1": "x2", "y2": "x3"})
    crn = CRN(CRN_RATES, [2.5, 2.0, 1.0, 0.0], CRN_TIMES,
              noise_level=None).data_sim
    crn = crn.rename(columns={"S": "x1", "E": "x2", "ES": "x3", "P": "x4"})
    return {
        "lv": lv.loc[:, ["time", *STATE_COLUMNS["lv"]]].copy(),
        "crn": crn.loc[:, ["time", *STATE_COLUMNS["crn"]]].copy(),
    }


def build_library(states: pd.DataFrame, basis: str, degree: int) -> pd.DataFrame:
    """Build the notebook's total-degree, no-bias library for one basis."""
    from Basis import OrthogonalLibrary
    from dae_finder import PolyFeatureMatrix

    if basis == "Monomial":
        library = PolyFeatureMatrix(degree).fit_transform(states).drop(columns=["1"])
    elif basis in ("Legendre", "Chebyshev"):
        transformer = OrthogonalLibrary(
            degree=degree, method=basis, include_bias=False,
            include_interaction=(degree > 1),
        )
        transformer.fit(states)
        library = transformer.transform_to_dataframe(states, list(states.columns))
    else:
        raise ValueError(f"Unknown basis: {basis!r}")
    n_terms = math.comb(states.shape[1] + degree, degree) - 1
    if library.shape != (len(states), n_terms):
        raise AssertionError(f"Unexpected {basis} library shape: {library.shape}.")
    if not np.isfinite(library.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite values in {basis} library.")
    return library


def scale_library(library: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Keep the original matrix or give every column unit Euclidean norm."""
    values = library.to_numpy(dtype=float)
    if mode == "existing":
        scaled = values.copy()
    elif mode == "unit_l2":
        norms = np.linalg.norm(values, ord=2, axis=0)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("Cannot L2-scale zero or nonfinite library columns.")
        scaled = values / norms
        if not np.allclose(np.linalg.norm(scaled, axis=0), 1.0, rtol=1e-10, atol=1e-12):
            raise AssertionError("Unit L2 column scaling failed.")
    else:
        raise ValueError(f"Unknown scaling mode: {mode!r}")
    return pd.DataFrame(scaled, index=library.index, columns=library.columns)


def generate_sampled_states() -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Run the notebook's IC generation, short simulations and subsampling."""
    from Sample_Analysis import sampling

    baseline = baseline_data()
    sampled: dict[tuple[str, str], pd.DataFrame] = {}
    diagnostics = []
    with tempfile.TemporaryDirectory(prefix="sindy_sampled_scaling_") as workdir:
        for system in ("lv", "crn"):
            for distribution in DISTRIBUTIONS:
                spec = SAMPLE_SPECS[(system, distribution)]
                generator = sampling(
                    baseline[system], spec["n_initial_conditions"],
                    distribution=distribution.lower(), noise_level=None,
                )
                model_kwargs = {"params": LV_PARAMS} if system == "lv" else {"k_rates": CRN_RATES}
                paths = generator.fit_and_save_base(
                    model_name="Lotka-Volterra" if system == "lv" else "CRN",
                    out_path=Path(workdir) / system / distribution.lower(),
                    start=0, end=spec["end"], n_step=5,
                    method="Legendre" if distribution == "Uniform" else "Chebyshev",
                    noise_level=None, **model_kwargs,
                )
                subsample = generator._subsample(paths, generator.L_orig, generator.U_orig)
                states = subsample.loc[:, STATE_COLUMNS[system]].reset_index(drop=True)
                if not np.isfinite(states.to_numpy(dtype=float)).all():
                    raise ValueError(f"Nonfinite sampled states for {system}/{distribution}.")
                sampled[(system, distribution)] = states
                for column in states.columns:
                    values = states[column].to_numpy(dtype=float)
                    diagnostics.append({
                        "system": system, "distribution": distribution, "state": column,
                        "n_sampled_rows": len(states), "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "n_outside_minus_one_one": int(np.count_nonzero((values < -1) | (values > 1))),
                        "normalization_lower_original": float(generator.L_orig[column]),
                        "normalization_upper_original": float(generator.U_orig[column]),
                    })
                print(f"{system}/{distribution}: {len(states)} sampled rows", flush=True)
    return sampled, pd.DataFrame(diagnostics)


def analyze(sampled: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for system in ("lv", "crn"):
        for distribution in DISTRIBUTIONS:
            states = sampled[(system, distribution)]
            for basis in BASIS_ORDER:
                library = build_library(states, basis, DEGREE)
                for scaling in SCALING_ORDER:
                    matrix = scale_library(library, scaling).to_numpy(dtype=float)
                    condition = float(np.linalg.cond(matrix, p=2))
                    rows.append({
                        "system": system, "distribution": distribution,
                        "basis": basis, "degree": DEGREE, "scaling": scaling,
                        "n_observations": matrix.shape[0],
                        "n_library_terms": matrix.shape[1],
                        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
                        "condition_number": condition,
                        "log10_condition_number": math.log10(condition),
                    })
    return pd.DataFrame(rows)


def draw_panel(ax, results: pd.DataFrame, system: str, scaling: str) -> None:
    x = np.arange(len(BASIS_ORDER))
    for distribution in DISTRIBUTIONS:
        rows = results[(results.system == system) & (results.scaling == scaling)
                       & (results.distribution == distribution)].set_index("basis")
        y = rows.loc[list(BASIS_ORDER), "log10_condition_number"].to_numpy(dtype=float)
        ax.plot(x, y, color=DISTRIBUTION_COLORS[distribution],
                marker=DISTRIBUTION_MARKERS[distribution], markersize=7,
                linewidth=1.8, label=distribution)
    ax.set_title(f"{SYSTEM_LABELS[system]}\n{SCALING_LABELS[scaling]}")
    ax.set_xticks(x, BASIS_ORDER)
    ax.grid(alpha=0.25)


def plot_condition_numbers(results: pd.DataFrame):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey="row")
    for row_index, system in enumerate(("lv", "crn")):
        for column_index, scaling in enumerate(SCALING_ORDER):
            ax = axes[row_index, column_index]
            draw_panel(ax, results, system, scaling)
            if column_index == 0:
                ax.set_ylabel(r"$\log_{10}\kappa_2(\Theta)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False)
    fig.suptitle("Condition numbers of sampled degree-5 libraries", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    return fig, axes


def save_outputs(results: pd.DataFrame, diagnostics: pd.DataFrame, fig,
                 output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "sampled_library_condition_numbers.csv", index=False)
    results.to_csv(output_dir / "full_library_condition_numbers.csv", index=False)
    diagnostics.to_csv(output_dir / "sampled_state_diagnostics.csv", index=False)
    config = {
        "degree": DEGREE, "distributions": list(DISTRIBUTIONS),
        "sample_specs": {f"{system}_{dist.lower()}": spec
                         for (system, dist), spec in SAMPLE_SPECS.items()},
        "trajectory_points": 5, "noise_level": None,
        "sampling": "Plots_2 sampling.fit_and_save_base followed by _subsample",
        "sampled_state_scale": "original baseline state min/max affine map during trajectory saving; values may exceed [-1,1]",
        "column_scalings": list(SCALING_ORDER),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    fig.savefig(output_dir / "sampled_library_scaling_comparison.svg", bbox_inches="tight")
    fig.savefig(output_dir / "sampled_library_scaling_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sampled_library_scaling_comparison.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(output_dir / "full_library_condition_scaling_comparison.svg", bbox_inches="tight")
    fig.savefig(output_dir / "full_library_condition_scaling_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "full_library_condition_scaling_comparison.png", dpi=300,
                bbox_inches="tight")


def save_selected_panels(results: pd.DataFrame, axes, selections,
                         output_dir: Path, extension: str) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for system, scaling in dict.fromkeys(selections):
        row = ("lv", "crn").index(system)
        column = SCALING_ORDER.index(scaling)
        source_axis = axes[row, column]
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        draw_panel(ax, results, system, scaling)
        ax.set_ylabel(r"$\log_{10}\kappa_2(\Theta)$")
        ax.set_ylim(source_axis.get_ylim())
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(output_dir / f"condition_{system}_{scaling}.{extension}",
                    bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    sampled, diagnostics = generate_sampled_states()
    results = analyze(sampled)
    fig, axes = plot_condition_numbers(results)
    print(results.pivot(index=["system", "distribution"], columns=["scaling", "basis"],
                        values="log10_condition_number").round(4).to_string())
    if args.save:
        out = args.output_dir.expanduser().resolve()
        save_outputs(results, diagnostics, fig, out)
        print(f"Saved sampled-library results to {out}")
    if args.save_panel:
        out = (args.panel_output_dir or args.output_dir / "selected_panels").expanduser().resolve()
        save_selected_panels(results, axes, args.save_panel, out, args.panel_format)
        print(f"Saved selected panels to {out}")
    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
