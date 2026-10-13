"""Condition-number robustness for the sampled libraries in Plots_2.ipynb.

Each repeat changes only the initial-condition sets (Sobol scramble and shift
seed). The model, baseline bounds, Uniform/Arcsine distributions, number of
trajectories, five-point short time grids, endpoint removal, degree-five
libraries and no-noise setting follow the notebook. All three bases receive
the same normalized sampled states. Library columns are then scaled to unit
L2 norm, as requested for this robustness experiment; the notebook's
condition-number cells use unscaled library columns.

Example:
    python Exp_IC_condition_robustness.py --repeats 20 --seed 2026 \
        --output-dir results/ic_condition_robustness_sampled
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

CACHE = Path(tempfile.gettempdir()) / "sindy_ic_condition_cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))

from Exp_library_scaling_comparison import (
    BASIS_ORDER,
    CRN_RATES,
    DEGREE,
    DISTRIBUTIONS,
    DISTRIBUTION_COLORS,
    DISTRIBUTION_MARKERS,
    LV_PARAMS,
    SAMPLE_SPECS,
    STATE_COLUMNS,
    SYSTEM_LABELS,
    baseline_data,
    build_library,
    scale_library,
)
from Sample_Analysis import _widen_bounds, sampling, sobol_u01

SCRIPT_DIR = Path(__file__).resolve().parent
N_TIME_POINTS = 5
START_TIME = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20,
                        help="Independent IC sets for each model and distribution (default: 20).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base seed for IC Sobol sampling (default: 2026).")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR.parent / "results" / "ic_condition_robustness_sampled")
    args = parser.parse_args()
    if args.repeats < 2 or args.seed < 0:
        parser.error("--repeats must be at least 2 and --seed must be nonnegative.")
    return args


def set_seeded_initial_conditions(generator: sampling, seed: int) -> None:
    """Reproduce the notebook IC distribution with a repeat-specific seed.

    Sample_Analysis currently hardcodes seed=0 inside both sampling methods,
    so assigning sample_orig here is necessary for independent IC sets.
    """
    columns = list(generator.data_states.columns)
    u01 = sobol_u01(generator.num, len(columns), seed=seed)
    lower = generator.L_orig.to_numpy(dtype=float)
    upper = generator.U_orig.to_numpy(dtype=float)
    if generator.distribution.lower() == "uniform":
        physical = lower + u01 * (upper - lower)
        canonical = np.clip(2.0 * u01 - 1.0, -1 + 1e-7, 1 - 1e-7)
    elif generator.distribution.lower() == "arcsine":
        canonical = np.clip(np.cos(np.pi * u01), -1 + 1e-7, 1 - 1e-7)
        lower_wide, upper_wide = _widen_bounds(
            generator.L_orig, generator.U_orig, min_span=1e-6, frac=0.05
        )
        physical = (0.5 * (canonical + 1.0)
                    * (upper_wide.to_numpy(dtype=float) - lower_wide.to_numpy(dtype=float))
                    + lower_wide.to_numpy(dtype=float))
    else:
        raise ValueError(generator.distribution)
    generator.sample_orig = pd.DataFrame(physical, columns=columns)
    generator.sample_norm = pd.DataFrame(canonical, columns=columns)


def sample_one(baseline: pd.DataFrame, system: str, distribution: str,
               repeat: int, ic_seed: int, workdir: Path):
    spec = SAMPLE_SPECS[(system, distribution)]
    generator = sampling(
        baseline, spec["n_initial_conditions"],
        distribution=distribution.lower(), noise_level=None,
    )
    set_seeded_initial_conditions(generator, ic_seed)
    kwargs = {"params": LV_PARAMS} if system == "lv" else {"k_rates": CRN_RATES}
    paths = generator.fit_and_save_base(
        model_name="Lotka-Volterra" if system == "lv" else "CRN",
        out_path=workdir / system / distribution.lower() / f"repeat_{repeat:03d}",
        start=START_TIME, end=spec["end"], n_step=N_TIME_POINTS,
        method="Legendre" if distribution == "Uniform" else "Chebyshev",
        noise_level=None, **kwargs,
    )
    subsample = generator._subsample(paths, generator.L_orig, generator.U_orig)
    states = subsample.loc[:, STATE_COLUMNS[system]].reset_index(drop=True)
    expected = spec["n_initial_conditions"] * (N_TIME_POINTS - 2)
    if len(states) != expected:
        raise RuntimeError(
            f"{system}/{distribution}/repeat {repeat}: {len(states)} sampled rows; "
            f"expected {expected}. Check _subsample selection."
        )
    if not np.isfinite(states.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite sampled states for {system}/{distribution}/{repeat}.")
    ics = generator.sample_orig.copy()
    ics.insert(0, "trajectory", np.arange(1, len(ics) + 1))
    ics.insert(0, "ic_seed", ic_seed)
    ics.insert(0, "repeat", repeat)
    ics.insert(0, "distribution", distribution)
    ics.insert(0, "system", system)
    diagnostics = []
    for column in states.columns:
        values = states[column].to_numpy(dtype=float)
        diagnostics.append({
            "system": system, "distribution": distribution, "repeat": repeat,
            "state": column, "n_sampled_rows": len(states),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "n_outside_minus_one_one": int(np.count_nonzero((values < -1) | (values > 1))),
            "normalization_lower_original": float(generator.L_orig[column]),
            "normalization_upper_original": float(generator.U_orig[column]),
        })
    return states, ics, diagnostics


def analyze(states: pd.DataFrame, system: str, distribution: str,
            repeat: int, ic_seed: int) -> list[dict]:
    rows = []
    for basis in BASIS_ORDER:
        library = build_library(states, basis, DEGREE)
        matrix = scale_library(library, "unit_l2").to_numpy(dtype=float)
        condition = float(np.linalg.cond(matrix, p=2))
        rows.append({
            "system": system, "distribution": distribution, "repeat": repeat,
            "ic_seed": ic_seed, "basis": basis, "degree": DEGREE,
            "scaling": "unit_l2", "n_trajectories": SAMPLE_SPECS[(system, distribution)]["n_initial_conditions"],
            "n_time_points_per_trajectory": N_TIME_POINTS,
            "n_observations": matrix.shape[0], "n_library_terms": matrix.shape[1],
            "matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "condition_number": condition,
            "log10_condition_number": math.log10(condition),
        })
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, distribution, basis), group in results.groupby(
        ["system", "distribution", "basis"], sort=False
    ):
        values = group.log10_condition_number.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "system": system, "distribution": distribution, "basis": basis,
            "degree": DEGREE, "n_repeats": len(values), "n_finite": len(finite),
            "n_rank_deficient": int((group.matrix_rank < group.n_library_terms).sum()),
            "median_log10_condition": float(np.median(finite)) if len(finite) else math.nan,
            "q25_log10_condition": float(np.quantile(finite, 0.25)) if len(finite) else math.nan,
            "q75_log10_condition": float(np.quantile(finite, 0.75)) if len(finite) else math.nan,
            "minimum_log10_condition": float(finite.min()) if len(finite) else math.nan,
            "maximum_log10_condition": float(finite.max()) if len(finite) else math.nan,
        })
    return pd.DataFrame(rows)


def ordering_consistency(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, distribution), group in results.groupby(
        ["system", "distribution"], sort=False
    ):
        wide = group.pivot(index="repeat", columns="basis", values="log10_condition_number")
        wide = wide.loc[np.isfinite(wide.loc[:, list(BASIS_ORDER)]).all(axis=1)]
        n = len(wide)
        count = int(((wide.Monomial > wide.Legendre)
                     & (wide.Monomial > wide.Chebyshev)).sum())
        rows.append({
            "system": system, "distribution": distribution,
            "n_repeats_with_all_finite": n,
            "n_monomial_greater_than_both": count,
            "fraction_monomial_greater_than_both": count / n if n else math.nan,
        })
    return pd.DataFrame(rows)


def save_figure(results: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    x = np.arange(len(BASIS_ORDER), dtype=float)
    for axis, system in zip(axes, ("lv", "crn")):
        for distribution, offset in zip(DISTRIBUTIONS, (-0.045, 0.045)):
            part = summary[(summary.system == system)
                           & (summary.distribution == distribution)]
            part = part.set_index("basis").loc[list(BASIS_ORDER)]
            center = part.median_log10_condition.to_numpy(dtype=float)
            q25 = part.q25_log10_condition.to_numpy(dtype=float)
            q75 = part.q75_log10_condition.to_numpy(dtype=float)
            if not np.isfinite(np.r_[center, q25, q75]).all():
                raise ValueError(f"Cannot draw nonfinite error bars for {system}/{distribution}.")
            axis.errorbar(
                x + offset, center, yerr=np.vstack((center - q25, q75 - center)),
                color=DISTRIBUTION_COLORS[distribution],
                marker=DISTRIBUTION_MARKERS[distribution], linestyle="-",
                linewidth=1.8, markersize=6, capsize=4, elinewidth=1.3,
                label=distribution,
            )
        axis.set_title(SYSTEM_LABELS[system])
        axis.set_xticks(x, BASIS_ORDER)
        axis.set_xlim(-0.28, len(BASIS_ORDER) - 0.72)
        axis.set_xlabel("Polynomial basis")
        axis.set_ylabel(r"$\log_{10}\kappa_2(\Theta)$")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle(f"Degree-{DEGREE} sampled libraries across {results.repeat.nunique()} IC sets",
                 y=0.995)
    fig.text(0.5, 0.015, "Points: median; error bars: 25th–75th percentiles across IC sets",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.84))
    audit_dir = os.environ.get("FIGURE_AUDIT_SCRIPT_DIR")
    if audit_dir:
        sys.path.insert(0, audit_dir)
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig, json_out=out / "ic_condition_robustness.alignment.json",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    fig.savefig(out / "ic_condition_robustness.svg", bbox_inches="tight")
    fig.savefig(out / "ic_condition_robustness.pdf", bbox_inches="tight")
    fig.savefig(out / "ic_condition_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline = baseline_data()
    result_rows, ic_tables, diagnostics = [], [], []
    with tempfile.TemporaryDirectory(prefix="sindy_ic_robustness_") as temp:
        workdir = Path(temp)
        for repeat in range(1, args.repeats + 1):
            for system_index, system in enumerate(("lv", "crn")):
                for distribution_index, distribution in enumerate(DISTRIBUTIONS):
                    ic_seed = args.seed + 4 * (repeat - 1) + 2 * system_index + distribution_index
                    states, ics, diag = sample_one(
                        baseline[system], system, distribution, repeat, ic_seed, workdir
                    )
                    result_rows.extend(analyze(states, system, distribution, repeat, ic_seed))
                    ic_tables.append(ics)
                    diagnostics.extend(diag)
                    print(f"{system}/{distribution} repeat {repeat}/{args.repeats}: "
                          f"{len(states)} sampled rows", flush=True)
    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    ordering = ordering_consistency(results)
    pd.concat(ic_tables, ignore_index=True).to_csv(out / "initial_conditions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "sampled_state_diagnostics.csv", index=False)
    results.to_csv(out / "condition_numbers_by_repeat.csv", index=False)
    summary.to_csv(out / "condition_number_summary.csv", index=False)
    ordering.to_csv(out / "ordering_consistency.csv", index=False)
    config = {
        "notebook": "Plots_2.ipynb",
        "notebook_sampling_cells_zero_based": {"crn_uniform": 8, "crn_arcsine": 18,
                                               "lv_uniform": 24, "lv_arcsine": 33},
        "repeats": args.repeats, "seed": args.seed,
        "ic_sampling": "Sobol scramble plus random shift, repeat-specific seed",
        "sample_specs": {f"{s}_{d.lower()}": spec for (s, d), spec in SAMPLE_SPECS.items()},
        "start": START_TIME, "n_time_points": N_TIME_POINTS,
        "trajectory_time_selection": "notebook _subsample, endpoints removed",
        "degree": DEGREE, "noise_level": None, "scaling": "unit_l2",
        "basis_input": "same sampled states, affine map from original baseline min/max to [-1,1]; no clipping",
        "condition": "numpy.linalg.cond(matrix, p=2)",
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_figure(results, summary, out)
    print(f"Saved {args.repeats} repeats to {out}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
