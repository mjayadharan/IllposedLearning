"""LV IC-count/trajectory-length recovery and distribution-mismatch experiment.

For each initial-condition count, this script generates one equal-length
candidate.  Counts below the reference produce fewer/longer trajectories;
counts above it produce more/shorter trajectories.  The script reports:

1. the number of equations whose recovered term set is exactly correct; and
2. the joint-histogram KL from its empirical reference to that candidate; and
3. the joint-histogram KL from the theoretical product target distribution
   (uniform or arcsine) to that candidate; and
4. the condition number of the run's full function library after each library
   column is scaled to unit L2 norm.

Uniform/Legendre and arcsine/Chebyshev are run independently
with the same Sobol seed. 
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Configure writable caches before the experiment module imports Matplotlib.
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "sindy_remark3_cache"
(RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning
)

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dependence.remark3_experiment import (  # noqa: E402
    DistributionComparison,
    EqualLengthExperiment,
    ModelRecovery,
    ReferenceSampling,
    StudySetup,
    joint_kl_theoretical_target_to_sample,
    l2_scaled_library_diagnostics,
)


REFERENCE_END = 0.05
DEGREE = 5
THRESHOLD = 1e-2
AFTER_THRESHOLD = 0.08
NOISE_LEVEL = 0.0
JOINT_BINS = 3
KL_PSEUDOCOUNT = 0.5

@dataclass(frozen=True)
class DistributionDesign:
    distribution: str
    basis: str
    edge: tuple[int, int]
    label: str
    color: str
    marker: str
    ic_counts: tuple[int, ...]


# These are explicit, distribution-specific empirical references.  The
# arcsine edge (24, 5) is the LV/Chebyshev edge recorded for the larger degree-5
# library in Plots_2.ipynb; with the common degree 2 used here it is a
# conservative reference, not a newly estimated minimal edge.
DESIGNS = (
    DistributionDesign(
        distribution="uniform",
        basis="Legendre",
        edge=(43, 5),
        label="Uniform / Legendre",
        color="#0072B2",
        marker="o",
        #ic_counts=(10, 8, 7, 6, 4, 3, 2, 1),
        ic_counts=(28, 21, 16, 14, 12, 10, 8, 7, 6, 4, 3, 2, 1),
        #ic_counts=(41, 28, 21, 16, 14, 12, 10, 8, 7, 6, 4, 3, 2, 1),
    ),
    DistributionDesign(
        distribution="arcsine",
        basis="Chebyshev",
        #edge=(5, 12),
        edge=(24, 5),
        label="Arcsine / Chebyshev",
        color="#D55E00",
        marker="^",
        #ic_counts=(10, 8, 7, 6, 4, 3, 2, 1),
        #ic_counts=(50, 31, 23, 21, 16, 14, 12, 10, 8, 7, 6, 4, 3, 2, 1),
        ic_counts=(16, 14, 12, 10, 8, 7, 6, 4, 3, 2, 1),
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LV uniform/arcsine IC-count/trajectory-length recovery, KL, "
            "and L2-scaled library conditioning."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sobol seed used by both distributions (default: 0).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save CSV, PNG, and PDF outputs (default: display only).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the figure window; normally used with --save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/remark3_lv_ic_trajectory_tradeoff).",
    )
    return parser.parse_args(argv)


def make_study(design: DistributionDesign) -> Any:
    # LV/arcsine has no module-level pair defaults, so every non-edge setting
    # is passed explicitly here.
    return StudySetup(
        model="lv",
        distribution=design.distribution,
        edge=design.edge,
        basis=design.basis,
        reference_end=REFERENCE_END,
        degree=DEGREE,
        threshold=THRESHOLD,
        after_threshold=AFTER_THRESHOLD,
        joint_bins=JOINT_BINS,
        kl_pseudocount=KL_PSEUDOCOUNT,
        # Only raw KL is requested; skip finite-sample null resampling.
        kl_null_repeats=0,
        noise_level=NOISE_LEVEL,
    ).prepare()


def evaluate_run(
    study: Any,
    design: DistributionDesign,
    run: Any,
    comparator: DistributionComparison,
    recoverer: ModelRecovery,
    experiment: str,
) -> dict[str, object]:
    """Compute raw KL and exact term-set recovery for one run."""
    distribution_result = comparator.compare(run)
    allocation = np.asarray(run.allocation, dtype=int)
    if allocation.size == 0 or not np.all(allocation == allocation[0]):
        raise AssertionError(
            "Reference and equal-length candidate runs must each use one "
            "trajectory length."
        )

    retained = int(allocation[0])
    row: dict[str, object] = {
        "distribution": design.distribution,
        "basis": design.basis,
        "label": design.label,
        "experiment": experiment,
        "run_id": run.run_id,
        "seed": run.seed,
        "n_initial_conditions": int(run.n_initial_conditions),
        "reference_n_initial_conditions": study.config.reference_n_ic,
        "reference_n_observations": comparator.reference.n_observations,
        "target_n_observations": run.target_n_observations,
        "actual_n_observations": run.n_observations,
        "n_observations_delta": run.n_observations_delta,
        "retained_per_trajectory": retained,
        "trajectory_end": (retained + 1) * study.config.dt,
        "joint_kl_reference_to_candidate": distribution_result.metrics[
            "joint_kl_reference_to_case"
        ],
        "joint_kl_theoretical_target_to_candidate": joint_kl_theoretical_target_to_sample(
            run.batch.x_norm,
            design.distribution,
            JOINT_BINS,
            KL_PSEUDOCOUNT,
        ),
        "outside_fraction": distribution_result.metrics["outside_fraction"],
    }
    # This uses the same normalized states, basis, degree, interactions, and
    # bias column as the SINDy recovery.  L2 scaling is diagnostic only and is
    # not passed into the optimizer.
    row.update(l2_scaled_library_diagnostics(study, run))

    n_equations = len(study.model.true_support)
    try:
        recovery = recoverer.fit(run, suppress_stdout=True)
        matches = [
            set(predicted) == set(truth)
            for predicted, truth in zip(
                recovery.predicted_support, study.model.true_support
            )
        ]
        if len(matches) != n_equations:
            raise AssertionError("Recovered and true equation counts differ.")
        correct = int(sum(matches))
        if correct != n_equations - int(recovery.metrics["wrong_equations"]):
            raise AssertionError("Equation count disagrees with support metrics.")
        row["correct_equations"] = correct
        row.update(
            {f"equation_{i}_correct": int(value) for i, value in enumerate(matches, 1)}
        )
        row["fit_error"] = ""
    except Exception as error:
        row["correct_equations"] = 0
        row.update({f"equation_{i}_correct": 0 for i in range(1, n_equations + 1)})
        row["fit_error"] = f"{type(error).__name__}: {error}"
    return row


def run_experiment(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []

    for design in DESIGNS:
        print(f"Running {design.label}, seed={seed} ...", flush=True)
        study = make_study(design)
        reference = ReferenceSampling(study, seed=seed).sample()
        comparator = DistributionComparison(study, reference)
        recoverer = ModelRecovery(study)
        equal_length = EqualLengthExperiment(study, reference)

        rows.append(
            evaluate_run(
                study, design, reference, comparator, recoverer, "reference"
            )
        )
        invalid_ic_counts = [
            n_ic
            for n_ic in design.ic_counts
            if (
                n_ic <= 0
                or n_ic == reference.n_initial_conditions
                or n_ic > reference.n_observations
            )
        ]
        if invalid_ic_counts:
            raise ValueError(
                f"Invalid IC grid for {design.label}: {invalid_ic_counts}. "
                "Counts must be positive, differ from the reference IC count, "
                "and not exceed the reference observation budget."
            )
        for n_ic in design.ic_counts:
            candidate = equal_length.simulate(
                n_ic,
                allow_more_initial_conditions=True,
            )
            rows.append(
                evaluate_run(
                    study,
                    design,
                    candidate,
                    comparator,
                    recoverer,
                    candidate.kind,
                )
            )
            retained = int(candidate.allocation[0])
            design_rows.append(
                {
                    "distribution": design.distribution,
                    "basis": design.basis,
                    "experiment": candidate.kind,
                    "n_initial_conditions": n_ic,
                    "retained_per_trajectory": retained,
                    "n_step": retained + 2,
                    "trajectory_end": (retained + 1) * study.config.dt,
                    "target_N": candidate.target_n_observations,
                    "actual_N": candidate.n_observations,
                    "delta_N": candidate.n_observations_delta,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["distribution", "experiment", "n_initial_conditions"]
    ).reset_index(drop=True)
    candidates = results[results.experiment != "reference"]
    if candidates.duplicated(["distribution", "n_initial_conditions"]).any():
        raise AssertionError("Each distribution/IC condition must have one result.")
    return results, pd.DataFrame(design_rows)


def make_figure(
    results: pd.DataFrame,
    output_dir: Path,
    save: bool,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    figure, (
        recovery_axis,
        reference_kl_axis,
        target_kl_axis,
        condition_axis,
    ) = plt.subplots(
        1, 4, figsize=(25.0, 5.3)
    )

    for design in DESIGNS:
        group = results[results.distribution == design.distribution]
        #candidate = group[group.experiment != "reference"].sort_values(
        #    "n_initial_conditions"
        #)
        curve = group.sort_values("n_initial_conditions")
        reference = group[group.experiment == "reference"].iloc[0]

        recovery_axis.plot(
            curve.n_initial_conditions,
            curve.correct_equations,
            #candidate.n_initial_conditions,
            #candidate.correct_equations,
            color=design.color,
            marker=design.marker,
            markersize=6,
            linewidth=2,
            label=design.label,
        )
        recovery_axis.scatter(
            [reference.n_initial_conditions],
            [reference.correct_equations],
            color=design.color,
            marker="*",
            s=130,
            zorder=5,
        )

        reference_kl_axis.plot(
            curve.n_initial_conditions,
            curve.joint_kl_reference_to_candidate,
            #candidate.n_initial_conditions,
            #candidate.joint_kl_reference_to_candidate,
            color=design.color,
            marker=design.marker,
            markersize=6,
            linewidth=2,
            label=design.label,
        )
        reference_kl_axis.scatter(
            [reference.n_initial_conditions],
            [reference.joint_kl_reference_to_candidate],
            color=design.color,
            marker="*",
            s=130,
            zorder=5,
        )

        target_kl_axis.plot(
            curve.n_initial_conditions,
            curve.joint_kl_theoretical_target_to_candidate,
            #candidate.joint_kl_theoretical_target_to_candidate,
            color=design.color,
            marker=design.marker,
            markersize=6,
            linewidth=2,
            label=design.label,
        )
        target_kl_axis.scatter(
            [reference.n_initial_conditions],
            [reference.joint_kl_theoretical_target_to_candidate],
            color=design.color,
            marker="*",
            s=130,
            zorder=5,
        )

        condition_axis.plot(
            curve.n_initial_conditions,
            curve.library_l2_log10_condition_number,
            color=design.color,
            marker=design.marker,
            markersize=6,
            linewidth=2,
            label=design.label,
        )
        condition_axis.scatter(
            [reference.n_initial_conditions],
            [reference.library_l2_log10_condition_number],
            color=design.color,
            marker="*",
            s=130,
            zorder=5,
        )

    recovery_axis.axhline(3, color="0.65", linestyle="--", linewidth=1)
    recovery_axis.set_yticks([0, 1, 2, 3])
    recovery_axis.set_ylim(-0.1, 3.15)
    recovery_axis.set(
        xlabel="Number of initial conditions",
        ylabel="Number of correctly recovered equations",
        title="A  Structural recovery",
    )

    reference_kl_axis.axhline(0, color="0.65", linestyle="--", linewidth=1)
    reference_kl_axis.set_ylim(bottom=-0.02)
    reference_kl_axis.set(
        xlabel="Number of initial conditions",
        ylabel=r"Joint $D_{KL}(P_{reference}\Vert P_{candidate})$",
        title="B  Mismatch to empirical reference",
    )

    target_kl_axis.axhline(0, color="0.65", linestyle="--", linewidth=1)
    target_kl_axis.set_ylim(bottom=-0.02)
    target_kl_axis.set(
        xlabel="Number of initial conditions",
        ylabel=r"Joint $D_{KL}(P_{target}\Vert P_{candidate})$",
        title="C  Mismatch to theoretical target",
    )

    condition_axis.set(
        xlabel="Number of initial conditions",
        ylabel=r"$\log_{10}\,\kappa_2(\Theta_{\mathrm{L2}})$",
        title="D  Scaled library conditioning",
    )

    max_plotted_ic = int(results["n_initial_conditions"].max())
    x_ticks = np.concatenate(([1], np.arange(5, max_plotted_ic + 1, 5)))
    for axis in (
        recovery_axis,
        reference_kl_axis,
        target_kl_axis,
        condition_axis,
    ):
        axis.set_xticks(x_ticks)
        axis.set_xlim(0, max_plotted_ic + 1.5)

    handles, labels = recovery_axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0], [0], marker="*", color="black", linestyle="none", markersize=10
        )
    )
    labels.append("Reference")
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0.12, 1, 1])

    if save:
        figure.savefig(
            output_dir / "lv_ic_trajectory_tradeoff_recovery_kl_and_condition.png",
            bbox_inches="tight",
        )
        figure.savefig(
            output_dir / "lv_ic_trajectory_tradeoff_recovery_kl_and_condition.pdf",
            bbox_inches="tight",
        )
    if show:
        plt.show()
    else:
        plt.close(figure)

#def


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative.")
    if args.no_show and not args.save:
        raise SystemExit("Nothing to output: remove --no-show or add --save.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "results" / "remark3_lv_ic_trajectory_tradeoff"
    )
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using: {SCRIPT_DIR / 'remark3_experiment.py'}")
    print(
        f"KL estimators: 3D target-CDF joint histograms, "
        f"{JOINT_BINS} bins/dimension, pseudocount={KL_PSEUDOCOUNT}. "
        "Panel B computes D_KL(reference || candidate); panel C computes "
        "D_KL(theoretical target || candidate). Panel D uses the full SINDy library with "
        "each column scaled to unit L2 norm."
    )
    results, design = run_experiment(args.seed)
    make_figure(
        results,
        output_dir,
        save=args.save,
        show=not args.no_show,
    )

    candidates = results[results.experiment != "reference"]
    print("\nCandidate results:")
    print(
        candidates[
            [
                "distribution",
                "basis",
                "n_initial_conditions",
                "actual_n_observations",
                "correct_equations",
                "joint_kl_reference_to_candidate",
                "joint_kl_theoretical_target_to_candidate",
                "library_l2_log10_condition_number",
                "library_n_rows",
                "library_n_columns",
                "library_numerical_rank",
                "library_full_column_rank",
            ]
        ].to_string(index=False)
    )
    failures = int(results.fit_error.ne("").sum())
    print(f"\nEvaluated {len(results)} runs; fit failures: {failures}")
    rank_deficient = results[~results.library_full_column_rank]
    if len(rank_deficient):
        print(
            "Condition diagnostic: "
            f"{len(rank_deficient)} run(s) are not full column rank; their "
            "finite plotted value describes the numerical singular spectrum, "
            "not unique coefficient identifiability."
        )

    if args.save:
        results.to_csv(output_dir / "lv_ic_trajectory_tradeoff_results.csv", index=False)
        design.to_csv(output_dir / "lv_ic_trajectory_tradeoff_design.csv", index=False)
        print(f"Saved results to: {output_dir}")
    else:
        print("Results were displayed only; no files were saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
