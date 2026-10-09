"""Summarize and plot stochastic_crn_recovery.py results.

Collects every recovery_results.csv under the input directory (default
Stochastic/output), writes one combined per-equation table, and plots equations
recovered and spurious terms versus system size omega, one series per number of
sampled initial conditions.

    python Stochastic/plot_recovery.py
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
KEYS = ["num", "end", "n_step", "degree", "omega", "replicates", "seed"]
STYLE = {35: dict(color="#eb6834", marker="^", ls="--"), 280: dict(color="#2a78d6", marker="o", ls="-")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", type=Path, default=HERE / "output")
    p.add_argument("--out", type=Path, default=HERE / "results")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    res = pd.concat([pd.read_csv(f) for f in sorted(args.inp.rglob("recovery_results.csv"))], ignore_index=True)
    res = res.drop_duplicates(KEYS + ["eq"]).sort_values(KEYS + ["eq"])
    res.to_csv(args.out / "recovery_summary.csv", index=False)

    runs = (res.groupby(KEYS).agg(eqs_correct=("correct_terms", "sum"), extra_terms=("n_extra", "sum"))
            .reset_index())
    omegas = sorted(o for o in runs["omega"].unique() if np.isfinite(o))
    xpos = {o: i for i, o in enumerate(omegas)}
    xpos[np.inf] = len(omegas)
    xlabels = [f"$10^{{{int(np.log10(o))}}}$" for o in omegas] + ["ODE"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (num, end, n_step), grp in runs.groupby(["num", "end", "n_step"]):
        st = STYLE.get(int(num), dict(color="gray", marker="s", ls=":"))
        off = 0.06 if num < 100 else -0.06
        for ax, col in zip(axes, ["eqs_correct", "extra_terms"]):
            x = grp["omega"].map(xpos) + off
            ax.scatter(x + np.random.default_rng(0).uniform(-0.04, 0.04, len(x)), grp[col],
                       color=st["color"], marker=st["marker"], alpha=0.35, s=18)
            mean = grp.groupby("omega")[col].mean()
            ax.plot(mean.index.map(xpos) + off, mean.values, color=st["color"], marker=st["marker"], ls=st["ls"],
                    label=f"{int(num)} ICs (window {end:g}, {int(n_step)} pts)")
    for ax, title, ylim in zip(axes, ["Equations recovered with exactly the true terms",
                                      "Spurious terms (sum over 4 equations)"], [(-0.2, 4.3), (0, None)]):
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_xlabel("system size Ω (molecules per unit concentration)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_yticks(range(5))
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Legendre recovery of the enzyme CRN from uniformly sampled ICs: SSA vs ODE "
                 "(dots = SSA seeds, line = mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "recovery_vs_omega.png", dpi=150, bbox_inches="tight")

    pd.set_option("display.width", 200)
    print(runs.to_string(index=False))
    print(f"\nSaved {args.out / 'recovery_vs_omega.png'} and {args.out / 'recovery_summary.csv'}")


if __name__ == "__main__":
    main()
