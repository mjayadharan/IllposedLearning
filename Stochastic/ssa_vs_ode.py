"""Check that the Gillespie SSA reproduces the deterministic CRN once parameters are mapped.

Simulates the Figure 1/2 enzyme CRN (Base_test.CRN: k, kr, kcat, IC [S,E,ES,P] =
[2.5, 2, 1, 0], t in [0, 10]) and the SSA from stochastic_crn_recovery.ssa_conc with

    counts = omega * conc,  k1 = k / omega,  k2 = kr,  k3 = kcat,

for several system sizes omega. Reports, at the 16 time points the notebooks
sample, the error of a single SSA run and of the SSA ensemble mean relative to
the ODE (as a fraction of each species' range), and saves a figure.

    python Stochastic/ssa_vs_ode.py
    python Stochastic/ssa_vs_ode.py --omegas 10 100 1000 --runs 200
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stochastic_crn_recovery as scr  # noqa: E402  (also puts Dependence/ on sys.path)

LABELS = ["S", "E", "ES", "P"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omegas", type=float, nargs="+", default=[10, 100, 1000])
    p.add_argument("--runs", type=int, default=100, help="SSA runs per omega")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=HERE / "results")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_plot = np.linspace(0, 10, 201)
    ode_plot = scr.CRN(scr.K_RATES, scr.IC_BASE, t_plot).data_sim[LABELS].to_numpy()
    ode_samp = scr.CRN(scr.K_RATES, scr.IC_BASE, scr.T_BASE).data_sim[LABELS].to_numpy()
    span = ode_plot.max(0) - ode_plot.min(0)
    rng = random.Random(args.seed)

    rows = []
    fig, axes = plt.subplots(len(args.omegas), 4, figsize=(12, 2.6 * len(args.omegas)), sharex=True, squeeze=False)
    for i, omega in enumerate(args.omegas):
        runs = np.array([scr.ssa_conc(scr.IC_BASE, omega, t_plot, rng) for _ in range(args.runs)])
        samp = np.array([scr.ssa_conc(scr.IC_BASE, omega, scr.T_BASE, rng) for _ in range(args.runs)])
        single = (np.sqrt(((samp - ode_samp) ** 2).mean(1)) / span).mean(0)
        mean_err = np.abs(samp.mean(0) - ode_samp).max(0) / span
        for j, lab in enumerate(LABELS):
            rows.append({"omega": omega, "species": lab, "single_run_rel_rmse": single[j],
                         "ensemble_mean_rel_maxerr": mean_err[j]})
            ax = axes[i, j]
            m, s = runs[:, :, j].mean(0), runs[:, :, j].std(0)
            ax.fill_between(t_plot, m - s, m + s, color="#2a78d6", alpha=0.2, lw=0, label="SSA mean ± sd")
            ax.plot(t_plot, runs[0, :, j], color="#eb6834", lw=1, drawstyle="steps-post", label="one SSA run")
            ax.plot(t_plot, ode_plot[:, j], color="k", lw=1.5, label="ODE (Base_test.CRN)")
            ax.plot(t_plot, m, color="#2a78d6", lw=1.5, ls="--", label="SSA mean")
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(f"Ω = {omega:g}\nconcentration")
            if i == len(args.omegas) - 1:
                ax.set_xlabel("time")
            ax.set_xlim(0, 6)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Enzyme CRN: SSA (k1 = k/Ω) vs ODE, {args.runs} SSA runs per Ω", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out / "ssa_vs_ode.png", dpi=150, bbox_inches="tight")

    err = pd.DataFrame(rows)
    err.to_csv(args.out / "ssa_vs_ode_errors.csv", index=False)
    table = err.pivot(index="omega", columns="species", values="single_run_rel_rmse")[LABELS]
    print("Single-run RMS error vs ODE at the 16 notebook time points (fraction of species range):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nEnsemble-mean max error: {err['ensemble_mean_rel_maxerr'].max():.4f} (all omega, species)")
    print(f"Saved {args.out / 'ssa_vs_ode.png'} and {args.out / 'ssa_vs_ode_errors.csv'}")


if __name__ == "__main__":
    main()
