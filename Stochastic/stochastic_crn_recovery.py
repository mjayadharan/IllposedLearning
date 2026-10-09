"""Does Legendre recovery from uniformly sampled ICs survive stochastic CRN dynamics?

Repeats the Figure 2 test for the enzyme CRN (Figure2/Plots_2.ipynb, uniform
sampling + Legendre library, cells "IC_uniform" and "Legendre_sampled") twice
on the same initial conditions:

  deterministic  sampling.fit_and_save_base(model_name='CRN', ...) as in the
                 notebook: each IC is integrated with Base_test.CRN over a short
                 window and differentiated with ps.FiniteDifference(order=2).
  stochastic     identical, except each short trajectory comes from a Gillespie
                 SSA (EnzymeKinetics.py) with system size omega:
                     counts = omega * conc,  k1 = k / omega,  k2 = kr,  k3 = kcat,
                 optionally averaged over `replicates` independent SSA runs.

Everything downstream (normalization, _subsample, Recover_Model_sindy with the
notebook's degree/threshold/after_threshold, denormalization) is the repo code,
unchanged. Recovered equations are scored against the true mass-action model.

Run from anywhere with the `illposed` env:
    python Stochastic/stochastic_crn_recovery.py
    python Stochastic/stochastic_crn_recovery.py --omegas 1000 10000 --replicates 1 10 100
"""
import argparse
import contextlib
import io
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Dependence"))
warnings.filterwarnings("ignore")

import pysindy as ps  # noqa: E402
from Base_test import CRN  # noqa: E402
from Basis import normalization  # noqa: E402
from Recover_Model import Recover_Model_sindy  # noqa: E402
from Sample_Analysis import sampling  # noqa: E402

# Parameters from Figure2/Plots_2.ipynb
K_RATES = {"k": 2.12456, "kr": 1.10124, "kcat": 1.5093}
IC_BASE = [2.5, 2, 1, 0]
T_BASE = (10 / 51) * np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 47, 51])
STATES = ["x1", "x2", "x3", "x4"]  # S, E, ES, P (Base_test.CRN order)

k, kr, kcat = K_RATES["k"], K_RATES["kr"], K_RATES["kcat"]
TRUE_MODEL = {
    "x1": {"x1 x2": -k, "x3": kr},
    "x2": {"x1 x2": -k, "x3": kr + kcat},
    "x3": {"x1 x2": k, "x3": -(kr + kcat)},
    "x4": {"x3": kcat},
}


def base_data():
    """The single deterministic trajectory the notebook samples ICs and bounds from."""
    df = CRN(K_RATES, IC_BASE, T_BASE).data_sim
    df = df.rename(columns={"S": "x1", "E": "x2", "ES": "x3", "P": "x4"})
    return df[["time"] + STATES]


def ssa_conc(ic_conc, omega, t_eval, rng):
    """EnzymeKinetics.py Gillespie loop, state sampled at t_eval, returned as counts/omega.

    State order matches Base_test.CRN: [S, E, ES, P].
    """
    k1, k2, k3 = k / omega, kr, kcat
    S, E, C, P = (int(round(v * omega)) for v in ic_conc)
    out = np.empty((len(t_eval), 4))
    t, i = 0.0, 0
    while i < len(t_eval):
        a1, a2, a3 = k1 * E * S, k2 * C, k3 * C
        a0 = a1 + a2 + a3
        tau = rng.expovariate(a0) if a0 > 0 else np.inf
        while i < len(t_eval) and t_eval[i] < t + tau:  # state is constant until next event
            out[i] = (S, E, C, P)
            i += 1
        if i == len(t_eval):
            break
        r = rng.random() * a0
        if r < a1:
            S -= 1; E -= 1; C += 1
        elif r < a1 + a2:
            S += 1; E += 1; C -= 1
        else:
            E += 1; C -= 1; P += 1
        t += tau
    return out / omega


def save_ssa_trajectories(Sample, out_path, start, end, n_step, omega, replicates, seed):
    """Stochastic twin of sampling.fit_and_save_base(model_name='CRN', method='Legendre').

    Same output directory layout, file names, normalization (global L_orig/U_orig)
    and derivative estimate; only the simulator differs.
    """
    out_dir, tag = Sample._prepare_outdir(out_path, n_step, "IC")
    Sample.ID = list(Sample.sample_orig.columns)  # _subsample relies on this
    t_eval = np.linspace(float(start), float(end), int(n_step))
    cfd = ps.FiniteDifference(order=2)
    rng = random.Random(seed)
    filepaths = []
    for j, row in Sample.sample_orig.iterrows():
        ic = [float(row[s]) for s in STATES]
        X = np.mean([ssa_conc(ic, omega, t_eval, rng) for _ in range(replicates)], axis=0)
        df = pd.DataFrame(X, columns=STATES)
        df_norm = normalization(df, Sample.L_orig, Sample.U_orig)[0]
        Xdot = cfd(df_norm.to_numpy(), t_eval)
        df_norm.insert(0, "time", t_eval)
        for c, s in enumerate(STATES):
            df_norm[f"d{s}/dt"] = Xdot[:, c]
        fp = Path(out_dir) / f"IC_{tag}_{j:03d}.xlsx"
        df_norm.to_excel(fp, index=False)
        filepaths.append(str(fp))
    return filepaths


def recover(Sample, filepaths, data_states, args):
    """Notebook cells 43 (after fit_and_save_base) and 45, verbatim settings."""
    sub = Sample._subsample(filepaths, Sample.L_orig, Sample.U_orig)
    sub["time"] = range(1, len(sub) + 1)
    model = Recover_Model_sindy(
        sub[["time"] + STATES], differential_order=2, poly_degree=args.degree,
        threshold=args.threshold, Xdot=sub[[f"d{s}/dt" for s in STATES]],
        method="Legendre", after_threshold=args.after_threshold,
        L=data_states.min(), U=data_states.max(), denormalize=True)
    return model


def score(model):
    """Per-equation comparison with TRUE_MODEL."""
    rec = model.get_final_coefficients(as_dataframe=False)
    rows = []
    for s in STATES:
        got, true = rec.get(s, {}), TRUE_MODEL[s]
        extra = sorted(set(got) - set(true))
        missing = sorted(set(true) - set(got))
        rel = [abs(got[t] - c) / abs(c) for t, c in true.items() if t in got]
        rows.append({
            "eq": f"d{s}/dt",
            "correct_terms": not extra and not missing,
            "n_extra": len(extra),
            "n_missing": len(missing),
            "max_rel_coef_err": max(rel) if rel else np.nan,
            "recovered": model.model_expression.loc[STATES.index(s), "Equation"],
        })
    return rows


def run_case(label, make_files, data, args, **meta):
    t0 = time.time()
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet if not args.verbose else sys.stdout):
        Sample = sampling(data, args.num, distribution="uniform", noise_level=None)
        filepaths = make_files(Sample)
        model = recover(Sample, filepaths, data[STATES], args)
    rows = score(model)
    for r in rows:
        r.update(case=label, **meta)
    n_ok = sum(r["correct_terms"] for r in rows)
    print(f"{label:28s} equations with exactly the true terms: {n_ok}/4   ({time.time() - t0:.0f}s)")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omegas", type=float, nargs="+", default=[100, 1000, 10000],
                   help="SSA system sizes (molecules per unit concentration)")
    p.add_argument("--replicates", type=int, nargs="+", default=[1],
                   help="SSA runs averaged per initial condition (1 = single noisy trajectory)")
    p.add_argument("--ssa-seeds", type=int, default=3, help="independent SSA repeats per setting")
    p.add_argument("--num", type=int, default=35, help="number of uniformly sampled ICs (notebook: 35)")
    p.add_argument("--end", type=float, default=0.01, help="length of each short trajectory (notebook: 0.01)")
    p.add_argument("--n-step", type=int, default=10, help="points per short trajectory (notebook: 10)")
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--threshold", type=float, default=1e-1)
    p.add_argument("--after-threshold", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=HERE / "output", help="where trajectories and results go")
    p.add_argument("--verbose", action="store_true", help="show the repo code's own printouts")
    args = p.parse_args()

    data = base_data()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    rows += run_case(
        "deterministic (ODE)",
        lambda S: S.fit_and_save_base(model_name="CRN", out_path=args.out / "ode", start=0, end=args.end,
                                      n_step=args.n_step, k_rates=K_RATES, method="Legendre", noise_level=None),
        data, args, omega=np.inf, replicates=0, seed=-1)

    for omega in args.omegas:
        for R in args.replicates:
            for seed in range(args.ssa_seeds):
                label = f"SSA omega={omega:g} R={R} seed={seed}"
                sub = args.out / f"ssa_omega{omega:g}_R{R}_seed{seed}"
                rows += run_case(
                    label,
                    lambda S, sub=sub, omega=omega, R=R, seed=seed: save_ssa_trajectories(
                        S, sub, 0, args.end, args.n_step, omega, R, seed),
                    data, args, omega=omega, replicates=R, seed=seed)

    res = pd.DataFrame(rows)
    res = res.assign(num=args.num, end=args.end, n_step=args.n_step, degree=args.degree)
    res.to_csv(args.out / "recovery_results.csv", index=False)

    summary = (res.groupby(["omega", "replicates", "seed"], sort=False)
               .agg(eqs_correct=("correct_terms", "sum"), extra_terms=("n_extra", "sum"),
                    missing_terms=("n_missing", "sum"), max_rel_coef_err=("max_rel_coef_err", "max"))
               .reset_index())
    pd.set_option("display.width", 200)
    print("\nPer run (4 equations each):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    print("\nRecovered equations, deterministic vs first SSA seed at each setting:")
    show = res[(res["seed"] <= 0)]
    for (case,), grp in show.groupby(["case"], sort=False):
        print(f"  {case}")
        for _, r in grp.iterrows():
            eq = r["recovered"] if len(r["recovered"]) < 110 else r["recovered"][:107] + "..."
            print(f"    {r['eq']} = {eq}")
    print("\nTrue model:")
    for s, terms in TRUE_MODEL.items():
        print(f"    d{s}/dt = " + " + ".join(f"{c:.3f} {t}" for t, c in terms.items()).replace("+ -", "- "))
    print(f"\nFull results: {args.out / 'recovery_results.csv'}")


if __name__ == "__main__":
    main()
