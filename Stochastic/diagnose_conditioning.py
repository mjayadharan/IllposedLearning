"""Why the deterministic Figure 2 CRN test (35 uniform ICs, Legendre, degree 5) recovers 2/4 equations.

Uses the notebook pipeline (sampling.fit_and_save_base with the ODE, _subsample,
Recover_Model_sindy) and compares:
  1. finite-difference derivatives (notebook) vs exact derivatives from the true RHS,
  2. the condition number of the Legendre library on the sampled data,
  3. recovery as the number of sampled ICs grows.

    python Stochastic/diagnose_conditioning.py
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stochastic_crn_recovery as scr  # noqa: E402  (also puts Dependence/ on sys.path)
from Basis import OrthogonalLibrary  # noqa: E402
from Sample_Analysis import sampling  # noqa: E402

DATA = scr.base_data()
L, U = DATA[scr.STATES].min(), DATA[scr.STATES].max()
DCOLS = [f"d{v}/dt" for v in scr.STATES]


def build(num, work):
    with contextlib.redirect_stdout(io.StringIO()):
        S = sampling(DATA, num, distribution="uniform", noise_level=None)
        fps = S.fit_and_save_base(model_name="CRN", out_path=work / f"num{num}", start=0, end=0.01, n_step=10,
                                  k_rates=scr.K_RATES, method="Legendre", noise_level=None)
        sub = S._subsample(fps, S.L_orig, S.U_orig)
    sub["time"] = range(1, len(sub) + 1)
    return S, sub


def exact_xdot(sub, S):
    """True RHS at the sampled points, in the normalized units the trajectory files use."""
    Ls, Us = S.L_orig.values, S.U_orig.values
    s, e, c, _ = (Ls + (sub[scr.STATES].to_numpy() + 1) * (Us - Ls) / 2).T
    f = np.c_[scr.kr * c - scr.k * e * s, (scr.kr + scr.kcat) * c - scr.k * e * s,
              scr.k * e * s - (scr.kr + scr.kcat) * c, scr.kcat * c]
    return pd.DataFrame(f * 2 / (Us - Ls), columns=DCOLS)


def recovered(sub, xdot):
    with contextlib.redirect_stdout(io.StringIO()):
        m = scr.Recover_Model_sindy(sub[["time"] + scr.STATES], differential_order=2, poly_degree=5, threshold=0.1,
                                    Xdot=xdot, method="Legendre", after_threshold=1.0, L=L, U=U, denormalize=True)
    rows = scr.score(m)
    return sum(r["correct_terms"] for r in rows), sum(r["n_extra"] for r in rows)


def cond(sub, degree=5):
    lib = OrthogonalLibrary(degree=degree, method="Legendre", include_bias=True)
    X = sub[scr.STATES].to_numpy()
    lib.fit([X])
    sv = np.linalg.svd(np.asarray(lib.transform([X])[0]), compute_uv=False)
    return sv[0] / sv[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nums", type=int, nargs="+", default=[35, 70, 140, 280])
    p.add_argument("--work", type=Path, default=HERE / "output" / "diagnose_conditioning")
    args = p.parse_args()

    rows = []
    for num in args.nums:
        S, sub = build(num, args.work)
        fd, ex = sub[DCOLS], exact_xdot(sub, S)
        rows.append({
            "ICs": num, "rows": len(sub), "cond(degree-5 Legendre)": f"{cond(sub):.2e}",
            "FD rel. error": f"{np.abs(fd.values - ex.values).max() / np.abs(ex.values).max():.1e}",
            "FD: eqs correct / extra terms": "%d/4, %d" % recovered(sub, fd),
            "exact: eqs correct / extra terms": "%d/4, %d" % recovered(sub, ex),
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
