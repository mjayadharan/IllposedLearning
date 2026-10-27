"""Joint STLSQ grid on original CRN/LV trajectories, with raw-state libraries.

No initial-condition sampling, state normalization, column normalization or
denormalization is applied. Legendre/Chebyshev coefficients are expanded into
monomials of the very same raw coordinates. Success requires all equations
to have exactly the true term sets after a stated fixed coefficient cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cache = Path(tempfile.gettempdir()) / "original_raw_sensitivity_cache"
cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.optimizers import SINDyOptimizer
from threadpoolctl import threadpool_limits

from Dependence.remark3_experiment import _model_spec, _original_data, OrthogonalLibrary
from Dependence.sindy_sensitivity_support import (
    build_conversion, convert_coefficients, true_coefficients, support_metrics,
    verify_conversion,
)

THRESHOLDS = sorted({0., 10., *[float(f"{m}e{e}") for e in range(-8, 1) for m in (1, 2, 5)]})
ALPHAS = [0., 1e-10, 1e-8, 1e-6, 1e-4, .001, .01, .05, .1, 1., 10., 100.]
BASES = ["Monomial", "Legendre", "Chebyshev"]


def term_name(power):
    return " ".join(f"x{i+1}" + (f"^{p}" if p > 1 else "")
                    for i, p in enumerate(power) if p) or "1"


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rows, validations, best_equations, library_rows = [], [], [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for model_name in ["crn", "lv"]:
            spec = _model_spec(model_name)
            original, _, _, _ = _original_data(spec)
            original.to_csv(out / f"{model_name}_original_states.csv", index=False)
            x = original.loc[:, list(spec.state_names)].to_numpy()
            t = original.time.to_numpy()
            y = np.asarray(ps.FiniteDifference(order=2)(x, t))
            pd.DataFrame(y, columns=[f"d{s}/dt" for s in spec.state_names]).to_csv(
                out / f"{model_name}_finite_difference_derivatives.csv", index=False)
            n_states = x.shape[1]
            # Identity substitution: midpoint=0, half_range=1. These bounds
            # configure coefficient expansion only; x and y remain untouched.
            identity_lower, identity_upper = -np.ones(n_states), np.ones(n_states)
            cutoffs = [0., .1, .8, 1.] if model_name == "crn" else [0., .08]
            for basis in BASES:
                library = (ps.PolynomialLibrary(5, include_bias=False) if basis == "Monomial"
                           else OrthogonalLibrary(5, method=basis, include_bias=True))
                library.fit(x)
                theta = np.asarray(library.transform(x))
                matrix, powers = build_conversion(library, basis, identity_lower, identity_upper)
                truth = true_coefficients(model_name, powers, spec.parameters)
                assert [{tuple(p) for p in powers[truth[i] != 0]} for i in range(n_states)] == [set(s) for s in spec.true_support]
                check = verify_conversion(library, basis, identity_lower, identity_upper, matrix, powers)
                raw_monomials = np.prod(x.astype(np.longdouble)[:, None, :] ** powers[None, :, :], axis=2)
                np.testing.assert_allclose(raw_monomials @ matrix, theta, rtol=1e-10, atol=1e-10)
                sv = np.linalg.svd(theta, compute_uv=False)
                rank = int((sv > sv[0] * max(theta.shape) * np.finfo(float).eps).sum())
                library_rows.append(dict(model=model_name, basis=basis, rows=len(x), columns=theta.shape[1], numerical_rank=rank))
                best = {}
                coefficients, parameter_pairs = [], []
                for threshold, alpha in itertools.product(THRESHOLDS, ALPHAS):
                    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=20, normalize_columns=False)
                    error = ""
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            fitted = SINDyOptimizer(optimizer, unbias=True).fit(theta, y)
                            coef = fitted.coef_.copy()
                            raw_coef = convert_coefficients(coef, matrix, basis, identity_lower, identity_upper)
                            if threshold == (.1 if model_name == "crn" else .01) and alpha == .05:
                                reference = ps.SINDy(feature_library=library,
                                    differentiation_method=ps.FiniteDifference(order=2),
                                    optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=20, normalize_columns=False))
                                reference.fit(x, t=t, unbias=True)
                                np.testing.assert_allclose(coef, reference.coefficients(), rtol=1e-9, atol=1e-9)
                                validations.append(dict(model=model_name, basis=basis,
                                    full_sindy_fit_max_difference=float(np.max(np.abs(coef-reference.coefficients()))),
                                    **check))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            raw_coef = None
                    parameter_pairs.append([threshold, alpha])
                    coefficients.append(np.asarray(raw_coef, float) if raw_coef is not None else np.full(truth.shape, np.nan))
                    for cutoff in cutoffs:
                        metrics = support_metrics(raw_coef, truth, cutoff) if raw_coef is not None else {}
                        correct = []
                        if raw_coef is not None:
                            correct = np.all((np.abs(raw_coef) > cutoff) == (truth != 0), axis=1).tolist()
                            assert sum(not v for v in correct) == metrics["wrong_equations"]
                        row = dict(model=model_name, basis=basis, state_coordinates="raw",
                            source="original", n_samples=len(x), n_features=theta.shape[1], degree=5,
                            threshold=threshold, alpha=alpha, effective_alpha=float(optimizer.alpha),
                            max_iter=20, normalize_columns=False, unbias=True,
                            derivative="finite_difference_order_2", after_threshold=cutoff,
                            all_equations_correct=bool(correct and all(correct)),
                            equation_correct=json.dumps(correct), fit_error=error,
                            warning_count=len(caught), warnings=" | ".join(sorted({str(w.message) for w in caught})), **metrics)
                        rows.append(row)
                        if raw_coef is not None:
                            score = (metrics["wrong_equations"], metrics["false_positives"]+metrics["false_negatives"])
                            if cutoff not in best or score < best[cutoff][0]:
                                best[cutoff] = (score, row.copy(), raw_coef.copy())
                for cutoff, (_, row, coef) in best.items():
                    eqs = []
                    for i in range(n_states):
                        expected, predicted = set(map(tuple, powers[truth[i] != 0])), set(map(tuple, powers[np.abs(coef[i]) > cutoff]))
                        eqs.append(dict(equation=f"dx{i+1}/dt", correct=expected == predicted,
                            missing=sorted(term_name(p) for p in expected-predicted),
                            extra=sorted(term_name(p) for p in predicted-expected),
                            retained_terms={term_name(p): float(c) for p,c in zip(powers,coef[i]) if abs(c)>cutoff}))
                    best_equations.append(dict(model=model_name,basis=basis,after_threshold=cutoff,
                        threshold=row["threshold"],alpha=row["alpha"],wrong_equations=row["wrong_equations"],equations=eqs))
                np.savez_compressed(out/f"{model_name}_{basis}_coefficients.npz",
                    parameters=np.asarray(parameter_pairs), raw_coefficients=np.stack(coefficients), powers=powers, truth=truth)
                print(f"{model_name}/{basis}: {len(parameter_pairs)} fits; minimum wrong by cutoff: "
                      f"{ {k:v[0][0] for k,v in best.items()} }", flush=True)
                pd.DataFrame(rows).to_csv(out/"grid_results.csv",index=False)
    results=pd.DataFrame(rows)
    summary=results.groupby(["model","basis","after_threshold"],as_index=False).agg(
        n_settings=("threshold","size"),fit_errors=("fit_error",lambda s:int(s.ne("").sum())),
        all_equations_success_settings=("all_equations_correct","sum"),
        minimum_wrong_equations=("wrong_equations","min"))
    summary.to_csv(out/"grid_summary.csv",index=False)
    pd.DataFrame(library_rows).to_csv(out/"library_diagnostics.csv",index=False)
    (out/"best_equations.json").write_text(json.dumps(best_equations,indent=2)+"\n")
    files=[Path(__file__)]+[ROOT/"Dependence"/name for name in ["remark3_experiment.py","sindy_sensitivity_support.py","Base_test.py","Basis.py","Recover_Model.py","Plots_2.ipynb"]]
    metadata=dict(source="Original single noiseless trajectory, no resampling; all libraries evaluated on RAW states.",
        thresholds=THRESHOLDS,alphas=ALPHAS,regression_fits=len(THRESHOLDS)*len(ALPHAS)*6,
        degree=5,normalize_columns=False,unbias=True,max_iter=20,derivative="FiniteDifference(order=2), endpoints retained",
        metric="Every equation's retained raw-coordinate monomial support must match truth; coefficient values and signs are not compared.",
        post_threshold_note="Each cutoff is reported separately and is not optimized inside a threshold/alpha comparison. Zero cutoff is an additional exact-nonzero diagnostic.",
        conversion="Identity affine map in conversion helper; no input/output rescaling or denormalization.",
        limitations=["Finite grid, not proof of impossibility for all hyperparameters.","Single original trajectory per model, no random-seed replication.","Full-precision coefficient scoring; no equation-string rounding."],
        pysindy=ps.__version__,numpy=np.__version__,elapsed_seconds=time.perf_counter()-started,
        validations=validations,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(summary.to_string(index=False),flush=True)
    if results.fit_error.ne("").any():
        raise RuntimeError("Some fits failed; inspect grid_results.csv before interpreting.")
    assert len(validations)==6


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/original_raw_sensitivity")
    args=parser.parse_args()
    run(args.output_dir.expanduser().resolve())
