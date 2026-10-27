"""Table 1 (LV, CRN) with the settings of Plots-1_modify.ipynb and L2 column scaling.

SINDy settings follow Plots-1_modify.ipynb (cells 4, 7, 22, 40, and the markdown in
cell 39). Each Table 1 row was produced by re-running the notebook with its own
threshold, so SINDy is fitted once per equation. The Wrong | Missing sublibrary is
scaled by its column L2 norms (no centering) before the condition number is taken.
Run from any directory: python /path/to/Dependence/Exp_table1_lv_crn_l2.py
Requires the adjacent repository modules and the project's Python environment.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "table1-l2-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "table1-l2-cache"))
import numpy as np
import pandas as pd
import pysindy as ps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Base_test import Lotka_Volterra, CRN
from Comparison import Terms_Analysis, Terms_Identification_original
from Recover_Model import Recover_Model_sindy
from dae_finder import PolyFeatureMatrix

ROOT = Path(__file__).resolve().parents[1]
LV_PARAMS = dict(r=.2, K=1.7, a1=.1, a2=.2, m1=.1, m2=.1, epsilon1=1., epsilon2=2.)
CRN_PARAMS = dict(k=2.12456, kr=1.10124, kcat=1.5093)
LV_INITIAL = [1.7, 1.0, 1.7]
LV_TIMES = np.linspace(0, 20, 21)                      # cell 7: one point per year
CRN_INITIAL = [2.5, 2, 1, 0]
CRN_HOURS = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 47, 51])
CRN_TIMES = 10 / 51 * CRN_HOURS                        # cell 4: final_time = 10

# Per-equation thresholds. LV: markdown of cell 39 (cell 40 hard-codes 2.7e-2).
# CRN: cell 22 uses 8e-2, which gives the dx4/dt row; 0.1 (dx1/dt) and 0.09
# (dx2/dt, dx3/dt) are the thresholds whose recovered equations match Table 1.
SPECS = [
    dict(name="LV", degree=3, alpha=.1,
         thresholds={"dx1/dt": 3e-2, "dx2/dt": 3e-3, "dx3/dt": 2.7e-2}),
    dict(name="CRN", degree=2, alpha=None,
         thresholds={"dx1/dt": .1, "dx2/dt": .09, "dx3/dt": .09, "dx4/dt": .08}),
]


class TermsOnly(Terms_Analysis):
    """Reuse the notebook's exact term parser without computing its z-score SVD."""
    def _SVD_analysis(self):
        return {}


def datasets():
    # Equations transcribed verbatim from Model_LV_1_2.xlsx / Model_CRN.xlsx.
    # Embedding them avoids dependencies on absolute paths outside this repo.
    lv = Lotka_Volterra(LV_PARAMS, (0, 20), LV_TIMES, LV_INITIAL, noise_level=None).data_sim
    crn = CRN(CRN_PARAMS, CRN_INITIAL, CRN_TIMES, noise_level=None).data_sim
    frames = {
        "LV": (lv.rename(columns=dict(x="x1", y1="x2", y2="x3")),
               ["r*x1*(1 - x1/K)-a1*x1*x2-a2*x1*x3", "x2*(-m1+epsilon1*a1*x1)",
                "x3*(-m2+epsilon2*a2*x1)"]),
        "CRN": (crn.rename(columns=dict(S="x1", E="x2", ES="x3", P="x4")),
                ["kr*x3 - k*x1*x2", "(kr + kcat)*x3 - k*x1*x2",
                 "k*x1*x2 - (kr + kcat)*x3", "kcat*x3"]),
    }
    for spec in SPECS:
        frame, rhs = frames[spec["name"]]
        states = [f"x{i+1}" for i in range(len(rhs))]
        truth = pd.DataFrame({"Variable": [f"d{x}/dt" for x in states], "Equation": rhs})
        yield spec, frame[["time", *states]].copy(), truth


def l2_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] == 0:
        return 0., 0, [], "no_wrong_or_missing_terms"
    norms = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all():
        raise ValueError("Nonfinite sublibrary or column norm")
    scaled = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * max(scaled.shape) * np.finfo(float).eps))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if np.any(norms == 0):
        condition = float("inf")
    return condition, rank, singular.tolist(), "zero_column" if np.any(norms == 0) else "computed"


def canonical(term):
    # Match the canonical library labels used by Terms_Analysis.
    return " ".join(sorted(term.split(), key=lambda v: int(v.split("^")[0][1:])))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for spec, data, truth in datasets():
        name, degree, alpha = spec["name"], spec["degree"], spec["alpha"]
        states = data.drop(columns="time")
        data.to_csv(output / f"{name}_trajectory.csv", index=False)
        truth.to_csv(output / f"{name}_true_equations.csv", index=False)
        mapping, _ = Terms_Identification_original(truth)._parse_model()
        library = PolyFeatureMatrix(degree).fit_transform(states).drop(columns="1")
        for index, (eq, threshold) in enumerate(spec["thresholds"].items()):
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                recovered = Recover_Model_sindy(
                    data, differential_order=2, poly_degree=degree, method="Monomial",
                    threshold=threshold, alpha=alpha, Xdot=None, after_threshold=None,
                    L=states.min(), U=states.max(), denormalize=False, debug=False)
                analyzer = TermsOnly(truth, recovered.model_expression, mapping, library)
            tag = f"{name}_{eq[1:3]}_threshold{threshold:g}"
            recovered.model_expression.to_csv(output / f"{tag}_equations.csv", index=False)
            np.savez(output / f"{tag}_fit.npz",
                     coefficients=recovered.model.coefficients(),
                     powers=recovered.model.feature_library.powers_)
            runs.append(dict(model=name, equation=eq, degree=degree, threshold=threshold,
                             optimizer_params=recovered.model.optimizer.get_params(),
                             warnings=[str(w.message) for w in caught]))
            # Sorted columns keep the result independent of set iteration order.
            terms = sorted(canonical(t) for t in analyzer.Mixed_terms[eq])
            condition, rank, singular, status = l2_diagnostics(library.loc[:, terms].to_numpy())
            rows.append(dict(model=name, degree=degree, threshold=threshold, alpha=alpha,
                equation=eq, discovered_equation=recovered.model_expression.iloc[index, 1],
                wrong_terms=json.dumps(sorted(analyzer.wrong_terms[eq])),
                missing_terms=json.dumps(sorted(analyzer.missing_terms[eq])),
                mixed_terms=json.dumps(terms), n_wrong=len(analyzer.wrong_terms[eq]),
                n_missing=len(analyzer.missing_terms[eq]), n_samples=len(data),
                n_mixed_terms=len(terms), rank=rank, full_column_rank=rank == len(terms),
                condition_l2=condition, singular_values=json.dumps(singular), status=status))
            print(f"{name} {eq} threshold={threshold:g}: done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "equation_diagnostics.csv", index=False)
    sources = [Path(__file__), *[ROOT / "Dependence" / f for f in
        ["Base_test.py", "Comparison.py", "Recover_Model.py", "Definitions.py", "dae_finder.py",
         "Plots-1_modify.ipynb"]]]
    metadata = dict(pysindy_version=ps.__version__, numpy_version=np.__version__,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        lv_parameters=LV_PARAMS, crn_parameters=CRN_PARAMS,
        settings=dict(lv_initial=LV_INITIAL, lv_times=LV_TIMES.tolist(),
            crn_initial=CRN_INITIAL, crn_times=CRN_TIMES.tolist(), noise=None,
            library="Monomial", include_bias=False, state_normalization=False,
            specs=SPECS, derivative="FiniteDifference(order=2), Xdot=None",
            support="Existing Recover_Model_sindy rounded equations and Terms_Analysis parser",
            condition="Wrong | Missing columns divided by L2 norms; no centering",
            empty_support_condition=0, runs=runs))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(frame[["model", "equation", "threshold", "discovered_equation", "missing_terms",
                     "wrong_terms", "rank", "n_mixed_terms", "condition_l2"]].to_string(index=False))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/table1_lv_crn_l2_plots1")
    run(parser.parse_args().output_dir)
