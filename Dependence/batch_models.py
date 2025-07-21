from __future__ import annotations
import argparse
import json
import pathlib
import re
from typing import List, Dict, Any
import math

import pandas as pd
from dae_finder import PolyFeatureMatrix
import Comparison

import concurrent.futures
import os

def load_thresholds(path: str) -> Dict[str, float]:
    """
    Read a text file of the form:
        Beer: 1e-5
        Wine: 2.5e-4
    and return a mapping {model_name: threshold}.
    Lines starting with '#' or blank lines are ignored.
    """
    thresholds: Dict[str, float] = {}
    p = pathlib.Path(path)
    if not p.exists():
        print(f"⚠️  Threshold file {path} not found — all models will use the default threshold 1e-5.")
        return thresholds

    with p.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                print(f"⚠️  Skipping malformed line in threshold file: {line}")
                continue
            key, val = [seg.strip() for seg in line.split(":", 1)]
            key = key.lower()  # case‑insensitive keys
            try:
                thresholds[key] = float(val)
            except ValueError:
                print(f"⚠️  Invalid threshold value for {key}: {val} — line ignored.")
    return thresholds

def adjust_threshold(raw_val):
    # Derive the threshold accoring to the minimum parameter value.
    exp = math.floor(math.log10(raw_val))
    pow10 = 10 ** exp
    if math.isclose(raw_val,pow10,rel_tol=1e-12,abs_tol=0,0):
        return raw_val
    else:
        return 10 ** (exp-1)

def _standardize_columns(df,time_col=None):
    # Convert variables to x1,x2,x3,...
    if time_col is None:
        candidate_names = {"t_Exp","t_fine"}
        found = [c for c in df.columns if c.lower() in candidate_names]
        if not found:
            raise ValueError("Uable to auto-detect time column. Need provide 'time_col'")
        time_col = [found[0]]
    elif isinstance(time_col,str):
        time_col = [time_col]
    var_cols = [c for c in df.columns if c not in time_col]
    new_names = {}
    for idx, col in enumerate(var_cols, start=1):
        if not re.fullmatch(r"x\d+", str(col)):
            new_names[col] = f"x{idx}"
    return df.rename(columns=new_names)

def _process_single_model(
        model_file: pathlib.Path,
        data_file: pathlib.Path,
        degree_range: List[int],
        degree_list: List[int],
        comb_list: List[int],
        out_dir: pathlib.Path,
        threshold_map: Dict[str, float]
):
    name = re.match(r"Model_(.*)\.xlsx", model_file.name).group(1)
    key_name = name.lower()  # match keys in threshold_map
    print(f"\n== Processing {name} ===")

    # 1. Load model expressions and data, identify terms in original model
    model_df = pd.read_excel(model_file)
    data_file = pd.read_excel(data_file)
    data_df = _standardize_columns(data_file,'t_Exp')
    data_states = data_df.copy().drop(columns=["t_Exp"])
    data_derivative = Comparison.compute_time_derivatives(data_df,'t_Exp',method='spline')
    model_terms = Comparison.Terms_Identification(model_df)
    variable_mapping = model_terms._parse_model()[0]

    # Determine the regularization threshold for this model
    if key_name not in threshold_map:
        raise KeyError(
            f"No threshold specified for model '{name}'. "
            "Please add an entry like\n    "
            f"{name}: <value>\n"
            "to your thresholds.txt file."
        )
    threshold_raw = threshold_map[key_name]
    threshold = adjust_threshold(threshold_raw)

    # 2. Derive condition number, wrong/missing terms
    candidate_libs: Dict[int,Any] = {}
    recovered: Dict[int,Any] = {}
    conds, missings, wrongs = {},{},{}

    for deg in degree_range:
        # Generate candidate library
        poly_feature_ob = PolyFeatureMatrix(deg)
        candidate_lib_full = poly_feature_ob.fit_transform(data_states)
        candidate_libs[deg] = candidate_lib_full.drop(["1"],axis=1)
        # Recover model
        recover = Comparison.Recover_Model(candidate_libs[deg], data_derivative, threshold)
        recovered[deg] = recover
        # Analysis
        analyzer = Comparison.Terms_Analysis(model_df,recover.model,variable_mapping,recover.mapping,candidate_libs[deg])
        conds[deg] = analyzer.con
        missings[deg] = analyzer.missing_terms
        wrongs[deg] = analyzer.wrong_terms
    
    # 3. Derive noise-free summary
    summary = Comparison.run_noise_free_analysis(data_df,degree_list,comb_list)

    # 4. Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"Summary_{name}.json"

    # Ensure summary is JSON‑serialisable
    if isinstance(summary, pd.DataFrame):
        summary_payload: Any = summary.to_dict(orient="records")  # list of row‑dicts
    else:
        summary_payload = summary  # assume already serialisable

    with outfile.open("w") as fh:
        json.dump(
            {
                "condition_number": conds,
                "missing_terms": missings,
                "wrong_terms": wrongs,
                "summary": summary_payload,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"Saved → {outfile}")

# Helper for multiprocessing.map
def _process_single_model_wrapper(args_tuple):
    """
    Unpack tuple and call _process_single_model. Needed for Pool.map
    so we can pass multiple arguments easily.
    """
    return _process_single_model(*args_tuple)

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run workflow for all models")
    parser.add_argument("--models_dir", default=".", help="Directory containing Model_*.xlsx")
    parser.add_argument("--output_dir", default="./results", help="Result output directory")
    parser.add_argument(
        "--threshold_file",
        default="./thresholds.txt",
        help="Path to text file mapping <model_name>: <threshold>"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--degree_range",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5],
        help="Degree range used to calculate condition/wrong/missing",
    )
    parser.add_argument(
        "--degree_list",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Degree list used to utilize run_noise_free_analysis",
    )
    parser.add_argument(
        "--comb_list",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Comb list used to utilize run_noise_free_analysis",
    )
    args = parser.parse_args()

    threshold_map = load_thresholds(args.threshold_file)

    models_dir = pathlib.Path(args.models_dir)
    out_dir = pathlib.Path(args.output_dir)

    # Build list of tasks
    task_args = []
    for model_file in sorted(models_dir.glob("Model_*.xlsx")):
        name = re.match(r"Model_(.*)\.xlsx", model_file.name).group(1)
        data_file = models_dir / f"{name}_exp.xlsx"
        if not data_file.exists():
            print(f"⚠️  Missing data file for {name}: {data_file}, skip.")
            continue
        task_args.append(
            (
                model_file,
                data_file,
                args.degree_range,
                args.degree_list,
                args.comb_list,
                out_dir,
                threshold_map,
            )
        )

    # Run in parallel
    if task_args:
        print(f"\n⏱  Launching {len(task_args)} tasks with n_jobs={args.n_jobs} …")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            for _ in pool.map(_process_single_model_wrapper, task_args):
                pass  # simply consume results; _process_single_model prints its own progress
    else:
        print("No valid models found to process.")

if __name__ == "__main__":
    main()
