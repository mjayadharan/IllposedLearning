import Definitions
from scipy.linalg import svd
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import os
from math import ceil
from threadpoolctl import threadpool_limits
import gc
from math import comb as n_choose_k

# --- Optional GPU (CuPy / PyTorch-MPS/CUDA) support ---
import importlib
# CuPy (CUDA on NVIDIA, Linux/Windows only)
try:
    cp = importlib.import_module("cupy")  # type: ignore[import-not-found]
    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False

# PyTorch (MPS on Apple Silicon, or CUDA if available)
try:
    import torch  # type: ignore
    _has_mps = getattr(torch.backends, 'mps', None)
    _mps_ok = bool(_has_mps and torch.backends.mps.is_available())
    _cuda_ok = torch.cuda.is_available()
    HAS_TORCH = _mps_ok or _cuda_ok
    TORCH_DEVICE = (
        torch.device('mps') if _mps_ok else (torch.device('cuda') if _cuda_ok else torch.device('cpu'))
    )
except Exception:
    torch = None  # type: ignore
    HAS_TORCH = False
    TORCH_DEVICE = None

from Basis import normalization, Monomials, OrthogonalLibrary

class SVD_analysis:
    """
    Create term combinations and perform numerically stable SVD analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe (candidate function library)
    comb : int
        Number of terms in each combination
    preprocessing : str
        Preprocessing method for numerical stability
    
    Returns
    -------
    dict
        Dictionary with combination results including processed and original and SVD analysis
    """
    def __init__(self, data, comb, preprocessing='standardize', threshold=15,
                 n_jobs=-1, backend='loky', batch_size=8, max_nbytes='16M',
                 chunk_size=1000, blas_threads=1):
        self.data = data
        self.comb = comb
        self.preprocessing = preprocessing
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.max_nbytes = max_nbytes
        self.chunk_size = int(chunk_size)
        self.blas_threads = int(blas_threads)
        
        self.combination_results_processed, self.combination_results_original = self._create_combinations_with_stable_svd()
        self.filtered_result, self.dropped_result = self._filter_combinations_cond()

    def _create_combinations_with_stable_svd(self):
        # Preprocess data for numerical stability
        processed_data, preproc_info = Definitions.preprocess_for_stable(self.data, self.preprocessing)
        
        if self.comb > len(processed_data.columns):
            raise ValueError(f"Combination size ({self.comb}) cannot exceed number of terms ({len(processed_data.columns)})")
        
        # Get all combinations
        column_names = processed_data.columns.tolist()
        all_combinations = list(combinations(range(len(column_names)), self.comb))

        # Define the single combination task
        def _single_combo_svd(data_vals, column_names, combo_indices):
            cols = list(combo_indices)
            combo_matrix = data_vals[:, cols]
            with threadpool_limits(limits=self.blas_threads):
                s = np.linalg.svd(combo_matrix, compute_uv=False)
            return {
                'terms'             : tuple(column_names[i] for i in cols),
                'columns'           : tuple(column_names[i] for i in cols),
                'matrix'            : combo_matrix,
                'singular_values'   : s,
                'min_singular_value': s[-1],
                'max_singular_value': s[0],
                'condition_number'  : s[0] / s[-1],
            }
        
        # Parallel computation of SVD for each combination, chunked
        data_vals_processed = processed_data.values
        data_vals_original = self.data.values

        combo_meta_processed = []
        combo_meta_original = []
        n_total = len(all_combinations)
        for start in range(0, n_total, self.chunk_size):
            chunk = all_combinations[start:start + self.chunk_size]
            # processed
            res_p = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer='processes',
                             batch_size=self.batch_size, max_nbytes=self.max_nbytes)(
                delayed(_single_combo_svd)(data_vals_processed, column_names, combo) for combo in chunk
            )
            combo_meta_processed.extend(res_p)
            # original
            res_o = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer='processes',
                             batch_size=self.batch_size, max_nbytes=self.max_nbytes)(
                delayed(_single_combo_svd)(data_vals_original, column_names, combo) for combo in chunk
            )
            combo_meta_original.extend(res_o)

        combination_results_processed = {}
        combination_results_original = {}
        for i,meta in enumerate(combo_meta_processed):
            combination_results_processed[i] = {
                **meta,
                'preprocessing_info':preproc_info
            }
        for i,meta in enumerate(combo_meta_original):
            combination_results_original[i] = {
                **meta,
                'preprocessing_info':'None'
        }
        
        return combination_results_processed,combination_results_original
    
    def _filter_combinations_cond(self):
        """
        Filter combinations with condition numbers larger than the threshold.
        
        Parameters:
        -----------
        combination_results : dict
            Results from create_combinations_with_stable_svd
        
        Returns:
        --------
        dict
            Dictionary containing filtered combinations that meet the threshold criteria
        """
        filtered_results = {}
        dropped_results = {}
        for combo_id, result_processed in self.combination_results_processed.items():
            result_original = self.combination_results_original.get(combo_id)
            condition_number = result_processed['condition_number']
            if condition_number > self.threshold:
                filtered_results[combo_id] = result_original.copy()
            else:
                dropped_results[combo_id] = result_original.copy()
        return filtered_results, dropped_results
    

class Regression_analysis:
    """
    Create term combinations and perform numerically stable regression analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe (candidate function library)
    comb : int
        Number of terms in each combination
    preprocessing : str
        Preprocessing method for numerical stability
    
    Returns
    -------
    dict
        Dictionary with combination results including processed and original and SVD analysis
    """
    def __init__(self, data, comb, preprocessing='standardize', threshold=0.95,
                 n_jobs=-1, backend='loky', batch_size=8, max_nbytes='16M',
                 chunk_size=1000, blas_threads=1, array_dtype='float32',
                 keep_equation=True, keep_coef=True):
        self.data = data
        self.comb = comb
        self.preprocessing = preprocessing
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.max_nbytes = max_nbytes
        self.chunk_size = int(chunk_size)
        self.blas_threads = int(blas_threads)
        self.array_dtype = str(array_dtype)
        self.keep_equation = bool(keep_equation)
        self.keep_coef = bool(keep_coef)
        # Avoid BLAS oversubscription by default
        os.environ.setdefault('OMP_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(self.blas_threads))

        self.run()
        self.filtered_result = self.filter_by_r2(threshold=self.threshold)

    def _create_combinations_with_reg(self, test_size=0.33, random_state=27):
        """Generate combinations for BOTH ORIGINAL and PROCESSED data in parallel with low memory.
        For each combination of size `self.comb`, try each term as response and
        keep the regression with the highest R^2.
        Returns ((results_original_list, results_original_dict), (results_processed_list, results_processed_dict)).
        """
        # Preprocess once for processed branch
        processed_data, preproc_info = Definitions.preprocess_for_stable(self.data, self.preprocessing)
        column_names = processed_data.columns.tolist()
        n_cols = len(column_names)
        if self.comb > n_cols:
            raise ValueError(f"Combination size ({self.comb}) cannot exceed number of terms ({n_cols})")

        # Cast to desired dtype to reduce memory
        data_vals_proc = processed_data.values.astype(self.array_dtype, copy=False)
        data_vals_orig = self.data[column_names].values.astype(self.array_dtype, copy=False)

        # One deterministic split shared by all combos (same indices applied to both branches)
        n_samples = data_vals_proc.shape[0]
        idx = np.arange(n_samples)
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

        # Capture small scalars to avoid pickling the whole self
        blas_thr = int(self.blas_threads)
        keep_eq = self.keep_equation
        keep_cf = self.keep_coef
        k = int(self.comb)

        # Streaming chunker for combinations to avoid building a huge list in memory
        def _combo_chunks(n_features, k, chunk_size):
            from itertools import combinations
            buf = []
            combo_id = 0
            for combo in combinations(range(n_features), k):
                buf.append((combo_id, combo))
                combo_id += 1
                if len(buf) >= chunk_size:
                    yield buf
                    buf = []
            if buf:
                yield buf

        def _process_one(pair, data_vals, preproc_label):
            combo_id, idx_tuple = pair
            cols = list(idx_tuple)
            terms = tuple(column_names[i] for i in cols)
            X_combo = data_vals[:, cols]
            best = None
            for target_local_idx in range(len(cols)):
                target_name = terms[target_local_idx]
                feat_local_idx = [i for i in range(len(cols)) if i != target_local_idx]
                feat_names = [terms[i] for i in feat_local_idx]
                X_train = X_combo[train_idx][:, feat_local_idx]
                y_train = X_combo[train_idx, target_local_idx]
                X_test  = X_combo[test_idx][:,  feat_local_idx]
                y_test  = X_combo[test_idx,  target_local_idx]
                try:
                    with threadpool_limits(limits=blas_thr):
                        mdl = LinearRegression(fit_intercept=True)
                        mdl.fit(X_train, y_train)
                        y_pred = mdl.predict(X_test)
                    r2  = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                except Exception:
                    continue
                equation = None
                if keep_eq:
                    parts = []
                    for c, fn in zip(mdl.coef_, feat_names):
                        if abs(c) < 1e-12:
                            continue
                        sign = '+' if c >= 0 else '-'
                        parts.append(f" {sign} {abs(c):.6g}*{fn}")
                    rhs = '0' if not parts else ''.join(parts).lstrip()
                    if abs(mdl.intercept_) > 1e-12:
                        sign0 = '+' if mdl.intercept_ >= 0 else '-'
                        rhs += f" {sign0} {abs(mdl.intercept_):.6g}"
                    equation = f"{target_name} = {rhs}"
                candidate = {
                    'combination_id' : combo_id,
                    'terms'          : terms,
                    'target'         : target_name,
                    'predictors'     : tuple(feat_names),
                    'coef'           : mdl.coef_.tolist() if keep_cf else None,
                    'intercept'      : float(mdl.intercept_),
                    'r2'             : float(r2),
                    'mse'            : float(mse),
                    'preprocessing'  : preproc_label,
                    'equation'       : equation,
                }
                if (best is None) or (candidate['r2'] > best['r2']):
                    best = candidate
            if best is None:
                best = {
                    'combination_id' : combo_id,
                    'terms'          : terms,
                    'target'         : None,
                    'predictors'     : tuple(terms),
                    'coef'           : None,
                    'intercept'      : np.nan,
                    'r2'             : np.nan,
                    'mse'            : np.nan,
                    'preprocessing'  : preproc_label,
                    'equation'       : None,
                }
            return combo_id, best

        # Total number of combos for result allocation (without materializing them)
        n_total = n_choose_k(n_cols, k)
        results_orig_list = [None] * n_total
        results_orig_dict = {}
        results_proc_list = [None] * n_total
        results_proc_dict = {}

        for chunk in _combo_chunks(n_cols, k, self.chunk_size):
            # processed
            res_proc = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer='processes',
                                 batch_size=self.batch_size, max_nbytes=self.max_nbytes)(
                delayed(_process_one)(pair, data_vals_proc, preproc_info) for pair in chunk
            )
            # original
            res_orig = Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer='processes',
                                 batch_size=self.batch_size, max_nbytes=self.max_nbytes)(
                delayed(_process_one)(pair, data_vals_orig, 'None') for pair in chunk
            )
            for combo_id, best in res_proc:
                results_proc_list[combo_id] = best
                results_proc_dict[combo_id] = best
            for combo_id, best in res_orig:
                results_orig_list[combo_id] = best
                results_orig_dict[combo_id] = best
            del res_proc, res_orig
            gc.collect()

        return (results_orig_list, results_orig_dict), (results_proc_list, results_proc_dict)

    def run(self, test_size=0.33, random_state=27, return_df=True):
        """Run regression analysis and return only PROCESSED results (for stability).
        """
        (results_original, results_original_dict), (results_processed, results_processed_dict) = \
            self._create_combinations_with_reg(test_size=test_size, random_state=random_state)

        # Attach to instance
        self.results_original = results_original
        self.results_original_dict = results_original_dict
        self.results_processed = results_processed
        self.results_processed_dict = results_processed_dict

        if return_df:
            df_original = pd.DataFrame(results_original)
            df_processed = pd.DataFrame(results_processed)
            self.results_original_df = df_original
            self.results_processed_df = df_processed
            return (
                {'original':results_original, 'processed': results_processed},
                {'original': df_original, 'processed': df_processed}
            )
        else:
            return {'original':results_original, 'preprocessed':results_processed}

# (old filter_by_r2 removed)

    def filter_by_r2(self, threshold: float, n_jobs: int | None = None):
        """
        Filter combinations in results_processed with R^2 >= threshold, then use the
        selected combination_ids to extract the corresponding combinations from
        results_original. Optionally parallelize the final filtering step when
        results are huge.

        Parameters
        ----------
        threshold : float
            Keep processed combinations with R^2 >= this threshold.
        n_jobs : int or None
            If provided and >1, split the original DataFrame into chunks and
            filter in parallel (useful for very large tables). Defaults to None
            which uses a single vectorized pass.

        Returns
        -------
        pandas.DataFrame
            Filtered combinations from results_original whose combination_id is
            selected by the processed-side R^2 filter.
        """
        # Default to class-level n_jobs when not provided
        if n_jobs is None:
            n_jobs = self.n_jobs
        # Ensure we have run results and DataFrames
        if not hasattr(self, 'results_processed') or self.results_processed is None:
            self.run(return_df=True)
        # Build processed/original DataFrames if not already present
        df_proc = getattr(self, 'results_processed_df', None)
        if df_proc is None:
            df_proc = pd.DataFrame(self.results_processed)
            self.results_processed_df = df_proc
        # Guard: reconstruct processed DataFrame if missing/empty or built from dict-of-dicts
        if df_proc is None or not isinstance(df_proc, pd.DataFrame) or df_proc.empty or ('r2' not in df_proc.columns and isinstance(getattr(self, 'results_processed', None), (list, dict))):
            rp = getattr(self, 'results_processed', None)
            if isinstance(rp, list):
                df_proc = pd.DataFrame(rp)
            elif isinstance(rp, dict):
                df_proc = pd.DataFrame(list(rp.values()))
            else:
                df_proc = pd.DataFrame()
            self.results_processed_df = df_proc
        df_orig = getattr(self, 'results_original_df', None)
        if df_orig is None:
            df_orig = pd.DataFrame(self.results_original)
            self.results_original_df = df_orig

        # If processed results are truly empty, short-circuit with empty outputs
        rp = getattr(self, 'results_processed', None)
        processed_is_truly_empty = (
            df_proc is None or df_proc.empty or (
                'r2' not in df_proc.columns and (
                    rp is None or (isinstance(rp, (list, tuple, dict)) and len(rp) == 0)
                )
            )
        )
        if processed_is_truly_empty:
            empty_cols = ['combination_id','terms','target','predictors','coef','intercept','r2','preprocessing','equation']
            empty_df = pd.DataFrame(columns=empty_cols)
            self.filtered_combo_ids = []
            self.filtered_processed_df = empty_df.copy()
            self.filtered_result = empty_df.copy()
            return empty_df.copy()

        # Basic column checks
        if 'r2' not in df_proc.columns:
            raise ValueError(f"results_processed DataFrame does not contain 'r2' column. Available columns: {list(df_proc.columns)}")
        if 'combination_id' not in df_proc.columns:
            raise ValueError("results_processed DataFrame does not contain 'combination_id' column")
        if 'combination_id' not in df_orig.columns:
            raise ValueError("results_original DataFrame does not contain 'combination_id' column")

        # 1) Filter processed by R^2 threshold
        df_proc_filt = df_proc[df_proc['r2'] >= float(threshold)].copy()
        combo_ids = sorted(df_proc_filt['combination_id'].unique().tolist())

        # 2) Use these ids to filter original (vectorized or parallel chunked)
        if n_jobs is not None and n_jobs > 1 and len(df_orig) > 10000:
            chunks = np.array_split(df_orig, n_jobs)
            def _filter_chunk(chunk):
                return chunk[chunk['combination_id'].isin(combo_ids)]
            parts = Parallel(n_jobs=n_jobs, backend=self.backend, prefer='processes')(
                delayed(_filter_chunk)(c) for c in chunks
            )
            filtered_result = pd.concat(parts, ignore_index=True)
        else:
            filtered_result = df_orig[df_orig['combination_id'].isin(combo_ids)].copy()

        filtered_result = filtered_result.reset_index(drop=True)

        # Store helpful attributes
        self.filtered_combo_ids = combo_ids
        self.filtered_processed_df = df_proc_filt.reset_index(drop=True)
        self.filtered_result = filtered_result

        return filtered_result


# --------------------- Fast correlation-matrix-based regression analysis ---------------------
class Regression_analysis_advanced:
    def _torch_prescreen_pairs(self, X_train_proc, threshold):
        """PyTorch prescreen for pairs on processed-train: return np.ndarray [[i,j,r2_train], ...] with i<j.
        Works on MPS (Apple Silicon) or CUDA if available.
        """
        if not HAS_TORCH or TORCH_DEVICE is None:
            return None
        import torch
        X = torch.as_tensor(X_train_proc, device=TORCH_DEVICE, dtype=torch.float32)
        # Correlation on train
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        sig = Xc.std(dim=0, unbiased=False)
        sig = torch.clamp(sig, min=1e-15)
        Z = Xc / sig
        C = (Z.T @ Z) / Z.shape[0]
        C = torch.clamp(C, -1.0, 1.0)
        C.fill_diagonal_(1.0)
        R2 = C * C
        iu, iv = torch.triu_indices(C.shape[0], C.shape[1], offset=1, device=TORCH_DEVICE)
        r2_vec = R2[iu, iv]
        mask = r2_vec >= threshold
        if mask.sum().item() == 0:
            return np.empty((0, 3), dtype=float)
        i_sel = iu[mask].to('cpu').numpy()
        j_sel = iv[mask].to('cpu').numpy()
        r2_sel = r2_vec[mask].to('cpu').numpy()
        return np.stack([i_sel, j_sel, r2_sel], axis=1)

    def _torch_prescreen_triplets(self, X_train_proc, threshold, topk=50):
        """PyTorch prescreen for triplets on processed-train: return np.ndarray [[y,u,v,r2_train], ...].
        Uses Top-K neighborhood per y; works with MPS/CUDA.
        """
        if not HAS_TORCH or TORCH_DEVICE is None:
            return None
        import torch
        X = torch.as_tensor(X_train_proc, device=TORCH_DEVICE, dtype=torch.float32)
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        sig = Xc.std(dim=0, unbiased=False)
        sig = torch.clamp(sig, min=1e-15)
        Z = Xc / sig
        C = (Z.T @ Z) / Z.shape[0]
        C = torch.clamp(C, -1.0, 1.0)
        C.fill_diagonal_(1.0)
        p = C.shape[0]
        absC = torch.abs(C)
        hits = []
        for y in range(p):
            # Top-K neighbors excluding y
            #vals, idx = torch.topk(absC[y], k=min(int(topk) if topk else p-1, p-1))
            # Remove y if present (topk may include y if topk==p-1)
            #mask_not_y = idx != y
            #S = idx[mask_not_y]
            order = torch.argsort(absC[y], descending=True)
            order = order[order != y]                 # First, eliminate y.
            if topk:                                  # topk=None indicates selecting all p-1 neighbors.
                order = order[:min(int(topk), p-1)]
            S = order
            K = S.numel()
            if K < 2:
                continue
            iu, iv = torch.triu_indices(K, K, offset=1, device=TORCH_DEVICE)
            ryS = C[y, S]  # (K,)
            a = ryS[:, None]
            b = ryS[None, :]
            c = C.index_select(0, S).index_select(1, S)
            num = a*a + b*b - 2*a*b*c
            den = torch.clamp(1 - c*c, min=1e-15)
            R2 = num / den
            r2_vec = R2[iu, iv]
            m = r2_vec >= threshold
            if m.sum().item() == 0:
                continue
            u_sel = S[iu[m]].to('cpu').numpy()
            v_sel = S[iv[m]].to('cpu').numpy()
            r2_sel = r2_vec[m].to('cpu').numpy()
            y_col = np.full_like(u_sel, int(y))
            hits.append(np.stack([y_col, u_sel, v_sel, r2_sel], axis=1))
        if not hits:
            return np.empty((0, 4), dtype=float)
        return np.concatenate(hits, axis=0)
    """
    Fast regression analysis using correlation-matrix closed forms (no per-combo fitting).

    Supports pair (comb=2) and triplet (comb=3) combinations. For each combination,
    evaluate all choices of the dependent variable y and keep the relation with the
    largest R^2. R^2 and coefficients are computed from the correlation matrix and
    (mu, sigma) without performing OLS fits.

    Evaluation Modes:
    -----------------
    - Method A (default): Use all data to compute correlation/statistics and evaluate R^2 (closed-form, train R^2).
    - Method B (use_split=True): Data is split into train/test. Closed-form coefficients are computed from training statistics,
      and the evaluation R^2 can be computed on the test set (r2_eval='test') or on the training set (r2_eval='train').
      This allows assessment of out-of-sample performance using closed-form regression.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe (candidate function library)
    comb : int
        Number of terms in each combination (must be 2 or 3)
    preprocessing : str
        Preprocessing method for numerical stability
    threshold : float
        R^2 threshold for keeping a combination (applied after evaluation)
    n_jobs, backend, batch_size, max_nbytes, chunk_size, blas_threads, array_dtype : parallel and numeric options
    keep_equation, keep_coef : store equation string/coefficients in results
    tie_rule_pairs : how to choose y in pairs when variances are tied ('larger_var' or 'first')
    use_split : bool, default True
        Whether to split data into train/test and evaluate R^2 on test set (Method B). If False, uses full data (Method A).
    test_size : float, default 0.33
        Fraction of data to use as test set if use_split is True.
    random_state : int, default 27
        Random seed for train/test split.
    r2_eval : {'test','train'}, default 'test'
        If use_split is True, whether to evaluate R^2 on test set ('test') or training set ('train').
    """

    def __init__(self, data, comb, preprocessing='standardize', threshold=0.95,
                 n_jobs=-1, backend='loky', batch_size=8, max_nbytes='16M',
                 chunk_size=1000, blas_threads=1, array_dtype='float64',
                 keep_equation=True, keep_coef=True, tie_rule_pairs='larger_var',
                 use_gpu: bool = True, gpu_topk: int | None = 50,
                 use_split=True, test_size=0.33, random_state=27, r2_eval='test'):
        self.data = data
        self.comb = int(comb)
        if self.comb not in (2, 3):
            raise ValueError("Regression_analysis_advanced supports comb=2 or comb=3 only")
        self.preprocessing = preprocessing
        self.threshold = float(threshold)
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.max_nbytes = max_nbytes
        self.chunk_size = int(chunk_size)
        self.blas_threads = int(blas_threads)
        self.array_dtype = str(array_dtype)
        self.keep_equation = bool(keep_equation)
        self.keep_coef = bool(keep_coef)
        self.tie_rule_pairs = tie_rule_pairs  # 'larger_var' or 'first'
        self.use_gpu = bool(use_gpu)
        self.gpu_topk = None if gpu_topk is None else int(gpu_topk)
        self.use_split = bool(use_split)
        self.test_size = float(test_size)
        self.random_state = random_state
        assert r2_eval in ('test', 'train')
        self.r2_eval = r2_eval

        # Limit BLAS oversubscription (consistent with Regression_analysis)
        os.environ.setdefault('OMP_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(self.blas_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(self.blas_threads))

        # Compute results now
        self.run()
        # And compute filtered_result as in Regression_analysis
        self.filtered_result = self.filter_by_r2(threshold=self.threshold)

    # --------------------- helpers (static) ---------------------
    @staticmethod
    def _corrcoef_from_array(X):
        """Return (C, mu, sig) where C is feature-wise correlation matrix."""
        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=0)
        Xc = X - mu
        sig = Xc.std(axis=0, ddof=0)
        # Avoid zeros
        sig_safe = np.clip(sig, 1e-15, None)
        Z = Xc / sig_safe
        C = (Z.T @ Z) / Z.shape[0]
        np.fill_diagonal(C, 1.0)
        C = np.clip(C, -1.0, 1.0)
        return C, mu, sig

    @staticmethod
    def _corrcoef_xp(X, xp):
        """Backend-agnostic correlation: returns (C, mu, sig) on the provided xp (np/cp)."""
        X = xp.asarray(X)
        mu = X.mean(axis=0)
        Xc = X - mu
        sig = Xc.std(axis=0, ddof=0)
        Z = Xc / xp.maximum(sig, 1e-15)
        C = (Z.T @ Z) / Z.shape[0]
        if xp is np:
            np.fill_diagonal(C, 1.0)
            C = np.clip(C, -1.0, 1.0)
        else:
            cp.fill_diagonal(C, 1.0)
            C = cp.clip(C, -1.0, 1.0)
        return C, mu, sig

    @staticmethod
    def _r2_from_corr(a, b=None, c=None):
        """
        Closed-form R^2.
        - If b is None: pair case, return a^2 where a=r_yx.
        - Else: triplet case with a=r_yx1, b=r_yx2, c=r_x1x2.
        """
        if b is None:
            r = a
            return float(np.clip(r*r, 0.0, 1.0))
        # triplet
        denom = 1.0 - c*c
        if denom < 1e-15:
            denom = 1e-15
        num = a*a + b*b - 2.0*a*b*c
        r2 = num / denom
        return float(np.clip(r2, 0.0, 1.0))

    @staticmethod
    def _beta_pair_from_corr(r_yx, sig_y, sig_x, mu_y, mu_x):
        beta = r_yx * (sig_y / (sig_x + 1e-15))
        alpha = mu_y - beta * mu_x
        return float(beta), float(alpha)

    @staticmethod
    def _beta_triplet_from_corr(a, b, c, sig_y, sig_u, sig_v, mu_y, mu_u, mu_v):
        # standardized betas
        denom = 1.0 - c*c
        if denom < 1e-15:
            denom = 1e-15
        b1_std = (a - b*c) / denom
        b2_std = (b - a*c) / denom
        # unstandardize
        b1 = b1_std * (sig_y / (sig_u + 1e-15))
        b2 = b2_std * (sig_y / (sig_v + 1e-15))
        alpha = mu_y - (b1 * mu_u + b2 * mu_v)
        return float(b1), float(b2), float(alpha)

    def _gpu_prescreen_pairs(self, X_train_proc, threshold):
        """GPU prescreen for pairs on processed-train: return np.ndarray [[i,j,r2_train], ...] with i<j."""
        if not HAS_CUPY:
            return None
        xp = cp
        C, _, _ = self._corrcoef_xp(X_train_proc, xp)
        R2 = C * C  # elementwise square
        # Only take upper triangle (i<j)
        iu, iv = xp.triu_indices(R2.shape[0], k=1)
        r2_vec = R2[iu, iv]
        mask = r2_vec >= threshold
        if not bool(mask.any()):
            return np.empty((0, 3), dtype=float)
        out = xp.stack([iu[mask], iv[mask], r2_vec[mask]], axis=1)
        return cp.asnumpy(out)

    def _gpu_prescreen_triplets(self, X_train_proc, threshold, topk=50):
        """GPU prescreen for triplets on processed-train: return np.ndarray [[y,u,v,r2_train], ...] with u<v and all in neigh[y]."""
        if not HAS_CUPY:
            return None
        xp = cp
        C, _, _ = self._corrcoef_xp(X_train_proc, xp)
        p = C.shape[0]
        absC = xp.abs(C)
        hits = []
        for y in range(p):
            order = xp.argsort(absC[y])[::-1]
            S = order[order != y][:min(topk or (p-1), p-1)]
            K = S.size
            if K < 2:
                continue
            iu, iv = xp.triu_indices(K, k=1)
            ryS = C[y, S]            # (K,)
            a = ryS[:, None]
            b = ryS[None, :]
            c = C[S[:, None], S[None, :]]
            num = a*a + b*b - 2*a*b*c
            den = xp.maximum(1 - c*c, 1e-15)
            R2 = num / den
            r2_vec = R2[iu, iv]
            mask = r2_vec >= threshold
            if not bool(mask.any()):
                continue
            u_sel = S[iu[mask]]
            v_sel = S[iv[mask]]
            r2_sel = r2_vec[mask]
            hits.append(xp.stack([xp.full_like(u_sel, y), u_sel, v_sel, r2_sel], axis=1))
        if not hits:
            return np.empty((0, 4), dtype=float)
        gpu_arr = xp.concatenate(hits, axis=0)
        return cp.asnumpy(gpu_arr)

    @staticmethod
    def _combo_chunks(n_features, k, chunk_size):
        from itertools import combinations
        buf = []
        combo_id = 0
        for combo in combinations(range(n_features), k):
            buf.append((combo_id, combo))
            combo_id += 1
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    # --------------------- core computation ---------------------
    def _create_combinations_corr_based(self):
        """
        Generate combinations for BOTH ORIGINAL and PROCESSED branches using
        correlation-based closed forms (no per-combo fitting).

        Returns ((results_original_list, results_original_dict),
                 (results_processed_list, results_processed_dict)).
        """
        # preprocess once
        processed_data, preproc_info = Definitions.preprocess_for_stable(self.data, self.preprocessing)
        column_names = processed_data.columns.tolist()
        n_cols = len(column_names)
        if self.comb > n_cols:
            raise ValueError(f"Combination size ({self.comb}) cannot exceed number of terms ({n_cols})")

        # cast arrays
        arr_proc = processed_data.values.astype(self.array_dtype, copy=False)
        arr_orig = self.data[column_names].values.astype(self.array_dtype, copy=False)

        # Method B: train/test split and stats on train
        if self.use_split:
            n_samples = arr_proc.shape[0]
            idx = np.arange(n_samples)
            train_idx, test_idx = train_test_split(idx, test_size=self.test_size, random_state=self.random_state, shuffle=True)
            arr_proc_tr = arr_proc[train_idx]
            arr_proc_te = arr_proc[test_idx]
            arr_orig_tr = arr_orig[train_idx]
            arr_orig_te = arr_orig[test_idx]
            C_proc, mu_proc, sig_proc = self._corrcoef_from_array(arr_proc_tr)
            C_orig, mu_orig, sig_orig = self._corrcoef_from_array(arr_orig_tr)
        else:
            arr_proc_te = arr_proc
            arr_orig_te = arr_orig
            C_proc, mu_proc, sig_proc = self._corrcoef_from_array(arr_proc)
            C_orig, mu_orig, sig_orig = self._corrcoef_from_array(arr_orig)

        k = self.comb
        n_total = n_choose_k(n_cols, k)
        # Processed: keep full size (我们会计算全部组合)
        results_proc_list = [None] * n_total
        results_proc_dict = {}
        # Original: 按需回填（只为命中组合计算，不预分配）
        results_orig_list = []
        results_orig_dict = {}

        # local shortcuts to avoid pickling self
        tie_rule_pairs = self.tie_rule_pairs
        keep_eq = self.keep_equation
        keep_cf = self.keep_coef
        threshold = self.threshold
        backend = self.backend
        batch_size = self.batch_size
        max_nbytes = self.max_nbytes
        n_jobs = self.n_jobs


        def _format_equation_pair(target_name, beta, intercept, feat_name):
            if not keep_eq:
                return None
            rhs = f"{beta:+.6g}*{feat_name}"
            if abs(intercept) > 1e-12:
                rhs += f" {intercept:+.6g}"
            # strip leading '+'
            rhs = rhs.lstrip()
            return f"{target_name} = {rhs}"

        def _format_equation_triplet(target_name, coef_tuple, intercept, feat_names):
            if not keep_eq:
                return None
            parts = []
            for c, fn in zip(coef_tuple, feat_names):
                if abs(c) < 1e-12:
                    continue
                parts.append(f" {c:+.6g}*{fn}")
            rhs = '0' if not parts else ''.join(parts)
            if abs(intercept) > 1e-12:
                rhs += f" {intercept:+.6g}"
            rhs = rhs.strip()
            return f"{target_name} = {rhs}"

        def _process_one_branch(pair, C, mu, sig, preproc_label, X_test=None):
            combo_id, idx_tuple = pair
            cols = list(idx_tuple)
            terms = tuple(column_names[i] for i in cols)

            if k == 2:
                i, j = cols
                # For pairs: evaluate both choices of dependent variable
                candidates = []
                for (y_idx, x_idx) in [(i, j), (j, i)]:
                    y_name = column_names[y_idx]
                    x_name = column_names[x_idx]
                    r = C[y_idx, x_idx]
                    # Closed-form coefficients from train
                    beta, alpha = self._beta_pair_from_corr(r, sig[y_idx], sig[x_idx], mu[y_idx], mu[x_idx])
                    # Evaluate R^2
                    if self.use_split and self.r2_eval == 'test' and X_test is not None:
                        y_test = X_test[:, y_idx]
                        x_test = X_test[:, x_idx]
                        yhat = alpha + beta * x_test
                        resid = y_test - yhat
                        ss_res = np.sum(resid * resid, dtype=np.float64)
                        center = y_test - np.mean(y_test)
                        ss_tot = np.sum(center * center, dtype=np.float64)
                        r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
                    else:
                        r2_eval = self._r2_from_corr(r)
                    eq = _format_equation_pair(y_name, beta, alpha, x_name)
                    candidate = {
                        'combination_id': combo_id,
                        'terms': terms,
                        'target': y_name,
                        'predictors': (x_name,),
                        'coef': [beta] if keep_cf else None,
                        'intercept': alpha,
                        'r2': float(r2_eval),
                        'preprocessing': preproc_label,
                        'equation': eq,
                    }
                    candidates.append(candidate)
                # Pick candidate with larger r2
                best = max(candidates, key=lambda d: d['r2'] if d['r2'] is not None else -np.inf)
                # Only after picking, compare r2 to threshold
                if best['r2'] < threshold:
                    empty = {
                        'combination_id': combo_id,
                        'terms': terms,
                        'target': None,
                        'predictors': tuple(terms),
                        'coef': None,
                        'intercept': np.nan,
                        'r2': np.nan,
                        'preprocessing': preproc_label,
                        'equation': None,
                    }
                    return combo_id, empty
                return combo_id, best

            # k == 3
            i, j, k3 = cols
            # For triplets: evaluate all 3 choices of dependent variable
            candidates = []
            # y=i, X=(j,k3)
            a = C[i, j]; b = C[i, k3]; c = C[j, k3]
            y_idx, u_idx, v_idx = i, j, k3
            # Closed-form coefficients
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            # Evaluate R^2
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                resid = y_test - yhat
                ss_res = np.sum(resid * resid, dtype=np.float64)
                center = y_test - np.mean(y_test)
                ss_tot = np.sum(center * center, dtype=np.float64)
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # y=j, X=(i,k3)
            a = C[j, i]; b = C[j, k3]; c = C[i, k3]
            y_idx, u_idx, v_idx = j, i, k3
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                resid = y_test - yhat
                ss_res = np.sum(resid * resid, dtype=np.float64)
                center = y_test - np.mean(y_test)
                ss_tot = np.sum(center * center, dtype=np.float64)
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # y=k3, X=(i,j)
            a = C[k3, i]; b = C[k3, j]; c = C[i, j]
            y_idx, u_idx, v_idx = k3, i, j
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                resid = y_test - yhat
                ss_res = np.sum(resid * resid, dtype=np.float64)
                center = y_test - np.mean(y_test)
                ss_tot = np.sum(center * center, dtype=np.float64)
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # Pick candidate with largest r2
            best = max(candidates, key=lambda d: d['r2'] if d['r2'] is not None else -np.inf)
            if best['r2'] < threshold:
                empty = {
                    'combination_id': combo_id,
                    'terms': terms,
                    'target': None,
                    'predictors': tuple(terms),
                    'coef': None,
                    'intercept': np.nan,
                    'r2': np.nan,
                    'preprocessing': preproc_label,
                    'equation': None,
                }
                return combo_id, empty
            return combo_id, best

        # Optional GPU/Torch prescreen: compute only for combos likely to pass threshold on processed branch
        if self.use_gpu and (HAS_CUPY or HAS_TORCH):
            selected_pairs = []  # sequence of (combo_id, idx_tuple)
            if k == 2:
                hits = None
                if HAS_CUPY:
                    hits = self._gpu_prescreen_pairs(arr_proc_tr if self.use_split else arr_proc, threshold)
                if (hits is None or (isinstance(hits, np.ndarray) and hits.size == 0)) and HAS_TORCH:
                    hits = self._torch_prescreen_pairs(arr_proc_tr if self.use_split else arr_proc, threshold)
                hits = hits if hits is not None else np.empty((0, 3), dtype=float)
                for cid, (i, j, r2t) in enumerate(hits):
                    selected_pairs.append((int(cid), (int(i), int(j))))
            elif k == 3:
                hits = None
                if HAS_CUPY:
                    hits = self._gpu_prescreen_triplets(arr_proc_tr if self.use_split else arr_proc, threshold, topk=self.gpu_topk)
                if (hits is None or (isinstance(hits, np.ndarray) and hits.size == 0)) and HAS_TORCH:
                    hits = self._torch_prescreen_triplets(arr_proc_tr if self.use_split else arr_proc, threshold, topk=self.gpu_topk)
                hits = hits if hits is not None else np.empty((0, 4), dtype=float)
                # Deduplicate by sorted (i,j,k)
                seen = {}
                for row in hits:
                    y, u, v = map(int, row[:3])
                    key = tuple(sorted((y, u, v)))
                    if key not in seen:
                        seen[key] = True
                for cid, key in enumerate(seen.keys()):
                    selected_pairs.append((int(cid), key))
            # Build processed results ONLY for selected_pairs using CPU helper (with test-R2 if configured)
            results_proc_list = []
            results_proc_dict = {}
            for pair in selected_pairs:
                combo_id, best = _process_one_branch(pair, C_proc, mu_proc, sig_proc, preproc_info,
                                                     arr_proc_te if self.use_split else None)
                results_proc_list.append(best)
                results_proc_dict[combo_id] = best
            # Stage B: original on demand for selected only
            results_orig_list = []
            results_orig_dict = {}
            for pair in selected_pairs:
                combo_id, best = _process_one_branch(pair, C_orig, mu_orig, sig_orig, 'None',
                                                     arr_orig_te if self.use_split else None)
                results_orig_list.append(best)
                results_orig_dict[combo_id] = best
            return (results_orig_list, results_orig_dict), (results_proc_list, results_proc_dict)

        def _format_equation_pair(target_name, beta, intercept, feat_name):
            if not keep_eq:
                return None
            rhs = f"{beta:+.6g}*{feat_name}"
            if abs(intercept) > 1e-12:
                rhs += f" {intercept:+.6g}"
            # strip leading '+'
            rhs = rhs.lstrip()
            return f"{target_name} = {rhs}"

        def _format_equation_triplet(target_name, coef_tuple, intercept, feat_names):
            if not keep_eq:
                return None
            parts = []
            for c, fn in zip(coef_tuple, feat_names):
                if abs(c) < 1e-12:
                    continue
                parts.append(f" {c:+.6g}*{fn}")
            rhs = '0' if not parts else ''.join(parts)
            if abs(intercept) > 1e-12:
                rhs += f" {intercept:+.6g}"
            rhs = rhs.strip()
            return f"{target_name} = {rhs}"

        def _process_one_branch(pair, C, mu, sig, preproc_label, X_test=None):
            combo_id, idx_tuple = pair
            cols = list(idx_tuple)
            terms = tuple(column_names[i] for i in cols)

            if k == 2:
                i, j = cols
                # For pairs: evaluate both choices of dependent variable
                candidates = []
                for (y_idx, x_idx) in [(i, j), (j, i)]:
                    y_name = column_names[y_idx]
                    x_name = column_names[x_idx]
                    r = C[y_idx, x_idx]
                    # Closed-form coefficients from train
                    beta, alpha = self._beta_pair_from_corr(r, sig[y_idx], sig[x_idx], mu[y_idx], mu[x_idx])
                    # Evaluate R^2
                    if self.use_split and self.r2_eval == 'test' and X_test is not None:
                        y_test = X_test[:, y_idx]
                        x_test = X_test[:, x_idx]
                        yhat = alpha + beta * x_test
                        ss_res = ((y_test - yhat) ** 2).sum()
                        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                        r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
                    else:
                        r2_eval = self._r2_from_corr(r)
                    eq = _format_equation_pair(y_name, beta, alpha, x_name)
                    candidate = {
                        'combination_id': combo_id,
                        'terms': terms,
                        'target': y_name,
                        'predictors': (x_name,),
                        'coef': [beta] if keep_cf else None,
                        'intercept': alpha,
                        'r2': float(r2_eval),
                        'preprocessing': preproc_label,
                        'equation': eq,
                    }
                    candidates.append(candidate)
                # Pick candidate with larger r2
                best = max(candidates, key=lambda d: d['r2'] if d['r2'] is not None else -np.inf)
                # Only after picking, compare r2 to threshold
                if best['r2'] < threshold:
                    empty = {
                        'combination_id': combo_id,
                        'terms': terms,
                        'target': None,
                        'predictors': tuple(terms),
                        'coef': None,
                        'intercept': np.nan,
                        'r2': np.nan,
                        'preprocessing': preproc_label,
                        'equation': None,
                    }
                    return combo_id, empty
                return combo_id, best

            # k == 3
            i, j, k3 = cols
            # For triplets: evaluate all 3 choices of dependent variable
            candidates = []
            # y=i, X=(j,k3)
            a = C[i, j]; b = C[i, k3]; c = C[j, k3]
            y_idx, u_idx, v_idx = i, j, k3
            # Closed-form coefficients
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            # Evaluate R^2
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                ss_res = ((y_test - yhat) ** 2).sum()
                ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # y=j, X=(i,k3)
            a = C[j, i]; b = C[j, k3]; c = C[i, k3]
            y_idx, u_idx, v_idx = j, i, k3
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                ss_res = ((y_test - yhat) ** 2).sum()
                ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # y=k3, X=(i,j)
            a = C[k3, i]; b = C[k3, j]; c = C[i, j]
            y_idx, u_idx, v_idx = k3, i, j
            b1, b2, alpha = self._beta_triplet_from_corr(
                a, b, c,
                sig[y_idx], sig[u_idx], sig[v_idx],
                mu[y_idx],  mu[u_idx],  mu[v_idx]
            )
            if self.use_split and self.r2_eval == 'test' and X_test is not None:
                y_test = X_test[:, y_idx]
                xu = X_test[:, u_idx]
                xv = X_test[:, v_idx]
                yhat = alpha + b1 * xu + b2 * xv
                ss_res = ((y_test - yhat) ** 2).sum()
                ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                r2_eval = 1.0 - ss_res / (ss_tot + 1e-15)
            else:
                r2_eval = self._r2_from_corr(a, b, c)
            eq = _format_equation_triplet(column_names[y_idx], (b1, b2), alpha, (column_names[u_idx], column_names[v_idx]))
            candidates.append({
                'combination_id': combo_id,
                'terms': terms,
                'target': column_names[y_idx],
                'predictors': (column_names[u_idx], column_names[v_idx]),
                'coef': [b1, b2] if keep_cf else None,
                'intercept': alpha,
                'r2': float(r2_eval),
                'preprocessing': preproc_label,
                'equation': eq,
            })
            # Pick candidate with largest r2
            best = max(candidates, key=lambda d: d['r2'] if d['r2'] is not None else -np.inf)
            if best['r2'] < threshold:
                empty = {
                    'combination_id': combo_id,
                    'terms': terms,
                    'target': None,
                    'predictors': tuple(terms),
                    'coef': None,
                    'intercept': np.nan,
                    'r2': np.nan,
                    'preprocessing': preproc_label,
                    'equation': None,
                }
                return combo_id, empty
            return combo_id, best

        # ========== Stage A: processed only, collect pass-list ==========
        selected_pairs = []  # list of (combo_id, idx_tuple) for combos passing threshold
        for chunk in self._combo_chunks(n_cols, self.comb, self.chunk_size):
            # Establish a mapping between block IDs and combinations to facilitate recording passed idx_tuple entries.
            id2combo = {cid: cmb for (cid, cmb) in chunk}

            if self.n_jobs is not None and self.n_jobs != 1:
                res_p = Parallel(n_jobs=n_jobs, backend=backend, prefer='processes',
                                 batch_size=batch_size, max_nbytes=max_nbytes)(
                    delayed(_process_one_branch)(pair, C_proc, mu_proc, sig_proc, preproc_info,
                                                 arr_proc_te if self.use_split else None)
                    for pair in chunk
                )
            else:
                res_p = [
                    _process_one_branch(pair, C_proc, mu_proc, sig_proc, preproc_info,
                                         arr_proc_te if self.use_split else None)
                    for pair in chunk
                ]

            # Write the processed results and record combinations that pass the threshold (saving their idx_tuple).
            for combo_id, best in res_p:
                results_proc_list[combo_id] = best
                results_proc_dict[combo_id] = best
                r2v = best.get('r2', np.nan)
                if np.isfinite(r2v) and r2v >= threshold:
                    selected_pairs.append((combo_id, id2combo[combo_id]))

        # ========== Stage B: original on demand (only selected combos) ==========
        if selected_pairs:
            # Process selected combinations in batches to avoid submitting too many tasks at once.
            def _chunk_pairs(pairs, size):
                for i in range(0, len(pairs), size):
                    yield pairs[i:i+size]

            for sel_chunk in _chunk_pairs(selected_pairs, self.chunk_size):
                if self.n_jobs is not None and self.n_jobs != 1:
                    res_o = Parallel(n_jobs=n_jobs, backend=backend, prefer='processes',
                                     batch_size=batch_size, max_nbytes=max_nbytes)(
                        delayed(_process_one_branch)(pair, C_orig, mu_orig, sig_orig, 'None',
                                                     arr_orig_te if self.use_split else None)
                        for pair in sel_chunk
                    )
                else:
                    res_o = [
                        _process_one_branch(pair, C_orig, mu_orig, sig_orig, 'None',
                                            arr_orig_te if self.use_split else None)
                        for pair in sel_chunk
                    ]
                for combo_id, best in res_o:
                    results_orig_list.append(best)
                    results_orig_dict[combo_id] = best
        # If there are no matches, the original branch remains an empty list/empty dictionary.

        return (results_orig_list, results_orig_dict), (results_proc_list, results_proc_dict)

    # --------------------- public API ---------------------
    def run(self, return_df=True):
        (results_original, results_original_dict), (results_processed, results_processed_dict) = \
            self._create_combinations_corr_based()

        # Attach to instance
        self.results_original = results_original
        self.results_original_dict = results_original_dict
        self.results_processed = results_processed
        self.results_processed_dict = results_processed_dict

        if return_df:
            df_original = pd.DataFrame(results_original)
            df_processed = pd.DataFrame(results_processed)
            self.results_original_df = df_original
            self.results_processed_df = df_processed
            return (
                {'original': results_original, 'processed': results_processed},
                {'original': df_original, 'processed': df_processed}
            )
        else:
            return {'original': results_original, 'processed': results_processed}

    def filter_by_r2(self, threshold: float, n_jobs: int | None = None):
        """
        Same semantics as Regression_analysis.filter_by_r2: filter processed-side
        results by R^2 >= threshold, then select rows with those combination_ids
        from the original-side table.
        """
        if n_jobs is None:
            n_jobs = self.n_jobs
        # Ensure results/DataFrames exist
        if not hasattr(self, 'results_processed') or self.results_processed is None:
            self.run(return_df=True)
        df_proc = getattr(self, 'results_processed_df', None)
        if df_proc is None:
            df_proc = pd.DataFrame(self.results_processed)
            self.results_processed_df = df_proc
        # Guard: reconstruct processed DataFrame if missing/empty or built from dict-of-dicts
        if df_proc is None or not isinstance(df_proc, pd.DataFrame) or df_proc.empty or ('r2' not in df_proc.columns and isinstance(getattr(self, 'results_processed', None), (list, dict))):
            rp = getattr(self, 'results_processed', None)
            if isinstance(rp, list):
                df_proc = pd.DataFrame(rp)
            elif isinstance(rp, dict):
                df_proc = pd.DataFrame(list(rp.values()))
            else:
                df_proc = pd.DataFrame()
            self.results_processed_df = df_proc
        df_orig = getattr(self, 'results_original_df', None)
        if df_orig is None:
            df_orig = pd.DataFrame(self.results_original)
            self.results_original_df = df_orig

        # If processed results are truly empty, short-circuit with empty outputs
        rp = getattr(self, 'results_processed', None)
        processed_is_truly_empty = (
            df_proc is None or df_proc.empty or (
                'r2' not in df_proc.columns and (
                    rp is None or (isinstance(rp, (list, tuple, dict)) and len(rp) == 0)
                )
            )
        )
        if processed_is_truly_empty:
            empty_cols = ['combination_id','terms','target','predictors','coef','intercept','r2','preprocessing','equation']
            empty_df = pd.DataFrame(columns=empty_cols)
            self.filtered_combo_ids = []
            self.filtered_processed_df = empty_df.copy()
            self.filtered_result = empty_df.copy()
            return empty_df.copy()

        if 'r2' not in df_proc.columns:
            raise ValueError(f"results_processed DataFrame does not contain 'r2' column. Available columns: {list(df_proc.columns)}")
        if 'combination_id' not in df_proc.columns:
            raise ValueError("results_processed DataFrame does not contain 'combination_id' column")
        if 'combination_id' not in df_orig.columns:
            raise ValueError("results_original DataFrame does not contain 'combination_id' column")

        df_proc_filt = df_proc[df_proc['r2'] >= float(threshold)].copy()
        combo_ids = sorted(df_proc_filt['combination_id'].unique().tolist())

        if n_jobs is not None and n_jobs > 1 and len(df_orig) > 10000:
            chunks = np.array_split(df_orig, n_jobs)
            def _filter_chunk(chunk):
                return chunk[chunk['combination_id'].isin(combo_ids)]
            parts = Parallel(n_jobs=n_jobs, backend=self.backend, prefer='processes')(
                delayed(_filter_chunk)(c) for c in chunks
            )
            filtered_result = pd.concat(parts, ignore_index=True)
        else:
            filtered_result = df_orig[df_orig['combination_id'].isin(combo_ids)].copy()
        filtered_result = filtered_result.reset_index(drop=True)

        self.filtered_combo_ids = combo_ids
        self.filtered_processed_df = df_proc_filt.reset_index(drop=True)
        self.filtered_result = filtered_result
        return filtered_result