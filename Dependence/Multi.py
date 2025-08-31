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

    #def filter_by_r2(self, threshold: float, n_jobs: int | None = None):
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

        # Basic column checks
        if 'r2' not in df_proc.columns:
            raise ValueError("results_processed DataFrame does not contain 'r2' column")
        if 'combination_id' not in df_proc.columns:
            raise ValueError("results_processed DataFrame does not contain 'combination_id' column")
        #if 'combination_id' not in df_orig.columns:
            #raise ValueError("results_original DataFrame does not contain 'combination_id' column")
        
        # Filter processed by R^2 threshold
        df_proc_filt = df_proc[df_proc['r2'] >= float(threshold)].copy()
        df_proc_filt = df_proc_filt.reset_index(drop=True)

        self.filtered_combo_ids = sorted(df_proc_filt['combination_id'].unique().tolist())
        self.filtered_processed_df = df_proc_filt
        self.filtered_result = df_proc_filt

        return df_proc_filt

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
        df_orig = getattr(self, 'results_original_df', None)
        if df_orig is None:
            df_orig = pd.DataFrame(self.results_original)
            self.results_original_df = df_orig

        # Basic column checks
        if 'r2' not in df_proc.columns:
            raise ValueError("results_processed DataFrame does not contain 'r2' column")
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