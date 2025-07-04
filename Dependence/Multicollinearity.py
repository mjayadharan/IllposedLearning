import Definitions
from scipy.linalg import svd
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed


def create_combinations_with_stable_svd(data, comb, preprocessing='standardize',n_jobs=-1,backend='loky'):
    """
    Create term combinations and perform numerically stable SVD analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    comb : int
        Number of terms in each combination
    preprocessing : str
        Preprocessing method for numerical stability
    
    Returns
    -------
    dict
        Dictionary with combination results including processed and original and SVD analysis
    """
    
    # Preprocess data for numerical stability
    processed_data, preproc_info = Definitions.preprocess_for_stable(data, preprocessing)
    
    if comb > len(processed_data.columns):
        raise ValueError(f"Combination size ({comb}) cannot exceed number of terms ({len(processed_data.columns)})")
    
    # Get all combinations
    column_names = processed_data.columns.tolist()
    all_combinations = list(combinations(range(len(column_names)), comb))

    # Define the single combination task
    def _single_combo_svd(data_vals,column_names,combo_indices):
        cols = list(combo_indices)
        combo_matrix = data_vals[:, cols]             
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
    
    # Parallel computation of SVD for each combination
    data_vals_processed = processed_data.values
    combo_meta_processed = Parallel(n_jobs=-1,backend=backend,prefer='processes')(
        delayed(_single_combo_svd)(data_vals_processed,column_names,combo)
        for combo in all_combinations
    )
    data_vals_original = data.values
    combo_meta_original = Parallel(n_jobs=-1,backend=backend,prefer='processes')(
        delayed(_single_combo_svd)(data_vals_original,column_names,combo)
        for combo in all_combinations
    )

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


def filter_combinations(combination_results_processed,combination_results_original,threshold=20):
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
    for combo_id, result_processed in combination_results_processed.items():
        result_original = combination_results_original.get(combo_id)
        condition_number = result_processed['condition_number']
        if condition_number > threshold:
            filtered_results[combo_id] = result_original.copy()
        else:
            dropped_results[combo_id] = result_original.copy()
    return filtered_results, dropped_results


def regression(filtered_results,model,degree,test_size=0.2,n_jobs=-1,fit_with_intercept=False,**model_params):
    """
    Run regression(linear/Lasso/Ridge) for each combination and calculate R^2 and MSE.

    Parameters:
    -----------
    filtered_results: dict
        Results from filter_combinations, which means terms in each combination are linearly dependent.
    model: Type of regression model('linear','lasso','ridge')
    degree: polynomial degree
    fit_with_intercept: bool, default: False
        Whether to include an intercept term in the regression models. 

    **model_params : dict
        Additional parameters for the regression model (e.g., alpha for Lasso/Ridge)

    Returns:
    --------
    dict
        Dictionary with regression results for each combination including R² and MSE
    """

    # Define the single regression task
    def _fit_one_target(combo_id,results,target_idx,model_lower,regression_model,train_idx,test_idx):
        full_matrix = results['matrix']
        column_names = results['columns']
        n_terms = len(column_names)
        feat_idx     = [i for i in range(n_terms) if i != target_idx]
        feat_names   = [column_names[i] for i in feat_idx]
        X_train,X_test      = full_matrix[train_idx][:, feat_idx],full_matrix[test_idx][:,  feat_idx]
        y_train,y_test      = full_matrix[train_idx,  target_idx],full_matrix[test_idx,   target_idx]
        mdl = clone(regression_model)
        mdl.fit(X_train,y_train)
        coef = mdl.coef_
        intercept = mdl.intercept_
        y_pred = mdl.predict(X_test)
        r2  = r2_score(y_test,  y_pred)
        mse = mean_squared_error(y_test, y_pred)

        return {
        'combo_id'   : combo_id,
        'target_idx' : target_idx,
        'target_name': column_names[target_idx],
        'feat_names' : feat_names,
        'coef'       : coef,
        'inter'      : intercept,
        'r2'         : r2,
        'mse'        : mse
    }

    # Model selection : Linear / Lasso / Ridge
    model_lower = model.lower()
    if model_lower == 'linear':
        regression_model = LinearRegression(fit_intercept=fit_with_intercept)
    elif model_lower == 'lasso':
        alpha = model_params.get('alpha',1.0)
        regression_model = Lasso(alpha = alpha,fit_intercept=fit_with_intercept)
    elif model_lower == 'ridge':
        alpha = model_params.get('alpha',1.0)
        regression_model = Ridge(alpha = alpha,fit_intercept=fit_with_intercept)

    # Create parallel tasks (number of combinations * length of terms in each combination, the later one is combo)
    tasks = []
    for combo_id, results in filtered_results.items():
        np.random.seed(27)
        # Pre-split data once
        n_samples = results['matrix'].shape[0]
        n_train = int((1-test_size)*n_samples)
        # Use same random permutation for all regressions in this combination
        # This procedure is aimed at selecting the best regression model for each combination
        perm = np.random.permutation(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        for tar_idx in range(len(results['columns'])):
            tasks.append((
                combo_id,results,tar_idx,
                model_lower,regression_model,
                train_idx, test_idx
            ))
    single_results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
    delayed(_fit_one_target)(*t) for t in tasks
    )

    regression_results = {}

    for res in single_results:
        key = res['combo_id']
        cur_best = regression_results.get(key,None)
        if (cur_best is None) or (res['r2'] > cur_best['r2_test']):
            # Combine the equation
            eq_parts = []
            for i, (c, fn) in enumerate(zip(res['coef'], res['feat_names'])):
                if abs(c) < 1e-10:
                    continue
                sign = '+' if (c >= 0 and i > 0) else ''
                eq_parts.append(f"{sign}{c:.4f}*{fn}")
            if abs(res['inter']) > 1e-10:
                eq_parts.append(f"{'+ ' if res['inter']>=0 else ''}{res['inter']:.4f}")
            equation = f"{res['target_name']} = " + " ".join(eq_parts) if eq_parts else f"{res['target_name']} = 0"

            regression_results[key] = {
                'polynomial_degree': degree,
                'combination_id'   : key,
                'combination_terms': filtered_results[key]['terms'],
                'r2_test'          : res['r2'],
                'mse_test'         : res['mse'],
                'condition_number' : filtered_results[key]['condition_number'],
                'equation'         : equation
            }
    return regression_results


def process_single_degree(poly_degree,candidate_libs,comb=3):
    combination_result = create_combinations_with_stable_svd(
        candidate_libs[poly_degree], comb=comb, preprocessing='None'
    )
    filtered_result, dropped_result = filter_combinations(combination_result)
    regression_result = regression(filtered_result,model='linear',degree=poly_degree)

    return {
        'combination': combination_result,
        'filtered': filtered_result,
        'dropped': dropped_result,
        'regression': regression_result
    }