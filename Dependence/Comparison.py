import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import UnivariateSpline

from dae_finder import PolyFeatureMatrix
from Multicollinearity import create_combinations_with_stable_svd,filter_combinations
from math import comb as n_choose_k
from joblib import Parallel, delayed

def target_relationship(data,target_col,test_size=0.2,fit_with_intercept=False):
    """
    Fit the relationship within target variables
    Return: 
        dict: Mathematical expression, R² and MSE
    """
    feature_cols = [col for col in data.columns if col != target_col]
    feature_names = feature_cols
    X = data[feature_cols].values
    y = data[target_col].values
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=test_size,random_state=27
    )

    model = LinearRegression(fit_intercept=fit_with_intercept)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)

    expression = f"{target_col} = "
    for i, coef in enumerate(model.coef_):
        if coef >= 0:
            expression += f" + {coef:.4f}*{feature_names[i]}"
        else:
            expression += f" - {abs(coef):.4f}*{feature_names[i]}"

    result = {
        'expression': expression,
        'R²': r2,
        'MSE': mse
    }   
    return result


class TimeSeriesDerivative:
    """
    Time series derivative calculation class.
    Available methods to calculate time series derivate include finite difference and spline interpolation. 
    """
    def __init__(self,data,time_col):
        self.data = data.copy()
        self.time_col = time_col
        self.feature_cols = [col for col in data.columns if col != time_col]
        self.derivative_results = {}
    
    def calculate_derivatives(
            self,method,order=1,**kwargs
    ):
        result = self.data.copy()
        t = self.data[self.time_col].values
        # Calculate derivative for each feature
        for col in self.feature_cols:
            x = self.data[col].values
            if method == 'finite_difference':
                dx_dt = self._finite_difference(t,x,order=1,**kwargs)
            elif method == 'spline':
                dx_dt = self._spline(t,x,order,**kwargs)
            
            derivative_col_name = f"d^{col}/dt"
            result[derivative_col_name] = dx_dt
            result.drop({col},axis=1,inplace=True)
        self.derivative_results = result
        return result
    
    def _finite_difference(self,t:np.ndarray,x:np.ndarray,order=1,scheme='central'):
        # scheme : {'forward','backward','central'}
        n = len(x)
        dx_dt = np.zeros(n)
        if scheme == 'forward':
            for i in range(n-1):
                dx_dt[i] = (x[i+1]-x[i]) / (t[i+1] - t[i])
            dx_dt[-1] = dx_dt[-2] # boundary point
        elif scheme == 'central':
            dx_dt[0] = (x[1] - x[0]) / (t[1] - t[0]) # froward 
            for i in range(1,n-1):
                dx_dt[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
            dx_dt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2]) # backward
        else: 
            dx_dt[0] = dx_dt[1]
            for i in range(1,n-1):
                dx_dt[i] = (x[i]-x[i-1]) / (t[i]-t[i-1])
        return dx_dt
    
    def _spline(self,t:np.ndarray,x:np.ndarray,order=1,smoothing:float = 0):
        try:
            spline = UnivariateSpline(t,x,s=smoothing)
            dx_dt = spline.derivative(n=order)(t)
            return dx_dt
        except Exception as e:
            print(f"Spline interpolation calculation failed, replaced with finite difference method:{e}")
            return self._finite_difference(t,x,order)
        
def compute_time_derivatives(data:pd.DataFrame,time_col:str,method:str,order=1,**method_kwargs):
    derivative_calculator  = TimeSeriesDerivative(data,time_col=time_col)
    result = derivative_calculator.calculate_derivatives(
        method = method,
        order = order,
        **method_kwargs
    )
    return result



class Noise_Free_results:
    """Compare different results without anthropogenic noise"""

    def __init__(self,data,degree_list,comb_list,n_jobs=-1):
        self.data = data
        self.degree_list = degree_list
        self.comb_list = comb_list
        self._library_cahche = {}
        self.n_jobs = n_jobs
        self.Num_library_terms,self.Num_comb,self.Num_ill_comb = self._Numbers()
        self.results = None
    
    def _generate_library(self,degree):
        data_states = self.data.drop(columns=['t_Exp'])

        poly_feature_ob = PolyFeatureMatrix(degree)
        candidate_lib = poly_feature_ob.fit_transform(data_states)
        candidate_lib = candidate_lib.drop(["1"], axis=1)
        
        return candidate_lib
    

    def _process_single_combination(self,degree,comb):
        candidate_lib = self._generate_library(degree)
        # Number of library terms
        num_library_terms = len(candidate_lib.columns)
        # Number of combinations
        num_combinations = n_choose_k(num_library_terms, comb)
        comb_processed,comb_original = create_combinations_with_stable_svd(candidate_lib,comb)
        filtered,_ = filter_combinations(comb_processed,comb_original,threshold=20)
        # Number of ill-posed combinations with comb
        num_ill_comb = len(filtered)

        return num_library_terms, num_combinations, num_ill_comb

    def _run_analysis(self):
        tasks = [(d,c) for d in self.degree_list for c in self.comb_list]
        results_dict = {}
        def _task(degree,comb):
            cand = self._generate_library(degree)
            return (degree, comb, *self._process_single_combination(degree, comb))
        
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(_task)(d, c) for d, c in tasks)
        results_dict = {(d, c): {'library_terms': nlib, 'combinations': ncomb, 'ill_posed': nill}
                        for d, c, nlib, ncomb, nill in results}
        self.results = self._build_summary(results_dict)
        return self.results
    
    def _build_summary(self,results_dict):
        rows = []

        for degree in self.degree_list:
            row = {"degree": degree}
            # key for the number of library terms
            key_lib = (degree,self.comb_list[0])
            row["# lib terms"] = results_dict[key_lib]['library_terms']
            for comb in self.comb_list:
                key = (degree, comb)
                row[f"#{comb}comb"] = results_dict[key]['combinations']
                row[f'# ill-posed {comb}comb'] = results_dict[key]['ill_posed']

            rows.append(row)
        
        summary = pd.DataFrame(rows)
        column_order = ["degree", "# lib terms"]
        column_order.extend([f'#{c}comb' for c in self.comb_list])
        column_order.extend([f'# ill-posed {c}comb' for c in self.comb_list])

        return summary[column_order]


# ---------------------- New function for external noise-free analysis ----------------------
def run_noise_free_analysis(data, degree_list, comb_list,n_jobs = None):
    """
    Wrapper function to run noise-free analysis externally.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    degree_list : list of int
        Polynomial degrees to analyze
    comb_list : list of int
        Combination sizes to analyze

    Returns:
    --------
    Tuple of:
        - Num_library_terms : dict
        - Num_comb : dict
        - Num_ill_comb : dict
    """
    print(">>> Running noise-free analysis...")
    analyzer = Noise_Free_results(data, degree_list, comb_list,n_jobs)
    
    print(">>> Generating candidate libraries...")
    print(f"    Degrees: {degree_list}")
    print(f"    Combinations: {comb_list}")
    
    return analyzer._run_analysis()


