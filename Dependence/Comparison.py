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

import re

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


# function for external noise-free analysis 
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


class Terms_Identification:
    def __init__(self,model_dataframe):
        # model dataframe normally has two columns.
        # The first column represents LHS of an equation, like dx/dt.
        # The second column represents RHS of an equation, like k1*x1 + k1*k2*x3^2
        self.original_model = model_dataframe

        self.variable_mapping = {} # variables to xi
        self.reverse_mapping = {} # xi to variables
        self.original_equations = {}
        self.all_original_terms = set()
        self.all_xi_terms = set()
    
    def _parse_model(self):
        # Create variables mapping
        self._create_variable_mapping()
        # Extract terms from all equations
        self._extract_all_terms_from_equations()
        # No need to call _convert_terms_to_xi here
        return self.variable_mapping, self.all_xi_terms
    
    def _create_variable_mapping(self):
        variable_order = []
        for i in range(len(self.original_model)):
            var_cell = self.original_model.iloc[i,0]
            eq_cell = self.original_model.iloc[i,1]

            if pd.notna(var_cell) and pd.notna(eq_cell):
                var = str(var_cell).strip()
                eq = str(eq_cell).strip()
                if 'd' in var.lower() and 'dt' in var.lower():
                    var_name = var.replace('/dt','').lstrip('d')
                    variable_order.append(var_name)
                    self.original_equations[var] = eq
        for i,var_name in enumerate(variable_order):
            xi_var = f"x{i+1}"
            self.variable_mapping[xi_var] = var_name
            self.reverse_mapping[var_name] = xi_var
    
    def _extract_all_terms_from_equations(self):
        all_terms = set()
        for var, equation in self.original_equations.items():
            terms = self._extract_terms(equation)
            for term in terms:
                # Convert original variables to xi
                term_xi = self._convert_single_term_to_xi(term)
                # Remove parameters and retain terms only contain states,like x1*x2, x3^2
                term_cleaned = self._strip_parameters(term_xi)
                if term_cleaned:
                    all_terms.add(term_cleaned)
        self.all_original_terms = all_terms
        self.all_xi_terms = all_terms

    def _strip_parameters(self, term: str) -> str:
        # Remove non-state-variable symbols (parameters) from the term, leaving only xi variables.
        term = term.strip()
        if not term:
            return ''
        factors = re.split(r'\*|\s', term)
        xi_factors = []
        xi_vars = self.reverse_mapping.values()
        for f in factors:
            for xi in xi_vars:
                if f.startswith(xi):
                    xi_factors.append(f)
                    break
        return '*'.join(sorted(xi_factors)) if xi_factors else ''

    def _filter_variable_terms(self,terms):
        # Filter terms contain state variables
        filtered = set()
        for term in terms:
            contains_state_var = False
            for var_name in self.reverse_mapping.keys():
                if re.search(r'\b' + re.escape(var_name) + r'\b',term):
                    contains_state_var = True
                    break
            if contains_state_var:
                    filtered.add(term)
        return filtered
    
    def _convert_terms_to_xi(self):
        xi_terms = set()
        for term in self.all_original_terms:
            xi_term = self._convert_single_term_to_xi(term)
            if xi_term:
                xi_terms.add(xi_term)
        self.all_xi_terms = xi_terms

    def _convert_single_term_to_xi(self,term):
        converted_term = term
        sorted_vars = sorted(self.reverse_mapping.keys(),key=len,reverse=True)
        for var_name in sorted_vars:
            xi_var = self.reverse_mapping[var_name]
            pattern = r'\b' + re.escape(var_name) + r'\b'
            converted_term = re.sub(pattern, xi_var, converted_term)
        return converted_term
    
    def _parse_discovered_model(slef):
        return
    
    def _extract_terms(self,equation):
        # Extract individual terms from an equation
        equation = equation.replace(' ','')
        # Expand the equation to prevent identifying (x+y) as a term
        expanded_equation = self._expand_brackets(equation)
        terms = self._split_equation_terms(expanded_equation)
        # Remove coefficients and signs
        cleaned_terms = set()
        for term in terms:
            original_term = term
            term = term.strip()
            while term.startswith('+') or term.startswith('-'):
                term = term[1:]
            term_no_coef = self._remove_coefficients(term)
            is_constant = self._is_pure_constant(term_no_coef)
            if term_no_coef and term_no_coef != '0' and not is_constant:
                cleaned_terms.add(term)
        
        return cleaned_terms
    
    def _expand_brackets(self,equation):
        # Expand brackets in the equation like 0.5*(x1+x2) -> 0.5*xx1+0.5*x2
        while '(' in equation:
            start = -1
            for i,char in enumerate(equation):
                if char == '(':
                    start = i
                elif char == ')' and start != -1:
                    bracket_content = equation[start+1:i]

                    coeff_start = start
                    while coeff_start > 0 and (equation[coeff_start-1].isdigit() or equation[coeff_start-1] in '.*/'):
                        coeff_start -= 1
                    
                    sign_start = coeff_start
                    if coeff_start > 0 and equation[coeff_start-1] in '+-':
                        sign_start = coeff_start - 1
                    coefficient_part = equation[sign_start:start]
                    expanded = self._expand_single_bracket(coefficient_part,bracket_content)
                    equation = equation[:sign_start] + expanded + equation[i+1:]
                    break
                    
        return equation
    
    def _expand_single_bracket(self,coefficient,bracket_content):
        bracket_terms = self._split_equation_terms(bracket_content)
        coeff = coefficient.strip()
        if coeff.endswith('*'):
            coeff = coeff[:-1]
        coeff_sign = 1
        coeff_value = ""
        if not coeff or coeff == '+': 
        # The real situation is coefficient value equals to 1 and the real term should looks like -x1
            coeff_value = ""
        elif coeff == '-':
            coeff_sign = -1
            coeff_value = ""
        else:
             if coeff.startswith('-'):
                 coeff_sign = -1
                 coeff_value = coeff[1:]
             elif coeff.startswith('+'):
                 coeff_value = coeff[1:]
             else:
                 coeff_value = coeff

        expanded_terms = []
        for term in bracket_terms:
            term = term.strip()
            term_sign = 1
            if term.startswith('+'):
                term = term[1:]
            elif term.startswith('-'):
                term_sign = -1
                term = term[1:]
            final_sign = coeff_sign * term_sign

            # Build the expanded term
            if coeff_value:
                if final_sign == -1:
                    expanded_term = '-' + coeff_value + '*' + term
                else:
                    expanded_term = coeff_value + '*' + term
            else:
                if final_sign == -1:
                    expanded_term = '-' + term
                else:
                    expanded_term  = term
            expanded_terms.append(expanded_term)
        
        # Join terms with proper signs
        result = ''
        for i,term in enumerate(expanded_terms):
            if i == 0:
                result = term
            else:
                if term.startswith('-'):
                    result += term
                else:
                    result += '+' + term

        return result
    
    def _split_equation_terms(self,equation:str):
        # Split equation into without preserving signs.
        terms = []
        current_term = ""

        i = 0
        while i < len(equation):
            char = equation[i]
            if char == '+' or char == '-':
                if current_term:
                    terms.append(current_term)
                if char == '-':
                    current_term = '-'
                else:
                    current_term = ''
            else:
                current_term += char
            i += 1
        
        if current_term:
            terms.append(current_term)

        return terms
    
    def _remove_coefficients(self, term: str) -> str:
        #Remove symbolic or numeric coefficients from a term (e.g., 'k1*A' or '3*A' → 'A').
        #Keeps only the parts of the term that include state variables (e.g., x1, x2, etc.).
        term = term.strip()
        if not term or term == '0':
            return term
        
        coeff_pattern = r'^[+-]?\d*\.?\d*\*(.+)$'
        match = re.match(coeff_pattern, term)
        if match:
            term = match.group(1)
        
        # If the entire term is just a number, return empty
        if re.match(r'^[+-]?\d+\.?\d*$', term):
            return ''
        
        return term
    
    def _is_pure_constant(self, term: str) -> bool:
        #Return True if the term contains no system state variables (x1, x2, ...).
        if not term or term.strip() == '':
            return True
        
        # If term contains any xi variables, it's a variable term
        if re.match(r'^[+-]?\d+\.?\d*$', term):
            return True
        
        # If no xi variables found, it's a parameter/constant
        if re.search(r'\b[A-Za-z][A-Za-z0-9_]*\b', term):
            return False
        return True