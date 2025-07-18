import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import UnivariateSpline

from dae_finder import PolyFeatureMatrix,sequentialThLin
from Multicollinearity import create_combinations_with_stable_svd,filter_combinations
from math import comb as n_choose_k
from joblib import Parallel, delayed

from Definitions import preprocess_for_stable

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
            
            derivative_col_name = f"d{col}/dt"
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
                #row[f"#{comb}comb"] = results_dict[key]['combinations']
                row[f'# ill-posed {comb}comb'] = results_dict[key]['ill_posed']

            rows.append(row)
        
        summary = pd.DataFrame(rows)
        column_order = ["degree", "# lib terms"]
        #column_order.extend([f'#{c}comb' for c in self.comb_list])
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


class Recover_Model:
    def __init__(self,candidate_lib_full:pd.DataFrame,data_derivatives:pd.DataFrame,threshold):
        # data_derivatives: dataframe generated by Comparison.compute_time_derivatives
        # threshold: coefficient threshold, depends on parameter values
        self.candidate_lib_full = candidate_lib_full
        self.data_derivatives = data_derivatives
        self.threshold = threshold

        self.mapping, self.model_expression = self._recovered_model()
        self.model = self._model_to_dataframe()
    
    def _recovered_equation(self,target_column):
        seq_th_model = sequentialThLin(
            model_id = "RR",
            coef_threshold = self.threshold,
            fit_intercept = False
        )
        seq_th_model.fit(self.candidate_lib_full,self.data_derivatives[target_column])

        final_coef = seq_th_model.coef_history_df.iloc[-1].fillna(0.0)
        discovered_terms = final_coef[final_coef != 0]
        # Build RHS string with correct ± signs
        rhs_terms = []
        for name, coef in discovered_terms.items():
            sign = "+" if coef >= 0 else "-"
            rhs_terms.append(f"{sign} {abs(coef):.4g}*{name}")
        rhs = " ".join(rhs_terms).lstrip("+ ").replace("+ -", "- ")

        equation = f"{target_column} = {rhs}"
        return discovered_terms,equation
    
    def _recovered_model(self):
        col_feature = [col for col in self.data_derivatives.columns if col not in ['t_Exp','t_Sim']]
        self.features = col_feature
        mapping = {}
        for feature in col_feature:
            discovered_terms,equation = self._recovered_equation(target_column=feature)
            mapping[feature] = {
                'discovered_terms': discovered_terms,
                'equation': equation
            }
        model_expression = "\n".join(
            mapping[feature]['equation'] for feature in col_feature
        )
        return mapping, model_expression
    
    def _model_to_dataframe(self):
        rows = []
        for feature in self.features:
            equation = self.mapping[feature]['equation']
            lhs,rhs = equation.split('=',1)
            rows.append({'Variable':lhs.strip(),
                        'Equation':rhs.strip()})
        return pd.DataFrame(rows)



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
        self.all_xi_terms = {self._convert_to_power_notation(t) for t in all_terms}

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
    
    def _convert_to_power_notation(self, term):
        """Convert x1*x1 to x1^2,keep x1*x2 as x1 x2"""
        if not term or '*' not in term:
            return term.replace('*', ' ')
        
        # Split Factor
        factors = term.split('*')
        
        # Count the number of occurrences of each variable
        var_count = {}
        for factor in factors:
            factor = factor.strip()
            if factor:
                # Check if it is already in power form（like x1^2）
                if '^' in factor:
                    base, power = factor.split('^', 1)
                    try:
                        var_count[base] = var_count.get(base, 0) + int(power)
                    except ValueError:
                        var_count[factor] = var_count.get(factor, 0) + 1
                else:
                    var_count[factor] = var_count.get(factor, 0) + 1
        
        # Construct result
        result_parts = []
        for var, count in sorted(var_count.items()):
            if count == 1:
                result_parts.append(var)
            else:
                result_parts.append(f"{var}^{count}")
        
        return ' '.join(result_parts)

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
            # var_name → xi_var
            pattern = r'\b' + re.escape(var_name) + r'\b'
            converted_term = re.sub(pattern, xi_var, converted_term)
            # var_name^n → xi_var^n
            power_pattern = r'\b' + re.escape(var_name) + r'\^(\d+)\b'
            converted_term = re.sub(power_pattern, rf'{xi_var}^\1', converted_term)
        return converted_term
    
    def _extract_terms(self,equation):
        # Extract individual terms from an equation
        #equation = equation.replace('·', '*')   
        #equation = equation.replace('*', ' ')
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
        equation = equation.replace(' ','')
        equation = self._protect_function_parentheses(equation)
        # Expand brackets in the equation like 0.5*(x1+x2) -> 0.5*xx1+0.5*x2
        while '(' in equation:
            start = -1
            for i,char in enumerate(equation):
                if char == '(':
                    start = i
                elif char == ')' and start != -1:
                    bracket_content = equation[start+1:i]

                    coeff_start = self._find_coefficient_start(equation, start)
                    coefficient_part = equation[coeff_start:start]
                    expanded = self._expand_single_bracket(coefficient_part,bracket_content)
                    equation = equation[:coeff_start] + expanded + equation[i+1:]
                    break
        equation = self._restore_function_parentheses(equation)

        return equation
    
    def _protect_function_parentheses(self,equation):
        function_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]*)\)'
        self.function_substitutions = {}
        counter = 0
        while re.search(function_pattern, equation):
            match = re.search(function_pattern, equation)
            if match:
                full_func = match.group(0)
                placeholder = f'__FUNC_{counter}__'
                self.function_substitutions[placeholder] = full_func
                equation = equation.replace(full_func, placeholder, 1)
                counter += 1

        return equation
    
    def _restore_function_parentheses(self, equation):
        if hasattr(self, 'function_substitutions'):
            for placeholder, original in self.function_substitutions.items():
                equation = equation.replace(placeholder, original)

        return equation

    def _find_coefficient_start(self, equation, bracket_start):
            """Coefficient starting position search"""
            i = bracket_start - 1
            
            # Search forward until an operator or the beginning of a string is encountered
            while i >= 0:
                char = equation[i]
                if char in '+-' and i != bracket_start - 1:
                    # If it is a symbol and not immediately adjacent to a bracket, stop
                    if i == 0 or equation[i-1] in '+-*/':
                        # This is a single symbol
                        break
                    else:
                        # This could be part of the expression
                        i -= 1
                elif char in '*/':
                    i -= 1
                elif char.isalnum() or char in '._':
                    i -= 1
                else:
                    break
            
            return i + 1 if i >= 0 else 0

    def _expand_single_bracket(self, coefficient, bracket_content):
        """Single bracket expansion"""
        # Split bracket contents
        bracket_terms = self._split_equation_terms(bracket_content)
        
        # Treatment coefficient
        coeff = coefficient.strip()
        if coeff.endswith('*'):
            coeff = coeff[:-1]
        
        # Processing Symbols
        coeff_sign = 1
        coeff_value = coeff
        
        if coeff.startswith('-'):
            coeff_sign = -1
            coeff_value = coeff[1:] if len(coeff) > 1 else ""
        elif coeff.startswith('+'):
            coeff_value = coeff[1:] if len(coeff) > 1 else ""
        
        # Expand each item
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
            
            # Build expansion items
            if coeff_value:
                if final_sign == -1:
                    expanded_term = '-' + coeff_value + '*' + term
                else:
                    expanded_term = coeff_value + '*' + term
            else:
                if final_sign == -1:
                    expanded_term = '-' + term
                else:
                    expanded_term = term
            
            expanded_terms.append(expanded_term)
        
        # Connection Item
        result = ''
        for i, term in enumerate(expanded_terms):
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
        if not term :
            return ''
        factors = term.split('*')
        variable_factors = []
        all_vars = set(self.reverse_mapping.keys())
        for factor in factors:
            factor = factor.strip()
            contains_var = False
            for var in all_vars:
                if var in factor:
                    contains_var = True
                    break
            if contains_var or '^' in factor or '**' in factor:
                variable_factors.append(factor)
        return '*'.join(variable_factors) if variable_factors else ''
    
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
    
class Terms_Analysis:
    """
    Input:
        original_model: DataFrame (the first column represents features,like dx1/dt)
        discovered_model: model form dae_finder (the first column represents features, like dx1/dt)
        data: candidate function library
    """
    def __init__(self,original_model,discovered_model,variable_mapping,discovered_mapping,data):
        self.original_model = original_model
        self.discovered_model = discovered_model
        self.variable_mapping = variable_mapping
        self.discovered_mapping = discovered_mapping
        self.data = data
        self.original_terms = Terms_Identification(original_model)._parse_model()[1]

        self.wrong_terms, self.missing_terms, self.Mixed_terms = self._wrong_missing_terms()
        self.con = self._SVD_analysis()
    #@staticmethod
    #def _sindy_model_to_df(model):
        # Convert a PySINDy model to a pandas DataFrame
        #equations = model.equations()
        #variables = model.feature_names
        #lhs_vars = [f"d{v}/dt" for v in variables]
        # Fix spacing in RHS by inserting '*' between numbers and variables
        #fixed_equations = []
        #for eq in equations:
            #eq = re.sub(r'(?<=[0-9]) (?=[a-zA-Z])', '*', eq)     # e.g., '0.5 x1' → '0.5*x1'
            #eq = re.sub(r'(?<=[a-zA-Z0-9]) (?=[a-zA-Z])', '*', eq) # e.g., 'x1 x2' → 'x1*x2'
            #fixed_equations.append(eq)

        #dataframe = pd.DataFrame({
            #"Variable": lhs_vars,
            #"Equation": fixed_equations
        #})
        #return dataframe

    def _get_key_from_value(self,dictionary,target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
        
    def _wrong_missing_terms(self):
        wrong_terms = {}
        missing_terms = {}
        mixed_terms = {}

        orig_id = Terms_Identification(self.original_model)
        orig_id._parse_model()
        for lhs in self.original_model.iloc[:,0].tolist():
            lhs_var = lhs.replace('/dt', '').lstrip('d')
            xi_name = self._get_key_from_value(self.variable_mapping,lhs_var)
            lhs_xi  = f"d{xi_name}/dt"
            rhs_str = orig_id.original_equations.get(lhs,"")
            if rhs_str == "":
                continue
            raw_terms = orig_id._extract_terms(rhs_str)
            orig_terms_xi = set()
            for t in raw_terms:
                t_xi = orig_id._convert_single_term_to_xi(t)
                t_clean = orig_id._strip_parameters(t_xi)
                if t_clean:
                    orig_terms_xi.add(orig_id._convert_to_power_notation(t_clean))
        
            discovered_series = self.discovered_mapping[lhs_xi]['discovered_terms']
            discovered_terms = set(discovered_series.index.tolist())
            wrong = discovered_terms - orig_terms_xi
            missing = orig_terms_xi - discovered_terms
            print(f"\n=== {lhs_xi} ===")
            print("ORI :", sorted(orig_terms_xi,key=str))
            print("DIS :", sorted(discovered_terms,key=str))
            print("Missing :", sorted(missing,key=str))
            print("Wrong:", sorted(wrong,key=str))
            wrong_terms[lhs] = {'Terms': wrong}
            missing_terms[lhs] = {'Terms': missing}
            mixed_terms[lhs] = {'Terms': wrong | missing}

        return wrong_terms,missing_terms,mixed_terms
    
    def _SVD_analysis(self):
        con_number = {}
        for feature, mixed in self.Mixed_terms.items():
            data = preprocess_for_stable(self.data[list(mixed['Terms'])],'standarize')[0]
            U, s, Vt = np.linalg.svd(data, full_matrices=False)
            condition_number = s[0] / s[-1]
            con_number[feature] = {'Condition Number': condition_number}
        return con_number