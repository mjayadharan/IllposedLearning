import numpy as np
from numpy.polynomial import Polynomial, chebyshev, legendre
import pandas as pd
from dae_finder import PolyFeatureMatrix
import Comparison
from itertools import product
import sympy as sp

class Monomials:
    def __init__(self,data,degree):
        # data: DataFrame
        # original data including the time column, columns could haven't been renamed
        # degreee: specific single degree
        self.data = data
        self.degree = degree
        # data_states only has state columns and has been renamed
        self.data_states = Comparison.standardize_columns(data)
    
    def _generate_library(self):
        poly_feature_ob = PolyFeatureMatrix(self.degree)
        candidate_lib_full = poly_feature_ob.fit_transform(self.data_states)
        candidate_lib = candidate_lib_full.drop(["1"], axis=1)
        return candidate_lib


def normalization(data_states):
    # data_states: DataFrame, only has state columns
    data_norm = data_states.copy()
    L,U = {},{}
    for col in data_norm.columns:
        Li = data_states[col].min()
        Ui = data_states[col].max()
        L[col] = Li
        U[col] = Ui
        if Li == Ui:
            data_norm[col] = 0
            print("Warning: All values are equal")
        else:
            data_norm[col] = 2*(data_states[col]-Li)/(Ui-Li)-1
    return data_norm, L, U

 
class Orthogonal_Basis:
    def __init__(self,data,degree,method):
        # data: DataFrame
        # original data including the time column, columns could haven't benn renamed
        # degree: specific single degree
        self.data = data
        self.degree = degree
        self.method = method

        data_std,self.time_col = Comparison.standardize_columns(data)
        self.data_states = data_std.copy().drop(columns={self.time_col})
        self.data_norm, self.L, self.U = normalization(self.data_states)
        self.val_states, self.expr_states = self._single_library()
        self.library = self._build_function_library()

    def _single_library(self):
        # Generate Orthogobal Basis for each state
        # Return: val_states: dictionary (dateframe) 
                            # basis for each state
                # expr_states: dictionary (dataframe)
                            # basis expression for each state
        states = self.data_norm.columns.tolist()
        expr_states = {}
        val_states = {}
        for s in states:
            df = pd.DataFrame()
            expr_str = {}
            x_vals = self.data_norm[s]
            for deg in range(self.degree+1):
                if self.method == 'Chebyshev':
                    T = chebyshev.Chebyshev.basis(deg,symbol=s)
                else:
                    T = legendre.Legendre.basis(deg,symbol=s)

                P = T.convert(kind=Polynomial)
                terms = []
                for i,c in enumerate(P.coef):
                    if abs(c) > 1e-12:
                        if i == 0:
                            terms.append(f'{c:.1f}')
                        elif i == 1:
                            terms.append(f'{"+ " if c > 0 else "- "} {abs(c):.1f} {s}')                        
                        else:
                            terms.append(f'{"+ " if c > 0 else "- "} {abs(c):.1f} {s}^{i}')
                expr_str[deg] = terms[0]
                for t in terms[1:]:
                    expr_str[deg] += f'{t}'
                #expr_str[deg] = str(P).split('↦')[-1].strip()
                df[f"T_{deg}({s})"] = T(x_vals)
            expr_states[s] = expr_str
            val_states[s] = df
        return val_states,expr_states
    
    def _build_function_library(self):
        # Generate basis library, column name interms of T_n(x1) and T_m(x2)
        # This function could create cross terms and contain original terms with specific degrees
        states = self.data_states.columns.tolist()
        library_data = []  # Store (total_degree, symbolic_expr, val_series)
        
        # Create symbolic variables
        sym_vars = {s: sp.Symbol(s) for s in states}
        
        for deg_combo in product(range(self.degree+1), repeat=len(states)):
            total_deg = sum(deg_combo)
            if total_deg == 0 or total_deg > self.degree:
                continue
            
            # Build symbolic expression
            symbolic_expr = 1
            #symbolic_expr = sp.Integer(1)
            val_series = pd.Series(1.0, index=self.data_states.index)
            
            for i, deg in enumerate(deg_combo):
                s = states[i]
                val_series = val_series * self.val_states[s].iloc[:, deg]
                
                if deg > 0:
                    if self.method == 'Chebyshev':
                        # Get Chebyshev polynomial and convert to symbolic expression
                        T = chebyshev.Chebyshev.basis(deg, symbol=s)
                    else:
                        T = legendre.Legendre.basis(deg,symbol=s)
                    P = T.convert(kind=Polynomial)
                    
                    # Build symbolic polynomial
                    poly_expr = 0
                    for j, coef in enumerate(P.coef):
                        if abs(coef) > 1e-12:
                            if abs(coef - round(coef)) < 1e-10:
                                coef = int(round(coef))
                            else:
                                coef = round(coef,6)
                            poly_expr += coef * sym_vars[s]**j
                    
                    symbolic_expr *= poly_expr
            
            # Expand and simplify the expression
            symbolic_expr = sp.expand(symbolic_expr)
            # Convert to string with proper formatting
            term_name = self._format_polynomial(symbolic_expr, sym_vars)
            
            library_data.append((total_deg, deg_combo, term_name, val_series))
        
        # Sort by total degree and then by degree combination
        library_data.sort(key=lambda x: (x[0],) + x[1])
        
        # Extract sorted names and values
        library_expr = [item[2] for item in library_data]
        library_vals = [item[3] for item in library_data]
        
        library = pd.DataFrame(dict(zip(library_expr, library_vals)))
        return library

    def _format_polynomial(self,expr,sym_vars):
        # Convert T_n(x1) * T_m(x2) into a polynomial in terms of x1 and x2
        expr = sp.expand(expr)
        # Convert to string with proper ordering
        terms = []
        expr_dict = expr.as_coefficients_dict()
        # Sort terms by total degree (descending) and then by variables
        sorted_terms = sorted(expr_dict.items(),
                              key=lambda x: (sum(x[0].as_powers_dict().values()), str(x[0])),
                              reverse=True)
        for term, coef in sorted_terms:
            if term == 1:
                terms.append(f"{coef}")
            else:
                # Format variables and powers
                var_parts = []
                powers = term.as_powers_dict()
                # Sort variables for consistent ordering
                for var in sorted(powers.keys(),key=str):
                    if var in sym_vars.values():
                        power = powers[var]
                        if power == 1:
                            var_parts.append(str(var))
                        elif power > 1:
                            var_parts.append(f"{var}^{power}")
                var_str = " ".join(var_parts)

                if coef == 1:
                    terms.append(var_str)
                elif coef == -1:
                    terms.append(f"-{var_str}")
                else:
                    terms.append(f"{coef} {var_str}")
        # Join terms with proper signs
        if not terms:
            return "0"
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f"{term}"
            else:
                result += f"+ {term}"
        return result