import numpy as np
import re
from collections import defaultdict
from itertools import product
import sympy as sp
from sympy import symbols, expand, collect
from typing import Dict, List, Tuple, Union

class MultivariatePolynomial:
    def __init__(self,variables: List[str]):
        self.variables = variables
        self.terms = defaultdict(float) # {(power_tuple): coefficient}

    def add_term(self, powers:Tuple[int,...], coefficient:float):
         """Add a term coefficient * x1^p1 * x2^p2 * ..."""
         if len(powers) != len(self.variables):
             raise ValueError("Powers tuple length must match number of variables")
         self.terms[powers] += coefficient

    def __add__(self,other):
        """Polynomial addition"""
        if self.variables != other.variables:
            raise ValueError("Variables must match for addition")
        
        result = MultivariatePolynomial(self.variables)

        for powers, coeff in self.terms.items():
            result.terms[powers] += coeff
        for powers, coeff in other.terms.items():
            result.terms[powers] += coeff

        return result

    def __mul__(self,other):
        """Polynomial multiplication"""
        if isinstance(other,(int,float)):
            result = MultivariatePolynomial(self.variables)
            for powers, coeff in self.terms.items():
                result.terms[powers] = coeff * other
            return result
        
        if self.variables != other.variables:
            raise ValueError("Variables must match for multiplication")
        
        result = MultivariatePolynomial(self.variables)
        for powers1, coeff1 in self.terms.items():
            for powers2, coeff2 in other.terms.items():
                new_powers = tuple(p1+p2 for p1,p2 in zip(powers1,powers2))
                result.terms[new_powers] += coeff1 * coeff2
        return result
    
    def __rmul__(self,other):
        "Right multiplication (scaler * polynomial)"
        return self.__mul__(other)
    
    def _simplify(self,tolerance=1e-27):
        """Simplify the polynomial by removing terms with coefficients close to zero"""
        self.terms = {k: v for k,v in self.terms.items() if abs(v) > tolerance}

    def _to_string(self,tolerance=1e-27,precision=6):
        """Convert to a readable string"""
        if not self.terms:
            return "0"
        
        terms_str = []
        for powers, coeff in sorted(self.terms.items()):
            if abs(coeff) < tolerance:
                continue

            # Construct coefficient
            if abs(coeff-1) < tolerance and sum(powers) > 0:
                coeff_str = ""
            elif abs(coeff+1) < tolerance and sum(powers) > 0:
                coeff_str = "-"
            else:
                coeff_str = f"{coeff:.{precision}f}"

            # Construct variable
            var_parts = []
            for i, power in enumerate(powers):
                if power > 0:
                    var_name = self.variables[i]
                    if power == 1:
                        var_parts.append(var_name)
                    else:
                        var_parts.append(f"{var_name}^{power}")
            if var_parts:
                if coeff_str == "":
                    term_str = " ".join(var_parts)
                elif coeff_str == "-":
                    term_str = "-" + " ".join(var_parts)
                else:
                    term_str = f"{coeff_str} " + " ".join(var_parts)
            else:
                term_str = coeff_str

            terms_str.append(term_str)
        
        if not terms_str:
            return "0"
        
        result = terms_str[0]
        for term in terms_str[1:]:
            if term.startswith("-"):
                result += f" {term}"
            else:
                result += f" + {term}"

        return result
    

class OrthogonalToPolynomialConverter:
    def __init__(self, max_degree, basis: str = 'Legendre'):
        self.max_degree = max_degree
        self.basis = basis.capitalize()
        if self.basis not in ('Chebyshev', 'Legendre'):
            raise ValueError(f"Unsupported basis '{basis}'. Use 'Chebyshev' or 'Legendre'.")
        self._chebyshev_cache = {}
        self._legendre_cache = {}
        if self.basis == 'Chebyshev':
            self._precompute_chebyshev_polynomials()
        else:
            self._precompute_legendre_polynomials()

    def _precompute_chebyshev_polynomials(self):
        # T_0(x) = 1, the first 0 represents 0 degree Chebyshev basis, the second zero represents x^0
        # 1.0 represents the coefficient of x^0
        self._chebyshev_cache[0] = {0: 1.0}
        # T_1(x) = x
        self._chebyshev_cache[1] = {1: 1.0}
        # Derive T_n(x) with recurrence definition: T_n+1(x) = 2x T_n(x) - T_n-1(x)
        for n in range(2,self.max_degree+1):
            self._chebyshev_cache[n] = defaultdict(float)
            for power, coeff in self._chebyshev_cache[n-1].items():
                self._chebyshev_cache[n][power+1] += 2 * coeff
            for power, coeff in self._chebyshev_cache[n-2].items():
                self._chebyshev_cache[n][power] -= coeff
            # Clean up coefficients close to zero
            self._chebyshev_cache[n] = {k: v for k,v in self._chebyshev_cache[n].items()
                                        if abs(v) > 1e-27}

    def _precompute_legendre_polynomials(self):
        # P_0(x) = 1
        self._legendre_cache[0] = {0: 1.0}
        # P_1(x) = x
        self._legendre_cache[1] = {1: 1.0}
        # (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
        for n in range(1, self.max_degree):
            next_poly = defaultdict(float)
            # (2n+1) x P_n(x)
            for power, coeff in self._legendre_cache[n].items():
                next_poly[power+1] += (2*n + 1) * coeff
            # - n P_{n-1}(x)
            for power, coeff in self._legendre_cache[n-1].items():
                next_poly[power] -= n * coeff
            # divide by (n+1)
            for power in list(next_poly.keys()):
                next_poly[power] /= (n + 1)
            # cleanup
            self._legendre_cache[n+1] = {k: v for k, v in next_poly.items() if abs(v) > 1e-27}
    
    def _get_univariate_poly(self, degree, variable: str) -> MultivariatePolynomial:
        """Get the polynomial corresponding to T_n(variable) or P_n(variable) depending on basis."""
        if degree > self.max_degree:
            raise ValueError(f"Degree {degree} exceeds maximum cached degree {self.max_degree}")
        poly = MultivariatePolynomial([variable])
        if self.basis == 'Chebyshev':
            cache = self._chebyshev_cache
        else:
            cache = self._legendre_cache
        for power, coeff in cache[degree].items():
            poly.add_term((power,), coeff)
        return poly
    
    def _parse_sindy_equation(self,equation_str:str) -> Tuple[str, str]:
        """Parse the SINDy equation string and return (left side, right side)"""
        parts = equation_str.split(' = ')
        if len(parts) != 2:
            raise ValueError("Invalid equation format")
        rhs = parts[0].strip()
        lhs = parts[1].strip()
        return rhs, lhs
    
    def _parse_orthogonal_term(self, term_str) -> Tuple[float, List[Tuple[str, int]]]:
        """
        Parse a term like `alpha T_1(x3) T_2(x1)` (Chebyshev) or `alpha P_1(x3) P_2(x1)` (Legendre).
        Returns (coefficient, [(variable, degree), ...])
        """
        if isinstance(term_str, list):
            term_str = ' '.join(str(item) for item in term_str)
        if not isinstance(term_str, str):
            term_str = str(term_str)
        term_str = term_str.strip()

        symbol = 'T_' if self.basis == 'Chebyshev' else 'P_'
        t_match = re.search(symbol, term_str)
        if t_match:
            coeff_str = term_str[:t_match.start()].strip()
            basis_part = term_str[t_match.start():].strip()
        else:
            # No basis functions present -> constant term
            try:
                coeff = float(term_str)
                return coeff, []
            except ValueError:
                raise ValueError(f"Cannot parse term: {term_str}")

        # coefficient
        if coeff_str in ('', '+'):
            coefficient = 1.0
        elif coeff_str == '-':
            coefficient = -1.0
        else:
            try:
                coefficient = float(coeff_str)
            except ValueError:
                raise ValueError(f"Invalid coefficient: {coeff_str}")

        # basis function occurrences
        pattern = r'T_(\d+)\(([^)]+)\)' if self.basis == 'Chebyshev' else r'P_(\d+)\(([^)]+)\)'
        matches = re.findall(pattern, basis_part)
        cheb_terms: List[Tuple[str, int]] = []
        for degree_str, variable in matches:
            cheb_terms.append((variable, int(degree_str)))
        return coefficient, cheb_terms
    
    def _convert_basis_term_to_polynomial(self,
                                          coefficient: float,
                                          basis_terms: List[Tuple[str,int]],
                                          all_variables: List[str]) -> MultivariatePolynomial:
        result = MultivariatePolynomial(all_variables)
        result.add_term(tuple(0 for _ in all_variables), 1.0)
        for variable, degree in basis_terms:
            if variable not in all_variables:
                raise ValueError(f"Variable {variable} not in variable list")
            single_var_poly = self._get_univariate_poly(degree, variable)
            multi_var_poly = MultivariatePolynomial(all_variables)
            var_index = all_variables.index(variable)
            for (power,), coeff in single_var_poly.terms.items():
                powers = [0] * len(all_variables)
                powers[var_index] = power
                multi_var_poly.add_term(tuple(powers), coeff)
            result = result * multi_var_poly
        result = result * coefficient
        return result
    
    def convert_sindy_model(self,equations:Union[List[str], List[str]],
                                                 variable_names: List[str] = None) -> Dict[str, MultivariatePolynomial]:
        """
        Convert the SINDy model
        equations: 1. Full equation list including both rhs and lhs
                    2. Equation list only has rhs
        variable names: If None and the input is the right side of the equation, it is automatically inferred

        Supports Chebyshev or Legendre equations, controlled by basis.

        Return:
        Dict[Variable name, Polynomial]
        """
        has_equations_format = any('=' in eq for eq in equations)

        if has_equations_format:
            # Format 1
            return self._convert_full_equations(equations)
        else:
            # Format 2
            return self._convert_rhs_equations(equations, variable_names)
        
    def _convert_full_equations(self,equations: List[str]) -> Dict[str, MultivariatePolynomial]:
        """
        Convert the full SINDy model
        equations: Equation list of the SINDy model

        Returns:
        Dict[Variable name, Polynomial]
        """
        results = {}
        # Extract all variables
        all_variables = set()
        for eq in equations:
            # From lhs
            lhs, rhs = self._parse_sindy_equation(eq)
            var_match = re.search(r'\((.*?)\)', lhs)
            if var_match:
                derivative_var = var_match.group(1)
                all_variables.add(derivative_var)
            # From rhs
            pattern = r'T_\d+\(([^)]+)\)' if self.basis == 'Chebyshev' else r'P_\d+\(([^)]+)\)'
            variables = re.findall(pattern, rhs)
            all_variables.update(variables)

        all_variables = sorted(list(all_variables))

        # Handle every equation
        for eq in equations:
            lhs, rhs = self._parse_sindy_equation(eq)
            var_match = re.search(r'\((.*?)\)', lhs)
            if not var_match:
                continue
            derivative_var = var_match.group(1)

            result_poly = self._convert_rhs(rhs, all_variables)
            results[derivative_var] = result_poly

        return results
    
    def _convert_rhs_equations(self,equations_list: List[str],
                               variable_names: List[str] = None) -> Dict[str, MultivariatePolynomial]:
        if variable_names is None:
            all_variables = set()
            pattern = r'T_\d+\(([^)]+)\)' if self.basis == 'Chebyshev' else r'P_\d+\(([^)]+)\)'
            for eq in equations_list:
                variables = re.findall(pattern, eq)
                all_variables.update(variables)
            variable_names = sorted(list(all_variables))

        if len(equations_list) != len(variable_names):
            print(f"Warning: The number of equtions({len(equations_list)})doesn't match the length the number of variables({len(variable_names)})")
            n_equations = min(len(equations_list), len(variable_names))
        else:
            n_equations = len(equations_list)

        results = {}
        for i in range(n_equations):
            rhs = equations_list[i]
            lhs = variable_names[i]
            result_poly = self._convert_rhs(rhs,variable_names)
            results[lhs] = result_poly

        return results
    
    def _convert_rhs(self,equation_rhs:str,all_variables:List[str]) -> MultivariatePolynomial:
        """Convert rhs of a single equation"""
        result_poly = MultivariatePolynomial(all_variables)
        terms = self._split_equation_terms(equation_rhs)

        for term in terms:
            if not term:
                continue
            try:
                coefficient, basis_terms = self._parse_orthogonal_term(term)
                term_poly = self._convert_basis_term_to_polynomial(
                    coefficient, basis_terms, all_variables
                )
                result_poly = result_poly + term_poly
            except Exception as e:
                print(f"Warning: Cannot parse '{term}':{e}")
                continue
        result_poly._simplify()
        return result_poly
    
    def _split_equation_terms(self,equation_str:str) -> List[str]:
        """Split terms in an equation"""

        # Process signs
        equation_str = equation_str.replace(' + ', ' +').replace(' - ', ' -')
        if not equation_str.startswith(('+', '-')):
            equation_str = '+' + equation_str
        
        # Split terms
        #terms = re.split(r'(?=\s*[+-])', equation_str)
        terms = re.findall(r'[+-]?\s*[^+-]+', equation_str)
        terms = [term.strip() for term in terms if term.strip()]
        string_terms = []
        for term in terms:
            if isinstance(term, str):
                string_terms.append(term.strip())
            elif isinstance(term, list):
                # If accidentally get a list, concatenate it into a string
                string_terms.append(' '.join(str(item) for item in term))
            else:
                print(f"Warning: Skip non-string items {term} (type: {type(term)})")
        # Remove terms that are just standalone '+' or '-'
        #string_terms = [term for term in string_terms if term.strip() and not re.fullmatch(r'[+-]', term.strip())]
        return string_terms
    
    def _print_model(self,polynomial_model: Dict[str, MultivariatePolynomial]):
        for var, poly in polynomial_model.items():
            print(f"d{var}/dt = {poly._to_string()}")


class EquationDenormalizer:
    def __init__(self, L: dict, U: dict):
        """
        Initialize with min (L) and max (U) values used in normalization.
        Args:
            L (dict): Dictionary of min values for each variable
            U (dict): Dictionary of max values for each variable
        """
        self.L = L
        self.U = U
        self.variables = list(L.keys())

        self.orig_vars = {var: sp.Symbol(var) for var in self.variables}
        # x_original = (x_norm + 1) * (U - L) / 2 + L
        self.norm_to_orig_mapping = {}
        for var in self.variables:
            if self.U[var] == self.L[var]:
                self.norm_to_orig_mapping[var] = 0
            else:
                self.norm_to_orig_mapping[var] = 2*(self.orig_vars[var] - self.L[var])/(self.U[var] - self.L[var]) - 1

    def _parse_equation(self,equation:str):
        """Extract lhs and rhs of an equation"""
        parts = equation.split('=')
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        lhs, rhs = lhs.strip(), rhs.strip()

        processed_rhs = self._preprocess_expression(rhs)
        rhs = sp.sympify(processed_rhs)

        return lhs, rhs
    
    def _preprocess_expression(self,expr):
        """Preprocess expression strings to conform to the Sympy format"""
        # replace ^ with **
        expr_str = expr.replace('^',"**")
        #  Handle multiplication signs between two variables, eg. x3 x4 -> x3*x4
        pattern = r'(\w+)\s+(\w+)'
        while re.search(pattern, expr_str):
            expr_str = re.sub(pattern, r'\1*\2', expr_str)
        # Handle multiplication signs between coefficients and variables, eg. 2 x -> 2*x
        pattern = r'(\d+\.?\d*)\s+([a-zA-Z])'
        expr_str = re.sub(pattern, r'\1*\2', expr_str)
        return expr_str
    
    def denormalize_expression(self,expr: sp.Expr) -> sp.Expr:
        """Convert the normalized expression to an expression of the original variables"""
        substitutions = {}
        for var in self.variables:
            if var in str(expr):
                if self.U[var] != self.L[var]:
                    norm_in_terms_of_orig = 2*(self.orig_vars[var] - self.L[var]) / (self.U[var] - self.L[var]) - 1
                    #norm_in_terms_of_orig = (self.orig_vars[var] + 1)*(self.U[var] - self.L[var])/2 + self.L[var]
                    substitutions[sp.Symbol(var)] = norm_in_terms_of_orig
                else:
                    substitutions[sp.Symbol(var)] = 0
        # Perform Replacement
        denorm_expr = expr.subs(substitutions)
        denorm_expr = sp.expand(denorm_expr)
        denorm_expr = sp.simplify(denorm_expr)

        return denorm_expr
    
    def denormalize_equations(self,equations:List[str]):
        """Convert a list of normalized equations to a list of equations in the original variables"""
        denorm_equations = []

        for eq in equations:
            lhs,rhs = self._parse_equation(eq)
            denorm_right_expr = self.denormalize_expression(rhs)
            denorm_eq_str = f"{lhs} = {self._to_custom_string(denorm_right_expr)}"
            denorm_equations.append(denorm_eq_str)
        
        return denorm_equations

    def _to_custom_string(self, expr: sp.Expr) -> str:
        """Convert sympy expression to a custom string with ^ and space-multiplied variables"""
        expr_str = str(expr)
        expr_str = expr_str.replace("**", "^")
        # Replace * between variables with space, keep * between numbers and variables
        expr_str = re.sub(r'(?<=[a-zA-Z0-9)])\*(?=[a-zA-Z(])', ' ', expr_str)
        return expr_str
    
    def _print_model_with_original_variables(self,denorm_equations):
        for i, eq in enumerate(denorm_equations):
            print(f"  {eq}")