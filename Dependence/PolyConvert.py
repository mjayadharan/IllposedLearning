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
                # Ensure power1 and powers2 have the same length
                if len(powers1) != len(powers2):
                    raise ValueError(f"Power tuples length mismatch: {powers1} vs {powers2}")
                new_powers = tuple(p1 + p2 for p1, p2 in zip(powers1, powers2))
                result.terms[new_powers] += coeff1 * coeff2
        return result
    
    def __rmul__(self,other):
        "Right multiplication (scaler * polynomial)"
        return self.__mul__(other)
    
    def _simplify(self,tolerance=1e-12):
        """Simplify the polynomial by removing terms with coefficients close to zero"""
        self.terms = {k: v for k,v in self.terms.items() if abs(v) > tolerance}

    def _to_string(self,tolerance=1e-12,precision=6):
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
                coeff_str = f"{coeff:.{precision}g}"

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
    def __init__(self, max_degree, basis: str = 'Legendre', simplify_tol: float = 1e-12, print_tol: float = 1e-12, print_precision: int = 6):
        self.max_degree = max_degree
        self.basis = basis.capitalize()
        if self.basis not in ('Chebyshev', 'Legendre', 'Hermite', 'Laguerre','Legendreoriginalrange'):
            raise ValueError(
                f"Unsupported basis '{basis}'. Use 'Chebyshev', 'Legendre', 'Hermite', or 'Laguerre'."
            )
        # Map special aliases to their effective polynomial family
        if self.basis == 'Legendreoriginalrange':
            self._effective_basis = 'Legendre'
            self._use_original_range = True
        else:
            self._effective_basis = self.basis
            self._use_original_range = False
        self._chebyshev_cache = {}
        self._legendre_cache = {}
        self._hermite_cache = {}
        self._laguerre_cache = {}
        self.simplify_tol = float(simplify_tol)
        self.print_tol = float(print_tol)
        self.print_precision = int(print_precision)
        if self._effective_basis == 'Chebyshev':
            self._precompute_chebyshev_polynomials()
        elif self._effective_basis == 'Legendre':
            self._precompute_legendre_polynomials()
        elif self._effective_basis == 'Hermite':
            self._precompute_hermite_polynomials()
        else:  # Laguerre
            self._precompute_laguerre_polynomials()

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
                                        if abs(v) > 1e-12}

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
            self._legendre_cache[n+1] = {k: v for k, v in next_poly.items() if abs(v) > 1e-12}

    def _precompute_hermite_polynomials(self):
        """Physicists' Hermite polynomials H_n(x):
        H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}."""
        self._hermite_cache[0] = {0: 1.0}
        if self.max_degree >= 1:
            self._hermite_cache[1] = {1: 2.0}
        for n in range(1, self.max_degree):
            next_poly = defaultdict(float)
            # 2x * H_n
            for power, coeff in self._hermite_cache[n].items():
                next_poly[power + 1] += 2.0 * coeff
            # - 2n * H_{n-1}
            for power, coeff in self._hermite_cache[n - 1].items():
                next_poly[power] -= 2.0 * n * coeff
            # cleanup
            self._hermite_cache[n + 1] = {k: v for k, v in next_poly.items() if abs(v) > 1e-12}

    def _precompute_laguerre_polynomials(self):
        """(Alpha=0) Laguerre polynomials L_n(x):
        L_0=1, L_1=1-x, (n+1)L_{n+1}=(2n+1-x)L_n - n L_{n-1}."""
        self._laguerre_cache[0] = {0: 1.0}
        if self.max_degree >= 1:
            self._laguerre_cache[1] = {0: 1.0, 1: -1.0}
        for n in range(1, self.max_degree):
            next_poly = defaultdict(float)
            # (2n+1) * L_n contribution (same powers)
            for power, coeff in self._laguerre_cache[n].items():
                next_poly[power] += (2 * n + 1) * coeff
            # - x * L_n contribution -> power+1 with negative sign
            for power, coeff in self._laguerre_cache[n].items():
                next_poly[power + 1] -= coeff
            # - n * L_{n-1}
            for power, coeff in self._laguerre_cache[n - 1].items():
                next_poly[power] -= n * coeff
            # divide by (n+1)
            inv = 1.0 / (n + 1)
            for power in list(next_poly.keys()):
                next_poly[power] *= inv
            # cleanup
            self._laguerre_cache[n + 1] = {k: v for k, v in next_poly.items() if abs(v) > 1e-12}
    
    def _basis_symbol(self) -> str:
        return {
            'Chebyshev': 'T_',
            'Legendre':  'P_',
            'Hermite':   'H_',
            'Laguerre':  'L_',
        }[self._effective_basis]

    def _basis_regex(self) -> str:
        sym = self._basis_symbol()
        return rf"{sym}\\d+\\(([^)]+)\\)"

    def _get_univariate_poly(self, degree, variable: str,
                         lower_bound: float=None, upper_bound: float=None) -> MultivariatePolynomial:
        """Get the polynomial corresponding to T_n(variable), P_n(variable), H_n(variable), or L_n(variable) depending on basis."""
        if degree > self.max_degree:
            raise ValueError(f"Degree {degree} exceeds maximum cached degree {self.max_degree}")
        poly = MultivariatePolynomial([variable])
        if self._effective_basis == 'Chebyshev':
            cache = self._chebyshev_cache
        elif self._effective_basis == 'Legendre':
            cache = self._legendre_cache
        elif self._effective_basis == 'Hermite':
            cache = self._hermite_cache
        else:  # Laguerre
            cache = self._laguerre_cache
        
        # If using original range, apply linear transformation
        if self._use_original_range and lower_bound is not None and upper_bound is not None:
            t_var = sp.Symbol('t')
            x_var = sp.Symbol(variable)
            std_poly = sum(coeff * t_var**power for power, coeff in cache[degree].items())
            x_in_t = (2 * x_var - (lower_bound + upper_bound)) / (upper_bound - lower_bound)
            expanded_poly = sp.expand(std_poly.subs(t_var, x_in_t))
            
            # 修正：正确提取系数的方法
            poly_sp = expanded_poly.as_poly(x_var)
            if poly_sp is not None:
                # 多项式形式
                all_coeffs = poly_sp.all_coeffs()  # 从最高次到常数项的系数列表
                degree_poly = poly_sp.degree()
                for i, coeff in enumerate(all_coeffs):
                    power = degree_poly - i
                    poly.add_term((power,), float(coeff))
            else:
                # Constant
                poly.add_term((0,), float(expanded_poly))
        else:
            for power, coeff in cache[degree].items():
                poly.add_term((power,), coeff)

        return poly
    
    def _parse_sindy_equation(self,equation_str:str) -> Tuple[str, str]:
        """Parse the SINDy equation string and return (left side, right side).
        Robust to spaces around '=' and to additional '=' in RHS (joins back)."""
        if not isinstance(equation_str, str):
            equation_str = str(equation_str)
        parts = equation_str.split('=')
        if len(parts) < 2:
            raise ValueError(f"Invalid equation format: {equation_str}")
        lhs = parts[0].strip()
        rhs = '='.join(parts[1:]).strip()
        return lhs, rhs

    def _extract_derivative_var_from_lhs(self, lhs: str):
        """Try to extract variable name from LHS like 'dx1/dt' or 'd(x1)/dt'.
        Fallback to a '(var)' pattern if present. Return None if not found."""
        lhs = lhs.strip()
        # dvar/dt
        m = re.fullmatch(r"d\s*([^/()\s]+)\s*/\s*dt", lhs)
        if m:
            return m.group(1)
        # d(var)/dt
        m = re.fullmatch(r"d\s*\(([^()]+)\)\s*/\s*dt", lhs)
        if m:
            return m.group(1)
        # Legacy: any '(var)'
        m = re.search(r"\(([^)]+)\)", lhs)
        if m:
            return m.group(1)
        return None
    
    def _parse_orthogonal_term(self, term_str) -> Tuple[float, List[Tuple[str, int]]]:
        """
        Parse a term like `alpha T_1(x3) T_2(x1)` (Chebyshev), `alpha P_1(x3)`, `H_2(x1)`, or `L_2(x1)` depending on basis.
        Returns (coefficient, [(variable, degree), ...])
        """
        if isinstance(term_str, list):
            term_str = ' '.join(str(item) for item in term_str)
        if not isinstance(term_str, str):
            term_str = str(term_str)
        term_str = term_str.strip()

        symbol = self._basis_symbol()
        t_match = re.search(symbol, term_str)
        
        if t_match:
            coeff_str = term_str[:t_match.start()].strip()
            basis_part = term_str[t_match.start():].strip()
        else:
            # No basis functions present -> constant term
            # FIX: Handle formats like "-0.039 1" by removing trailing "1" or "* 1"
            term_cleaned = re.sub(r'\s*\*?\s*1\s*$', '', term_str)
            
            try:
                # Try to parse the cleaned version first
                if term_cleaned and term_cleaned != term_str:
                    coeff = float(term_cleaned)
                else:
                    coeff = float(term_str)
                return coeff, []
            except ValueError:
                raise ValueError(f"Cannot parse term: {term_str}")

        # coefficient parsing
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
        pattern = rf"{self._basis_symbol()}(\d+)\(([^)]+)\)"
        matches = re.findall(pattern, basis_part)
        basis_terms: List[Tuple[str, int]] = []
        for degree_str, variable in matches:
            basis_terms.append((variable, int(degree_str)))
        
        return coefficient, basis_terms
    #def _parse_orthogonal_term(self, term_str) -> Tuple[float, List[Tuple[str, int]]]:
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
                                      all_variables: List[str],
                                      bounds: Dict[str, Tuple[float, float]] = None) -> MultivariatePolynomial:
        result = MultivariatePolynomial(all_variables)
        result.add_term(tuple(0 for _ in all_variables), 1.0)
        for variable, degree in basis_terms:
            if variable not in all_variables:
                raise ValueError(f"Variable {variable} not in variable list")
            # Get the data range for this variable
            lower_bound, upper_bound = bounds.get(variable, (None, None)) if bounds else (None, None)
            single_var_poly = self._get_univariate_poly(degree, variable, lower_bound, upper_bound)
            # Create a multivariate polynomial whose variables are consistent with all_variables
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
                                                 variable_names: List[str] = None,
                                                 bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, MultivariatePolynomial]:
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
            return self._convert_full_equations(equations, bounds)
        else:
            # Format 2
            return self._convert_rhs_equations(equations, variable_names, bounds)

    def _convert_full_equations(self,equations: List[str], bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, MultivariatePolynomial]:
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
            lhs, rhs = self._parse_sindy_equation(eq)
            dv = self._extract_derivative_var_from_lhs(lhs)
            if dv:
                all_variables.add(dv)
            # From rhs
            variables = re.findall(self._basis_regex(), rhs)
            all_variables.update(variables)

        all_variables = sorted(list(all_variables))

        # Handle every equation
        for eq in equations:
            lhs, rhs = self._parse_sindy_equation(eq)
            derivative_var = self._extract_derivative_var_from_lhs(lhs)
            if derivative_var is None:
                continue
            result_poly = self._convert_rhs(rhs, all_variables)
            results[derivative_var] = result_poly

        return results
    
    def _convert_rhs_equations(self,equations_list: List[str],
                               variable_names: List[str] = None,
                               bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, MultivariatePolynomial]:
        if variable_names is None:
            all_variables = set()
            for eq in equations_list:
                variables = re.findall(self._basis_regex(), eq)
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
            result_poly = self._convert_rhs(rhs,variable_names, bounds)
            results[lhs] = result_poly

        return results

    def _convert_rhs(self,equation_rhs:str,all_variables:List[str], bounds: Dict[str, Tuple[float, float]] = None) -> MultivariatePolynomial:
        """Convert rhs of a single equation"""
        result_poly = MultivariatePolynomial(all_variables)
        terms = self._split_equation_terms(equation_rhs)

        for term in terms:
            if not term:
                continue
            try:
                coefficient, basis_terms = self._parse_orthogonal_term(term)
                term_poly = self._convert_basis_term_to_polynomial(
                    coefficient, basis_terms, all_variables, bounds
                )
                result_poly = result_poly + term_poly
            except Exception as e:
                print(f"Warning: Cannot parse '{term}':{e}")
                continue
        result_poly._simplify(getattr(self, 'simplify_tol', 1e-12))
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
            print(f"d{var}/dt = {poly._to_string(tolerance=getattr(self,'print_tol',1e-12), precision=getattr(self,'print_precision',6))}")


class EquationDenormalizer:
    def __init__(self, L: dict, U: dict,basis):
        """
        Initialize with min (L) and max (U) values used in normalization.
        Args:
            L (dict): Dictionary of min values for each variable
            U (dict): Dictionary of max values for each variable
        """
        self.L = L
        self.U = U
        self.basis = basis
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
                    if self.basis in ('Laguerre','Hermite'):
                        norm_in_terms_of_orig = self.orig_vars[var]
                    else:
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


class MatrixEquationDenormalizer:
    """
    Denormalize equations using matrix formulation.
    
    Normalization relation: u = A*x + b
    where:
        - u: normalized variables
        - x: original variables  
        - A: diagonal matrix with a_i = 2/(U_i - L_i)
        - b: vector with b_i = -(U_i + L_i)/(U_i - L_i)
        - U_i, L_i: max and min values of x_i
    
    For equations: du/dt = f(u)
    Convert to: dx/dt = A^(-1) * f(A*x + b)
    """
    
    def __init__(self, L: dict, U: dict, basis: str = 'Legendre'):
        """
        Initialize with min (L) and max (U) values used in normalization.
        
        Args:
            L (dict): Dictionary of min values for each variable
            U (dict): Dictionary of max values for each variable
            basis (str): Basis type (for compatibility, not used in matrix method)
        """
        self.L = L
        self.U = U
        self.basis = basis
        self.variables = list(L.keys())
        self.n_vars = len(self.variables)
        
        # Create symbolic variables
        self.x_syms = {var: sp.Symbol(var) for var in self.variables}
        self.u_syms = {var: sp.Symbol(f'u_{var}') for var in self.variables}
        
        # Build transformation matrices and vectors
        self._build_transformation_matrices()
        
    def _build_transformation_matrices(self):
        """
        Build A matrix and b vector for transformation u = A*x + b
        """
        # A is diagonal matrix with a_i = 2/(U_i - L_i)
        a_diag = []
        b_vec = []
        
        for var in self.variables:
            if self.U[var] == self.L[var]:
                # Handle degenerate case
                a_i = 1.0
                b_i = 0.0
            else:
                a_i = 2.0 / (self.U[var] - self.L[var])
                b_i = -(self.U[var] + self.L[var]) / (self.U[var] - self.L[var])
            
            a_diag.append(a_i)
            b_vec.append(b_i)
        
        # Store as sympy matrices for symbolic computation
        self.A_sym = sp.diag(*a_diag)
        self.b_sym = sp.Matrix(b_vec)
        self.A_inv_sym = sp.diag(*[1/a if a != 0 else 0 for a in a_diag])
        
        # Store as numpy arrays for numerical computation
        self.A_np = np.diag(a_diag)
        self.b_np = np.array(b_vec)
        self.A_inv_np = np.diag([1/a if a != 0 else 0 for a in a_diag])
        
        print("Transformation matrices built:")
        print(f"A (diagonal): {a_diag}")
        print(f"b (vector): {b_vec}")
        print(f"A^(-1) (diagonal): {[1/a if a != 0 else 0 for a in a_diag]}")
        
    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess expression string to conform to Sympy format"""
        expr_str = expr.replace('^', '**')
        # Handle multiplication between variables: x3 x4 -> x3*x4
        pattern = r'(\w+)\s+(\w+)'
        while re.search(pattern, expr_str):
            expr_str = re.sub(pattern, r'\1*\2', expr_str)
        # Handle multiplication between coefficients and variables: 2 x -> 2*x
        pattern = r'(\d+\.?\d*)\s+([a-zA-Z])'
        expr_str = re.sub(pattern, r'\1*\2', expr_str)
        return expr_str
    
    def _parse_equation(self, equation: str):
        """Extract lhs and rhs of an equation"""
        parts = equation.split('=')
        lhs = parts[0].strip()
        rhs = parts[1].strip() if len(parts) > 1 else ''
        
        processed_rhs = self._preprocess_expression(rhs)
        rhs_expr = sp.sympify(processed_rhs)
        
        return lhs, rhs_expr
    
    def denormalize_single_equation(self, equation: str, var_index: int) -> sp.Expr:
        """
        Denormalize a single equation using matrix formulation.
        
        Args:
            equation (str): Normalized equation string "du_i/dt = f(u)"
            var_index (int): Index of the variable (0 to n-1)
            
        Returns:
            sp.Expr: Denormalized expression in terms of original variables
        """
        lhs, rhs_expr = self._parse_equation(equation)
        
        # Substitute u variables with A*x + b
        # For each u_j, replace with a_j * x_j + b_j
        substitutions = {}
        for j, var in enumerate(self.variables):
            u_sym = self.u_syms.get(var, sp.Symbol(var))
            x_sym = self.x_syms[var]
            
            # u_j = a_j * x_j + b_j
            a_j = float(self.A_sym[j, j])
            b_j = float(self.b_sym[j])
            
            substitutions[u_sym] = a_j * x_sym + b_j
            # Also handle case where variable name is used directly
            substitutions[sp.Symbol(var)] = a_j * x_sym + b_j
        
        # Apply substitution
        rhs_in_x = rhs_expr.subs(substitutions)
        
        # Multiply by 1/a_i (since dx_i/dt = (1/a_i) * du_i/dt)
        a_i = float(self.A_sym[var_index, var_index])
        
        if a_i != 0:
            denorm_expr = rhs_in_x / a_i
        else:
            denorm_expr = rhs_in_x
        
        # Expand and simplify
        denorm_expr = sp.expand(denorm_expr)
        denorm_expr = sp.simplify(denorm_expr)
        
        return denorm_expr
    
    def denormalize_equations(self, equations: List[str]) -> List[str]:
        """
        Denormalize a list of equations using matrix method.
        
        Args:
            equations (List[str]): List of normalized equations
            
        Returns:
            List[str]: List of denormalized equations
        """
        denorm_equations = []
        
        for i, eq in enumerate(equations):
            # Extract variable name from equation
            lhs, _ = self._parse_equation(eq)
            
            # Determine which variable this equation corresponds to
            # Try to extract from "d{var}/dt" pattern
            match = re.match(r'd\s*(\w+)\s*/\s*dt', lhs)
            if match:
                var_name = match.group(1)
                # Remove 'u_' prefix if present
                if var_name.startswith('u_'):
                    var_name = var_name[2:]
                
                # Find index of this variable
                if var_name in self.variables:
                    var_index = self.variables.index(var_name)
                else:
                    var_index = i  # Fallback to position
            else:
                var_index = i
            
            # Denormalize this equation
            denorm_expr = self.denormalize_single_equation(eq, var_index)
            
            # Format output
            var_name = self.variables[var_index]
            denorm_eq_str = f"d{var_name}/dt = {self._to_custom_string(denorm_expr)}"
            denorm_equations.append(denorm_eq_str)
        
        return denorm_equations
    
    def denormalize_from_dataframe(self, df, var_col='Variable', eq_col='Equation'):
        """
        Denormalize equations from a pandas DataFrame.
        
        Args:
            df: DataFrame with equations
            var_col (str): Column name containing variable names
            eq_col (str): Column name containing equations
            
        Returns:
            List[str]: List of denormalized equations
        """
        equations = []
        
        for _, row in df.iterrows():
            var_name = str(row[var_col]).strip()
            rhs = str(row[eq_col]).strip()
            
            # Extract actual variable name from "dX/dt" if present
            match = re.match(r'd\s*(\w+)\s*/\s*dt', var_name)
            if match:
                var_name = match.group(1)
            
            # Build equation string
            eq = f"d{var_name}/dt = {rhs}"
            equations.append(eq)
        
        return self.denormalize_equations(equations)
    
    def _to_custom_string(self, expr: sp.Expr) -> str:
        """Convert sympy expression to custom string with ^ and space-multiplied variables"""
        expr_str = str(expr)
        expr_str = expr_str.replace("**", "^")
        # Replace * between variables with space, keep * between numbers and variables
        expr_str = re.sub(r'(?<=[a-zA-Z0-9)])\*(?=[a-zA-Z(])', ' ', expr_str)
        return expr_str
    
    def _print_model_with_original_variables(self, denorm_equations: List[str]):
        """Print denormalized equations"""
        print("\nDenormalized equations (in original variables):")
        print("=" * 60)
        for i, eq in enumerate(denorm_equations):
            print(f"  {eq}")
        print("=" * 60)
    
    def get_transformation_summary(self) -> Dict:
        """
        Get a summary of the transformation parameters.
        
        Returns:
            Dict: Dictionary containing transformation details
        """
        summary = {
            'variables': self.variables,
            'n_variables': self.n_vars,
            'A_diagonal': [float(self.A_sym[i, i]) for i in range(self.n_vars)],
            'b_vector': [float(self.b_sym[i]) for i in range(self.n_vars)],
            'A_inv_diagonal': [float(self.A_inv_sym[i, i]) for i in range(self.n_vars)],
            'bounds': {var: (self.L[var], self.U[var]) for var in self.variables}
        }
        return summary
    
    def verify_transformation(self, test_values: Dict[str, float]) -> Dict:
        """
        Verify the transformation u = A*x + b for given test values.
        
        Args:
            test_values (Dict[str, float]): Dictionary of original variable values
            
        Returns:
            Dict: Dictionary containing original values, normalized values, and recovered values
        """
        x_vec = np.array([test_values[var] for var in self.variables])
        
        # Forward: u = A*x + b
        u_vec = self.A_np @ x_vec + self.b_np
        
        # Backward: x = A^(-1) * (u - b)
        x_recovered = self.A_inv_np @ (u_vec - self.b_np)
        
        result = {
            'original_x': {var: x_vec[i] for i, var in enumerate(self.variables)},
            'normalized_u': {var: u_vec[i] for i, var in enumerate(self.variables)},
            'recovered_x': {var: x_recovered[i] for i, var in enumerate(self.variables)},
            'recovery_error': {var: abs(x_vec[i] - x_recovered[i]) 
                             for i, var in enumerate(self.variables)}
        }
        
        return result