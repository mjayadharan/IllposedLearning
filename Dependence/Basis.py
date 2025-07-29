import numpy as np
from numpy.polynomial import Polynomial, chebyshev, legendre
import pandas as pd
from dae_finder import PolyFeatureMatrix
import Comparison
from itertools import product
import sympy as sp
import pysindy as ps

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


#class Orthogonal_Basis:
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
    

#class Orthogonal_Basis:
    def __init__(self,data,degree,method):
        self.data = data
        self.degree = degree
        self.method = method
        
        data_std,self.time_col = Comparison.standardize_columns(data)
        self.data_states = data_std.copy().drop(columns={self.time_col})
        self.data_norm, self.L, self.U = self._normalization()
        self.library_functions, self.library_function_names = self._build_orthogonal_func()
        #self.val_states, self.expr_states = self._single_library()
        #self.library = self._build_function_library()
    
    def _normalization(self):
        # data_states: DataFrame, only has state columns
        data_norm = self.data_states.copy()
        L,U = {},{}
        for col in data_norm.columns:
            Li = self.data_states[col].min()
            Ui = self.data_states[col].max()
            L[col] = Li
            U[col] = Ui
            if Li == Ui:
                data_norm[col] = 0
                print("Warning: All values are equal")
            else:
                data_norm[col] = 2*(self.data_states[col]-Li)/(Ui-Li)-1
        return data_norm, L, U
    
    #def _build_orthogonal_func(self):
        states = self.data_states.columns.tolist()
        library_functions = []
        library_function_names = []
        for deg_combo in product(range(self.degree+1),repeat=len(states)):
            total_deg = sum(deg_combo)
            if total_deg == 0 or total_deg > self.degree:
                continue

            poly_list = []
            name_parts = []
            for i, deg in enumerate(deg_combo):
                s = states[i]
                if self.method == 'Chebyshev':
                    T = chebyshev.Chebyshev.basis(deg)
                    E = T.convert(kind=Polynomial)
                elif self.method == 'Legendre':
                    P = legendre.Legendre.basis(deg)
                    E = P.convert(kind=Polynomial)
                else:
                    raise ValueError("Basis currently unavailable")
                
                poly_list.append((i, E))
                # Create name for this term
                if self.method == 'Chebyshev':
                    name_parts.append(f"T_{deg}({s})")
                elif self.method == 'Legendre':
                    name_parts.append(f"P_{deg}({s})")

            if poly_list:
                library_functions.append(self._make_interaction_function(poly_list))
                function_name = " * ".join(name_parts)
                library_function_names.append(function_name)

            return library_functions, library_function_names

    def _build_orthogonal_func(self):
        library_functions = []
        library_function_names = []
        for deg in range(self.degree+1):
            if self.method == 'Chebyshev':
                T = chebyshev.Chebyshev.basis(deg)
                E = T.convert(kind=Polynomial)
                library_functions.append(self._make_poly_function(E))
                expr_str = str(E).split('↦')[-1].strip()
                library_function_names.append(expr_str)
            elif self.method == 'Legendre':
                P = legendre.Legendre.basis(deg)
                E = P.convert(kind=Polynomial)
                library_functions.append(self._make_poly_function(E))
                expr_str = str(E).split('↦')[-1].strip()
                library_function_names.append(expr_str)

        return library_functions, library_function_names
    
    def _make_poly_function(self, poly):
        """Create a function that properly handles AxesArray inputs"""
        def poly_func(x):
            # Convert AxesArray to regular numpy array if needed
            if hasattr(x, 'array'):
                x_array = x.array
            else:
                x_array = np.asarray(x)
            return poly(x_array)
        return poly_func
    
    #def _make_interaction_function(self, poly_list):
        """Create a function that evaluates the product of multiple polynomials on different variables"""
        def interaction_func(x):
            # Convert AxesArray to regular numpy array if needed
            if hasattr(x, 'array'):
                x_array = x.array
            else:
                x_array = np.asarray(x)
            
            # Start with ones
            result = np.ones(x_array.shape[0])
            
            # Multiply by each polynomial evaluated on its corresponding column
            for col_idx, poly in poly_list:
                poly_values = poly(x_array[:, col_idx])
                result *= poly_values
            
            return result
        
        return interaction_func
    
    def _get_pysindy_library(self):
        return ps.CustomLibrary(
            library_functions = self.library_functions,
            function_names = [lambda x, name=name: name for name in self.library_function_names],
            interaction_only = False
            #function_names = self.library_function_names
        )
    
    
from typing import Iterator, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
import warnings

# 导入pysindy的必要组件
import pysindy as ps
from pysindy.utils import AxesArray
from pysindy.utils import comprehend_axes
from pysindy.utils import wrap_axes
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.feature_library.base import x_sequence_or_item


class ChebyshevLibrary(BaseFeatureLibrary):
    """Generate Chebyshev polynomial features.

    This library generates features using Chebyshev polynomials of the first kind.
    Chebyshev polynomials are orthogonal polynomials that are particularly useful
    for approximation and numerical analysis.

    Parameters
    ----------
    degree : integer, optional (default 2)
        The maximum degree of the Chebyshev polynomial features.

    include_interaction : boolean, optional (default True)
        Determines whether interaction features are produced.
        If false, features are all of the form ``T_k(x[i])`` where T_k is
        the k-th Chebyshev polynomial.

    interaction_only : boolean, optional (default False)
        If true, only interaction features are produced: features that are
        products of Chebyshev polynomials of distinct input features.

    include_bias : boolean, optional (default True)
        If True (default), then include a bias column (T_0(x) = 1).

    Attributes
    ----------
    powers_ : array, shape (n_output_features, n_input_features)
        powers_[i, j] is the degree of the Chebyshev polynomial of the jth input 
        in the ith output feature.

    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features.
    """

    def __init__(
        self,
        degree=2,
        include_interaction=True,
        interaction_only=False,
        include_bias=True,
    ):
        super().__init__()
        self.degree = degree
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _get_coord_axis(self, x):
        """Get the coordinate axis"""
        if hasattr(x, 'ax_coord'):
            return x.ax_coord
        elif hasattr(x, 'axes') and 'ax_coord' in x.axes:
            return x.axes['ax_coord']
        else:
            # Default: assume the last axis is the coordinate axis
            return -1

    @staticmethod
    def _chebyshev_polynomial(x: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the n-th Chebyshev polynomial of the first kind T_n(x).
        
        Uses the recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        """
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            T_prev_prev = np.ones_like(x)  # T_0
            T_prev = x                      # T_1
            
            for i in range(2, n + 1):
                T_curr = 2 * x * T_prev - T_prev_prev
                T_prev_prev = T_prev
                T_prev = T_curr
            
            return T_prev

    @staticmethod
    def _combinations(
        n_features: int,
        degree: int,
        include_interaction: bool,
        interaction_only: bool,
        include_bias: bool,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Create selection tuples of input indexes and degrees for each Chebyshev term.
        
        Each tuple represents the degrees of Chebyshev polynomials for each feature.
        """
        if not include_interaction:
            # Only single-variable terms
            combinations = []
            if include_bias:
                combinations.append(tuple(0 for _ in range(n_features)))
            
            for feat_idx in range(n_features):
                for deg in range(1, degree + 1):
                    powers = [0] * n_features
                    powers[feat_idx] = deg
                    combinations.append(tuple(powers))
            
            return iter(combinations)
        else:
            # Multi-variable combinations
            if interaction_only:
                # Only terms where each feature appears at most once
                combinations = []
                if include_bias:
                    combinations.append(tuple(0 for _ in range(n_features)))
                
                # Generate all combinations of features with degree <= 1
                for total_degree in range(1, degree + 1):
                    for selected_features in combinations(range(n_features), total_degree):
                        powers = [0] * n_features
                        for feat_idx in selected_features:
                            powers[feat_idx] = 1
                        combinations.append(tuple(powers))
                
                return iter(combinations)
            else:
                # All combinations up to total degree
                combinations = []
                
                def generate_combinations(current_powers, remaining_degree, start_feature):
                    if start_feature == n_features:
                        if sum(current_powers) <= degree:
                            combinations.append(tuple(current_powers))
                        return
                    
                    for deg in range(remaining_degree + 1):
                        new_powers = current_powers + [deg]
                        generate_combinations(new_powers, remaining_degree - deg, start_feature + 1)
                
                generate_combinations([], degree, 0)
                
                # Filter based on include_bias
                if not include_bias:
                    combinations = [c for c in combinations if sum(c) > 0]
                
                return iter(combinations)

    @property
    def powers_(self) -> NDArray[np.int_]:
        """
        The degrees of the Chebyshev polynomials as an array of shape
        (n_features_out, n_features_in), where each item is the degree of the
        Chebyshev polynomial of the jth input variable in the ith feature.
        """
        check_is_fitted(self)
        combinations = list(self._combinations(
            n_features=self.n_features_in_,
            degree=self.degree,
            include_interaction=self.include_interaction,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        ))
        return np.array(combinations, dtype=np.int_)

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        powers = self.powers_
        if input_features is None:
            input_features = ["x%d" % i for i in range(powers.shape[1])]
        
        feature_names = []
        for row in powers:
            terms = []
            for feat_idx, degree in enumerate(row):
                if degree > 0:
                    if degree == 1:
                        terms.append(f"T_1({input_features[feat_idx]})")
                    else:
                        terms.append(f"T_{degree}({input_features[feat_idx]})")
            
            if len(terms) == 0:
                name = "1"  # Constant term (T_0)
            else:
                name = " ".join(terms)
            
            feature_names.append(name)
        
        return feature_names

    @x_sequence_or_item
    def fit(self, x_full: list[AxesArray], y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x_full : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        if self.degree < 0 or not isinstance(self.degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not self.include_interaction) and self.interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only be True"
            )
        
        # Get the number of features
        coord_axis = self._get_coord_axis(x_full[0])
        n_features = x_full[0].shape[coord_axis]
        
        combinations = list(self._combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        ))
        
        self.n_features_in_ = n_features
        self.n_output_features_ = len(combinations)
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to Chebyshev polynomial features.

        Parameters
        ----------
        x_full : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray or CSR/CSC sparse matrix,
                shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number
            of Chebyshev polynomial features generated.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            if sparse.issparse(x):
                raise NotImplementedError("Sparse matrices not yet supported for ChebyshevLibrary")
            
            # Get the number of axes and features
            coord_axis = self._get_coord_axis(x)
            n_features = x.shape[coord_axis]
            
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")

            combinations = list(self._combinations(
                n_features,
                self.degree,
                self.include_interaction,
                self.interaction_only,
                self.include_bias,
            ))

            # Create output array
            output_shape = list(x.shape)
            output_shape[coord_axis] = self.n_output_features_
            
            xp = AxesArray(
                np.empty(output_shape, dtype=x.dtype),
                x.axes if hasattr(x, 'axes') else comprehend_axes(x),
            )

            for i, powers in enumerate(combinations):
                # Compute the product of Chebyshev polynomials
                result = np.ones(x.shape[:-1] if coord_axis == -1 else 
                               [x.shape[j] for j in range(len(x.shape)) if j != coord_axis], 
                               dtype=x.dtype)
                
                for feat_idx, degree in enumerate(powers):
                    if degree > 0:
                        # Get features from the specified axis
                        if coord_axis == -1:
                            feature_data = x[..., feat_idx]
                        else:
                            # Create index tuple
                            idx = [slice(None)] * len(x.shape)
                            idx[coord_axis] = feat_idx
                            feature_data = x[tuple(idx)]
                        
                        chebyshev_val = self._chebyshev_polynomial(feature_data, degree)
                        result = result * chebyshev_val
                
                # Put the results into the output array
                if coord_axis == -1:
                    xp[..., i] = result
                else:
                    idx = [slice(None)] * len(xp.shape)
                    idx[coord_axis] = i
                    xp[tuple(idx)] = result
            
            xp_full.append(xp)
        
        return xp_full

def n_chebyshev_features(
    n_in_feat: int,
    degree: int,
    include_bias: bool = False,
    include_interaction: bool = True,
    interaction_only: bool = False,
) -> int:
    """Calculate number of Chebyshev polynomial features

    Args:
        n_in_feat: number of input features, e.g. 3 for x, y, z
        degree: maximum polynomial degree, e.g. 2 for up to T_2
        include_bias: whether to include a constant term (T_0 = 1)
        include_interaction: whether to include terms mixing multiple inputs
        interaction_only: whether to omit terms of T_m(x_i) * T_n(x_j) for m,n > 1
    """
    if not include_interaction and interaction_only:
        raise ValueError("Cannot set interaction only if include_interaction is False")
    
    # Use the combinations generator to count features
    combinations = list(ChebyshevLibrary._combinations(
        n_in_feat, degree, include_interaction, interaction_only, include_bias
    ))
    return len(combinations)


def IdentityChebyshevLibrary():
    """
    Generate an identity library using Chebyshev polynomials which maps all 
    input features to themselves. This is equivalent to T_1(x_i) for each feature.
    """
    return ChebyshevLibrary(degree=1, include_bias=False)


