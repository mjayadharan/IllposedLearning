import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial, chebyshev, legendre
from typing import Iterator, Tuple
from numpy.typing import NDArray
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
import re
import pysindy as ps
from pysindy.utils import AxesArray
from pysindy.utils import comprehend_axes
from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item

from dae_finder import PolyFeatureMatrix

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


def standardize_columns(df,time_col=None):
    # Convert variables to x1,x2,x3,...
    if time_col is None:
        candidate_names = {"t_Exp","t_fine","time"}
        found = [c for c in df.columns if c in candidate_names]
        if not found:
            raise ValueError("Uable to auto-detect time column. Need provide 'time_col'")
        time_col = found[0]
    elif isinstance(time_col,str):
        time_col = [time_col]
    var_cols = [c for c in df.columns if c not in time_col]
    new_names = {}
    for idx, col in enumerate(var_cols, start=1):
        if not re.fullmatch(r"x\d+", str(col)):
            new_names[col] = f"x{idx}"
    return df.rename(columns=new_names),time_col


class Monomials:
    def __init__(self,data,degree):
        # data: DataFrame
        # original data including the time column, columns could haven't been renamed
        # degreee: specific single degree
        self.data = data
        self.degree = degree
        # data_states only has state columns and has been renamed
        data_std,self.time_col = standardize_columns(self.data)
        self.data_states = data_std.drop(columns=self.time_col)
    
    def _generate_library(self):
        poly_feature_ob = PolyFeatureMatrix(self.degree)
        candidate_lib_full = poly_feature_ob.fit_transform(self.data_states)
        candidate_lib = candidate_lib_full.drop(["1"], axis=1)
        return candidate_lib
    
class OrthogonalLibrary(BaseFeatureLibrary):
    """Generate Chebyshev polynomial features.
    This library generates features using Chebyshev polynomials of the first kind.

    Parameters
    ----------
    degree : integer, default 2
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
        degree,
        method,
        include_interaction=True,
        interaction_only=False,
        include_bias=False,
    ):
        super().__init__()
        self.method = method
        self.degree = degree
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        # Stable, shared ordering for transform() and get_feature_names()
        self._powers_sorted = None  # list[tuple[int,...]] set in fit()
        self._n_features = None     # cache input dimensionality

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
        #if n == 0:
            #return np.ones_like(x)
        #elif n == 1:
            #return x
        #else:
            #T_prev_prev = np.ones_like(x)  # T_0
            #T_prev = x                      # T_1
            
            #for i in range(2, n + 1):
                #T_curr = 2 * x * T_prev - T_prev_prev
                #T_prev_prev = T_prev
                #T_prev = T_curr
            
            #return T_prev
        T_1 = chebyshev.Chebyshev.basis(n)
        T = T_1.convert(kind=Polynomial)

        return T(x)
    
    def _legendre_polynomial(self,x: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the n-th Legendre polynomial P_n(x).
        """
        T = legendre.Legendre.basis(n)
        P = T.convert(kind=Polynomial)

        return P(x)

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
        if self._powers_sorted is None:
            # Fallback to combinations if fit() hasn't cached (shouldn't happen)
            combinations = list(self._combinations(
                n_features=self.n_features_in_,
                degree=self.degree,
                include_interaction=self.include_interaction,
                interaction_only=self.interaction_only,
                include_bias=self.include_bias,
            ))
            powers_sorted = sorted(combinations, key=lambda row: (np.sum(row), tuple(row)))
            return np.array(powers_sorted, dtype=np.int_)
        return np.array(self._powers_sorted, dtype=np.int_)

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
        powers = self._powers_sorted
        if powers is None:
            # Fallback to property (still stable)
            powers = [tuple(int(v) for v in row) for row in self.powers_]

        # Input feature names
        if input_features is None:
            input_features = [f"x{i+1}" for i in range(self.n_features_in_)]

        prefix = 'T_' if str(self.method).lower().startswith('cheb') else 'P_'
        names = []
        for row in powers:
            terms = []
            for feat_idx, deg in enumerate(row):
                if deg <= 0:
                    continue
                if int(deg) == 1:
                    terms.append(f"{prefix}1({input_features[feat_idx]})")
                else:
                    terms.append(f"{prefix}{int(deg)}({input_features[feat_idx]})")
            if not terms:
                names.append('1')
            else:
                names.append(' '.join(terms))
        return names

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
        # Cache dimensionality and a stable, shared order of powers
        self._n_features = n_features
        powers_sorted = sorted(combinations, key=lambda row: (np.sum(row), tuple(row)))
        self._powers_sorted = [tuple(int(v) for v in row) for row in powers_sorted]
        self.n_output_features_ = len(self._powers_sorted)
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
        if self._powers_sorted is None:
            raise RuntimeError("OrthogonalLibrary not fitted: powers order is missing.")

        xp_full = []
        for x in x_full:
            if sparse.issparse(x):
                raise NotImplementedError("Sparse matrices not yet supported for OrthogonalLibrary")

            # Determine coordinate axis and validate shape
            coord_axis = self._get_coord_axis(x)
            n_features = x.shape[coord_axis]
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")

            # Create output container
            output_shape = list(x.shape)
            output_shape[coord_axis] = self.n_output_features_
            xp = AxesArray(
                np.empty(output_shape, dtype=x.dtype),
                x.axes if hasattr(x, 'axes') else comprehend_axes(x),
            )

            # Helper to slice out a single feature along coord_axis
            def take_feature(arr, idx):
                if coord_axis == -1:
                    return arr[..., idx]
                idx_tuple = [slice(None)] * arr.ndim
                idx_tuple[coord_axis] = idx
                return arr[tuple(idx_tuple)]

            # Build columns in the cached, stable order
            for i, powers in enumerate(self._powers_sorted):
                # Start with ones on the sample axes
                result = np.ones(x.shape[:coord_axis] + x.shape[coord_axis+1:], dtype=x.dtype)
                for feat_idx, deg in enumerate(powers):
                    if deg <= 0:
                        continue
                    feat = take_feature(x, feat_idx)
                    if self.method == 'Chebyshev':
                        val = self._chebyshev_polynomial(feat, int(deg))
                    else:  # Legendre
                        val = self._legendre_polynomial(feat, int(deg))
                    result = result * val

                # Write column i
                if coord_axis == -1:
                    xp[..., i] = result
                else:
                    idx_tuple = [slice(None)] * xp.ndim
                    idx_tuple[coord_axis] = i
                    xp[tuple(idx_tuple)] = result

            xp_full.append(xp)
        return xp_full
    
    def transform_to_dataframe(self, x: np.ndarray, input_feature_names: list[str] = None):
        """
        Transform the input array to a pandas DataFrame where each column is a basis function.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input data to be transformed.
        
        input_feature_names : list of str
            Names for the input features
        
        Returns
        -------
        df : pandas.DataFrame
            A DataFrame where each column corresponds to a basis function and rows to samples.
        """
        check_is_fitted(self)
        x_transformed = self.transform([x])[0]  # Transform expects a list of arrays
        feature_names = self.get_feature_names(input_feature_names)
        return pd.DataFrame(x_transformed, columns=feature_names)


def n_features(
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
    combinations = list(OrthogonalLibrary._combinations(
    n_in_feat, degree, include_interaction, interaction_only, include_bias
    ))
    return len(combinations)


def IdentityChebyshevLibrary():
    """
    Generate an identity library using Chebyshev polynomials which maps all 
    input features to themselves. This is equivalent to T_1(x_i) for each feature.
    """
    return OrthogonalLibrary(degree=1, method='Chebyshev',include_bias=False)

def IdentityLegendreLibrary():
    return OrthogonalLibrary(degree=1, method='Legendre',include_bias=False)