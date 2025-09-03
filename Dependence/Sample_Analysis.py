import pandas as pd
import numpy as np
import tellurium as te
import os
from scipy.stats import qmc
from scipy.stats import chisquare
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from Comparison import standardize_columns, run_noise_free_analysis
from Basis import normalization
from Base_test import Lotka_Volterra, CRN


# Helper for parallel analysis
def _process_combo(data, distribution, sbml_path, model_name, model_kwargs,
                   num, n_step, start, end, degree_list, comb_list, method, out_dir_str):
    """Worker: run one (num, n_step) combo and return summary dict."""
    # Ensure directory exists in worker
    os.makedirs(out_dir_str, exist_ok=True)

    # Build sampler and simulate
    Sample = sampling(data, int(num), distribution=distribution)
    if sbml_path is not None:
        filepaths = Sample.fit_and_save(sbml_path, out_dir_str, start, end, int(n_step))
    elif model_name is not None:
        mk = model_kwargs or {}
        filepaths = Sample.fit_and_save_base(model_name=model_name,
                                             out_path=out_dir_str,
                                             start=start,
                                             end=end,
                                             n_step=int(n_step),
                                             **mk)
    else:
        raise ValueError("Either sbml_path or model_name must be provided to simulate trajectories.")

    # Subsample
    subsample = Sample._subsample(filepaths)

    # Resolve time column robustly: prefer 'time' from simulation outputs
    available_cols = subsample.columns
    if 'time' in available_cols:
        _time_cols = ['time']
    else:
        # fall back to Sample.time_col if present in subsample
        if isinstance(Sample.time_col, (list, tuple)):
            _time_cols = [c for c in Sample.time_col if c in available_cols]
        else:
            _time_cols = [Sample.time_col] if Sample.time_col in available_cols else []

    state_cols = list(_time_cols) + list(Sample.ID)
    subsample_states = subsample[state_cols]

    # Analysis according to distribution/method
    dist_lower = distribution.lower()
    if dist_lower == 'uniform':
        if method in ('Monomial', 'Legendre'):
            summary_result = run_noise_free_analysis(subsample_states, degree_list, comb_list, method)
        else:
            raise ValueError("Cannot assign basis for this distribution, try another distribution")
    elif dist_lower == 'arcsine':
        if method == 'Chebyshev':
            summary_result = run_noise_free_analysis(subsample_states, degree_list, comb_list, method)
        else:
            raise ValueError("Cannot assign basis for this distribution, try another distribution")
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return {
        'num': int(num),
        'start': float(start),
        'end': float(end),
        'n_step': int(n_step),
        'summary': summary_result,
        'method': method,
    }

def sobol_u01(num,d,seed=0):
    # Sobol seeds (with random shift) and FPS
    m = int(np.ceil(np.log2(max(2,num))))
    sampler = qmc.Sobol(d=d,scramble=True,seed=seed)
    u01 = sampler.random_base2(m=m)[:num]

    rng = np.random.default_rng(seed + 12345)
    shift = rng.random(d)

    return (u01 + shift) % 1.0

def fps_indices(X,M,seed=0):
    # Farthest Point Sampling on rows of X (numpy array)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    M = int(min(max(1,M),N))
    chosen = np.empty(M,dtype=int)
    chosen[0] = rng.integers(N)
    dist = np.linalg.norm(X - X[chosen[0]], axis=1)

    for i in range(1,M):
        chosen[i] = int(np.argmax(dist))
        dist = np.minimum(dist,np.linalg.norm(X - X[chosen[i]],axis=1)
                          )
    return chosen

class create_initial_conditions:
    """
    This class creates a group of initial conditions.
    These conditions are uniformly/arcsine distributed in the original state space
    """
    def __init__(self,data,num,distribution='uniform'):
        # data: original dataframe including time and states columns, they are not necessary standardized or normalized.
        # num: the number of initial conditions in the group
        # distribution: expected distribution including arcsine and uniform, the defeault setting is uniform
        self.data = data
        self.num = num
        self.distribution = distribution

        data_std, self.time_col = standardize_columns(self.data)
        self.data_states = data_std.drop(columns=self.time_col)
        self.data_norm, self.L, self.U = normalization(self.data_states)

        self.n_features = len(self.data_norm.columns.tolist())
        # sample_orig: sampled on the data_states(original)
        # sample_norm: sampled on the data_norm(normalized)
        if self.distribution.lower() == 'uniform':
            self.sample_orig, self.sample_norm = self._uniformly_sample()
        elif self.distribution.lower() == 'arcsine':
            self.sample_orig, self.sample_norm = self._arcsine_sample()

    def _uniformly_sample(self):
        L_orig, U_orig = self.data_states.min(), self.data_states.max()
        L_norm, U_norm = self.data_norm.min(), self.data_norm.max()
        """
        # Initial conditions in the form of num * n_features
        random_initial_original = np.random.uniform(low=L_orig, high=U_orig, size=(self.num,self.n_features))
        random_initial_normalized = np.random.uniform(low=L_norm, high=U_norm, size=(self.num,self.n_features))
        """
        cols = self.data_states.columns.tolist()
        d = len(cols)
        # Sobol + random shift in [0,1]^d
        u01 = sobol_u01(self.num,d=d,seed=0)
        # Map to original and normalized ranges
        Lo, Uo = L_orig.values, U_orig.values
        Ln, Un = L_norm.values, U_norm.values
        X_orig = Lo + u01 * (Uo - Lo)
        X_norm = Ln + u01 * (Un - Ln)
        return pd.DataFrame(X_orig, columns=cols), pd.DataFrame(X_norm, columns=cols)
        #return pd.DataFrame(random_initial_original,columns=self.data_states.columns), pd.DataFrame(random_initial_normalized,columns=self.data_states.columns)
    
    def _arcsine_sample(self):
        L_orig, U_orig = self.data_states.min(), self.data_states.max()
        L_norm, U_norm = self.data_norm.min(), self.data_norm.max()
        cols = self.data_states.columns.tolist()
        d = len(cols)
        # Sobol + random shift in [0,1]^d
        u01 = sobol_u01(self.num,d=d,seed=0)
        # arcsine transformation
        x = np.cos(np.pi * u01)
        x = np.clip(x, -1+1e-7, 1-1e-7)

        Lo, Uo = L_orig.values, U_orig.values
        X_orig = 0.5 * (x+1.0) * (Uo-Lo) + Lo

        return pd.DataFrame(X_orig, columns=cols), pd.DataFrame(x, columns=cols)
        """
        L_orig_array, U_orig_array = L_orig.values, U_orig.values
        L_norm_array, u_norm_array = L_norm.values, U_norm.values
        # Initial conditions in the form of num * n_features
        u = np.random.uniform(low=L_norm_array,high=u_norm_array,size=(self.num,self.n_features))
        x = np.cos(np.pi * u)
        arcsine_X_original = 0.5 * (x + 1) * (U_orig_array - L_orig_array) + L_orig_array
        return pd.DataFrame(arcsine_X_original,columns=self.data_states.columns), pd.DataFrame(x,columns=self.data_states.columns)
        """


class sampling:
    """
    This class is used for sampling under different initial conditions and finally deriving the time series 
    distributed as a specific distribution.
    """
    def __init__(self,data,num,distribution='uniform'):
        # num: how many initial conditions are expected to be generated
        # n_step: how many steps are expected to simulated
        self.data = data
        self.num = num
        self.distribution = distribution

        Creator = create_initial_conditions(self.data,self.num,self.distribution)
        self.sample_orig = Creator.sample_orig
        self.data_states = Creator.data_states
        self.time_col = Creator.time_col
        self.state_cols = self.data_states.columns.tolist()

    def _prepare_outdir(self, out_path, n_step, filename_prefix):
        """Ensure output directory contains a run tag with num & n_step.
        If `out_path` already includes the tag, reuse it; otherwise create
        a subdirectory like: {filename_prefix}_{distribution}_n{num}_pts{n_step}.
        Returns (final_dir, tag).
        """
        tag = f"n{int(self.num)}_pts{int(n_step)}"
        base = str(out_path)
        # If any path component already contains the tag, don't nest again
        if any(tag in part for part in Path(base).parts):
            final_dir = base
        else:
            final_dir = os.path.join(base, f"{filename_prefix}_{self.distribution}_{tag}")
        os.makedirs(final_dir, exist_ok=True)
        return final_dir, tag

    def fit_and_save(self,sbml_path:str,out_path:str,start,end,n_step,
                     filename_prefix="IC"):
        out_dir, tag = self._prepare_outdir(out_path, n_step, filename_prefix)
        rr = te.loadSBMLModel(sbml_path)
        species_ids = list(rr.model.getFloatingSpeciesIds())
        self.ID = species_ids
        self.sample_orig.columns = species_ids
        ic_df = self.sample_orig.reindex(columns=species_ids).copy()
        rr.timeCourseSelections = ['time'] + species_ids

        # Load parameters dictionary
        param_dict = {}
        for param in rr.model.getGlobalParameterIds():
            val = rr.model[param]
            param_dict[param] = val
        # align columns to model species order; missing columns become NaN
        ic_df = self.sample_orig

        filepaths = []
        # ensure integer number of points and include both endpoints (step=5 in the Beer model)
        for k in range(len(ic_df)):
            row = ic_df.iloc[k]
            for j, s in enumerate(species_ids):
                val = row[s]
                if not pd.isna(val):
                    v = float(val)
                    rr[f'init({s})'] = v

            rr.reset()
            res = rr.simulate(start, end, n_step)
            df = pd.DataFrame(res, columns=rr.timeCourseSelections)

            rates_list = []
            for t in range(len(df)):
                for j, s in enumerate(species_ids):
                    rr[s] = df.loc[t, s]
                rates = rr.getRatesOfChange()
                rates_list.append(rates)

            derivative_df = pd.DataFrame(rates_list, columns=[f'd{s}/dt' for s in species_ids])
            data_augmented = pd.concat([df, derivative_df], axis=1)

            fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
            data_augmented.to_excel(fp, index=False)
            filepaths.append(fp)

        return filepaths
    
    def fit_and_save_base(self, model_name: str, out_path, start, end, n_step,
                          filename_prefix="IC", **model_kwargs):
        """
        Simulate and save trajectories for models implemented in Base_test
        (i.e., not SBML-based). The output schema matches `fit_and_save`:
        one Excel file per initial condition with columns [time, states, d(state)/dt].

        Parameters
        ----------
        model_name : str
            Currently supports 'Lotka-Volterra','Chemical Reaction Network'
        **model_kwargs :
            Extra parameters required by a given model. 
            For Lotka-Volterra:
              - params: dict with keys {alpha, beta, gamma, delta}
              - noise_level: float or None
              - state_names: list[str] giving the model's internal state names
                (default ["x","y"]).
            For Chemical Reaction Network:
              - k_rates: dict with keys {k,kr,kcat}
              - noise_level: float or None
              - state_names: list[str] giving the model's internal state names
                (default ["S","E","ES","P"]).
        """
        out_dir, tag = self._prepare_outdir(out_path, n_step, filename_prefix)

        # Use the standardized names coming from create_initial_conditions, e.g. ['x1','x2', ...]
        std_names = list(self.sample_orig.columns)
        self.ID = std_names  # downstream methods rely on this

        # Time grid
        t_eval = np.linspace(float(start), float(end), int(n_step))
        t_span = (float(start), float(end))

        filepaths = []

        if model_name.lower() == 'lotka-volterra':
            # Map standardized names (x1,x2,...) to the Base_test internal names (x,y)
            model_state_names = model_kwargs.get('state_names', ["x", "y1","y2"])  # order matters
            if len(model_state_names) != 3:
                raise ValueError("Lotka-Volterra expects exactly 3 model state names.")
            if len(std_names) < 3:
                raise ValueError("Lotka-Volterra requires at least three standardized state columns (e.g., x1,x2,x3).")

            # Build a consistent mapping between model names and standardized names, in order
            model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
            std_in_model_order = [model_to_std[m] for m in model_state_names]  # e.g. ['x1','x2']

            params = model_kwargs.get('params')
            if params is None:
                raise ValueError("'params' must be provided for Lotka-Volterra in model_kwargs.")
            noise_level = model_kwargs.get('noise_level', None)

            for k in range(len(self.sample_orig)):
                row = self.sample_orig.iloc[k]
                # Initial condition in the model's variable order (x,y)
                z0= [float(row[s]) for s in std_in_model_order]

                # Simulate via the Base_test class
                lv = Lotka_Volterra(params=params, t_span=t_span, t_eval=t_eval, z0=z0, noise_level=noise_level)
                df_raw = lv.data_sim.copy()  # columns: ['time','x','y']

                # Rename model outputs (x,y) -> standardized names (x1,x2) for consistency with the rest of the pipeline
                rename_map = {m: model_to_std[m] for m in model_state_names}
                df = df_raw.rename(columns=rename_map)

                # Compute derivative at each time using the model RHS, feeding values in (x,y) order
                deriv = []
                for i in range(len(df)):
                    t = float(df.loc[i, 'time']) if 'time' in df.columns else float(t_eval[i])
                    # Values in model order
                    state_vec_model_order = [float(df.loc[i, model_to_std[m]]) for m in model_state_names]
                    dstate = lv.lv_rhs_1_2(t, state_vec_model_order)
                    deriv.append(dstate)
                deriv_df = pd.DataFrame(deriv, columns=[f"d{model_to_std[m]}/dt" for m in model_state_names])

                data_augmented = pd.concat([df, deriv_df], axis=1)
                fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                data_augmented.to_excel(fp, index=False)
                filepaths.append(fp)

        elif model_name.lower() == 'crn':
            model_state_names = model_kwargs.get('state_names', ["S","E","ES","P"])

            # Build a consistent mapping between model names and standardized names, in order
            model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
            std_in_model_order = [model_to_std[m] for m in model_state_names]  # e.g. ['x1','x2','x3','x4']

            k_rates = model_kwargs.get('k_rates')
            if k_rates is None:
                raise ValueError("'k_rates' must be provided for Chemical Reaction Network in model_kwargs.")
            noise_level = model_kwargs.get('noise_level', None)

            for k in range(len(self.sample_orig)):
                row = self.sample_orig.iloc[k]
                # Initial condition in the model's variable order (x,y)
                init_cond = [float(row[s]) for s in std_in_model_order]

                # Simulate via the Base_test class
                crn = CRN(k_rates=k_rates, init_cond=init_cond, solvedT=t_eval, noise_level=noise_level)
                df_raw = crn.data_sim.copy()  # columns: ['time','x','y']

                # Rename model outputs (x,y) -> standardized names (x1,x2) for consistency with the rest of the pipeline
                rename_map = {m: model_to_std[m] for m in model_state_names}
                df = df_raw.rename(columns=rename_map)

                # Compute derivative at each time using the model RHS, feeding values in (x,y) order
                deriv = []
                for i in range(len(df)):
                    t = float(df.loc[i, 'time']) if 'time' in df.columns else float(t_eval[i])
                    # Values in model order
                    state_vec_model_order = [float(df.loc[i, model_to_std[m]]) for m in model_state_names]
                    dstate = crn.toyEnzRHS(state_vec_model_order, t)
                    deriv.append(dstate)
                deriv_df = pd.DataFrame(deriv, columns=[f"d{model_to_std[m]}/dt" for m in model_state_names])

                data_augmented = pd.concat([df, deriv_df], axis=1)
                fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                data_augmented.to_excel(fp, index=False)
                filepaths.append(fp)
        
        else:
            raise ValueError(f"Unsupported base model: {model_name}")

        return filepaths
    
    def _subsample(
        self,
        filepaths,
        n_bins: int = 10,
        k_max: int = 1,
    ):
        """
        Parameters
        ----------
        filepaths : list[str]
            Paths returned by ``fit_and_save`` (each is an Excel file that holds one
            trajectory with states *and* their derivatives).
        n_bins : int, default 10
            Number of equal‑width bins per dimension (stratified grid).
        k_max : int, default 1
            Maximum number of rows kept per grid cell.

        Returns
        -------
        pd.DataFrame
            Subsampled dataframe containing *all* original columns
            (time, states, derivatives).
        """
        # Load and concatenate every trajectory dataframe
        dfs = [pd.read_excel(fp) for fp in filepaths]
        data = pd.concat(dfs, ignore_index=True)

        X = data[self.ID]

        # Affine‑map each state to [-1, 1]
        L = X.min()
        U = X.max()
        X_norm = 2 * (X - L) / (U - L) - 1

        if self.distribution.lower() == 'uniform':
            """
            # Assign every point to a grid cell
            idx = ((X_norm + 1) * (n_bins / 2)).astype(int).clip(0, n_bins - 1)
            idx_cols = [f"b{i}" for i in range(len(self.ID))]
            bins = pd.DataFrame(idx.values, columns=idx_cols)

            # Stratified sampling: keep ≤ k_max rows per cell
            chosen = []
            for _, g in data.join(bins).groupby(idx_cols, sort=False):
                chosen.append(g.sample(min(k_max, len(g)), random_state=0))
            result = pd.concat(chosen, ignore_index=True)
            """
            d = len(self.ID)
            target = min(len(data),int((n_bins ** d) * k_max))
            idx = fps_indices(X_norm.to_numpy(),M=target,seed=0)
            result = data.iloc[idx].reset_index(drop=True)
        elif self.distribution.lower() == 'arcsine':
            eps = 1e-7
            # Clip the edge
            Xn = X_norm.clip(-1+eps,1-eps).to_numpy()
            N, d = Xn.shape

            # density log t(x) = - 0.5 * sum_j log(1-x_j^2)
            log_t = -0.5 * np.log(1.0 - Xn**2).sum(axis=1)

            # 
            edges = np.linspace(-1,1,n_bins+1)
            widths = np.diff(edges)
            log_p = np.zeros(N)
            for j in range(d):
                # The frequency of this dimension
                counts, _ = np.histogram(Xn[:,j],bins=edges)
                # Frequency -> Density (avoid 0 by taking log
                dens = counts / (N * widths[0])
                dens = np.maximum(dens,1e-12) # floor
                # Find the bin for each sample
                bj = np.clip(np.searchsorted(edges,Xn[:,j],side='right')-1,0,n_bins-1)
                log_p += np.log(dens[bj])

            # weight: w ∝ target / pool
            log_w = log_t - log_p
            log_w -= log_w.max()
            w = np.exp(log_w)
            w_sum = w.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                w = np.exp(log_t - log_t.max())
                w_sum = w.sum()
            p = w / w_sum

            # the number of the sample
            target = min(len(data),int((n_bins ** d) * k_max))

            # Sampling without replacement by weight yields an approximate arcsine distribution
            rng = np.random.default_rng(0)
            idx = rng.choice(len(data),size=target,replace=False,p=p)
            result = data.iloc[idx].reset_index(drop=True)

            """
            n_target = int(len(data) / 2)
            eps = 1e-12
            w = 1.0 / np.sqrt(1 - X_norm.pow(2) + eps)
            w = w.prod(axis=1)

            result = data.sample(n=n_target,weights=w,random_state=27).reset_index(drop=True)
            """
        return result
    

class distribution_test:
    def __init__(self,sample):
        self.sample = sample
        self.X = self.sample.to_numpy()

    def _auto_bins_1d(self, N, min_exp=5, k_max=200):
        """
        Taking into account the constraints of resolution and expected frequency
        N/k >= min_exp
        """
        k1 = max(8, int(np.sqrt(max(N,1))))       
        k2 = max(2, int(N // max(min_exp,1)))     # Ensure Expctation≥min_exp
        k = min(k1, k2, k_max)
        return max(k, 2)

    def _auto_bins_joint(self, N, d, min_exp=5, k_max=10):
        """
        The number of equal-width bins per dimension for the n-dimensional joint test (k^d in total)
        Constraint: N / k^d >= min_exp
        N: the number of the sample
        d: dimension
        """
        if N <= 0:
            return 2
        k = int(np.floor((N / float(min_exp)) ** (1.0 / d)))
        k = max(2, min(k, k_max))
        return k

    def chi2_uniform_marginals(self, bounds=None, k=None, min_exp=5):
        """
        Perform chi-square test for each dimension: H0 = The dimension is uniformly distributed on [L,U]
        Parameter
        ----
        X : (N, d) ndarray
        bounds : None or [(L1,U1),...,(Ld,Ud)] 
                If None, use the extreme value of the sample in this dimension as the boundary
                (Will use ddof=2 to do parameter consumption correction)
        k : None or int, number of bins; None will be automatically selected
        min_exp : Lower bound of expected frequency per box (For automatic selection of k)
        Return
        ----
        results : list[dict], length equals to d (each dimension has a result)
        { 'dim': j, 'stat': χ2, 'pvalue': p, 'df': degree of freedom, 
            'k': k, 'expected_per_bin': E, 'edges': edges, 'observed': counts }
        """
        X = np.asarray(self.X)
        assert X.ndim == 2, "X must be (N, d)"
        N, d = X.shape
        results = []

        # Bounds and ddof (ddof=2 if a,b are estimated, 0 otherwise)
        if bounds is None:
            bounds = [(np.nanmin(X[:, j]), np.nanmax(X[:, j])) for j in range(d)]
            ddof_dim = 2
        else:
            assert len(bounds) == d
            ddof_dim = 0

        for j in range(d):
            xj = X[:, j]
            xj = xj[np.isfinite(xj)]
            Nj = len(xj)
            if Nj == 0:
                results.append({'dim': j, 'stat': np.nan, 'pvalue': np.nan, 'df': np.nan,
                                'k': None, 'expected_per_bin': None, 'edges': None, 'observed': None})
                continue

            Lj, Uj = bounds[j]
            if not np.isfinite(Lj) or not np.isfinite(Uj) or Uj <= Lj:
                raise ValueError(f"Bad bounds for dim {j}: {(Lj, Uj)}")

            kj = self._auto_bins_1d(Nj, min_exp) if k is None else int(k)
            edges = np.linspace(Lj, Uj, kj + 1)
            counts, _ = np.histogram(xj, bins=edges)

            # Expected frequency (equal width + uniform -> equal expectation for each box)
            Ej = Nj / kj
            f_exp = np.full(kj, Ej, dtype=float)

            # degree of random：k-1 - ddof
            df = kj - 1 - ddof_dim
            if df <= 0:
                # If there are too few bins and df<=0, force the number of bins to increase.
                kj = max(kj + (1 - df), 2)
                edges = np.linspace(Lj, Uj, kj + 1)
                counts, _ = np.histogram(xj, bins=edges)
                Ej = Nj / kj
                f_exp = np.full(kj, Ej, dtype=float)
                df = kj - 1 - ddof_dim

            chi2 = ((counts - Ej) ** 2 / np.where(Ej > 0, Ej, 1)).sum()
            # Calculate the p-value using scipy's formula (equivalent to chisquare(..., ddof=ddof_dim))
            from scipy.stats import chi2 as chi2_dist
            p = chi2_dist.sf(chi2, df)

            results.append({
                'dim': j,
                'stat': float(chi2),
                'pvalue': float(p),
                'df': int(df),
                'k': int(kj),
                'expected_per_bin': float(Ej),
                'edges': edges,
                'observed': counts
            })
        return results

    def chi2_uniform_joint(self, bounds=None, k=None, min_exp=5):
        """
        Chi-square test of joint distribution: H0 = samples are jointly uniform in the d-dimensional cuboid
        Parameter
        ----
        X : (N, d)
        bounds : None or [(L1,U1),...,(Ld,Ud)] 
                If None, use the extreme value of the sample in this dimension as the boundary
                (Will use ddof=2 to do parameter consumption correction)
        k : None or int, number of bins; None will be automatically selected
        min_exp : Lower bound of expected frequency per box (For automatic selection of k)
        Return
        ----
        result : dict
        { 'stat': χ2, 'pvalue': p, 'df': degree of freedom, 'k': k, 'edges': edges_list,
            'expected_per_cell': E_cell, 'observed_shape': counts.shape }
        """
        X = np.asarray(self.X)
        assert X.ndim == 2, "X must be (N, d)"
        N, d = X.shape

        if bounds is None:
            bounds = [(np.nanmin(X[:, j]), np.nanmax(X[:, j])) for j in range(d)]
            ddof = 2 * d
        else:
            assert len(bounds) == d
            ddof = 0

        if k is None:
            k = self._auto_bins_joint(N, d, min_exp=min_exp, k_max=12)

        # Generate equal-width boundaries in all dimensions
        edges_list = []
        for j in range(d):
            Lj, Uj = bounds[j]
            if not np.isfinite(Lj) or not np.isfinite(Uj) or Uj <= Lj:
                raise ValueError(f"Bad bounds for dim {j}: {(Lj, Uj)}")
            edges_list.append(np.linspace(Lj, Uj, int(k) + 1))

        counts, _ = np.histogramdd(X, bins=edges_list)
        obs = counts.ravel()
        cells = obs.size
        E_cell = N / float(cells)
        f_exp = np.full(cells, E_cell, dtype=float)

        # If df is too small/negative, relax: increase k or decrease ddof (increase k first)
        df = cells - 1 - ddof
        if df <= 0:
            raise ValueError(
                f"degree of freedom <=0, current k={k}, cells={cells}, ddof={ddof}.Please reduce ddof (provide known bounds) or increase .。"
            )

        # Calculate chi-square and p-value directly by definition
        # Equivalent to chisquare(obs, f_exp, ddof=ddof), but avoids numerical bounds issues)
        mask = f_exp > 0
        chi2 = ((obs[mask] - f_exp[mask]) ** 2 / f_exp[mask]).sum()

        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.sf(chi2, df)

        return {
            'stat': float(chi2),
            'pvalue': float(p),
            'df': int(df),
            'k': int(k),
            'edges': edges_list,
            'expected_per_cell': float(E_cell),
            'observed_shape': counts.shape
        }
    
    def ks_arcsine_marginals(self, bounds=None):
        """
        One-sample K–S test for arcsine (Beta(0.5,0.5)) marginals on each dimension.
        H0: X_j ~ arcsine on [L_j, U_j] (i.e., Beta(0.5,0.5) after linear mapping).
        Parameters
        ----------
        bounds : None or list of (L, U)
            If None, use sample min/max for each dim (note: p-values are approximate because
            loc/scale are estimated). If provided, treated as known support.
        Returns
        -------
        results : list of dicts
            [{'dim': j, 'stat': D, 'pvalue': p, 'n': N_j, 'bounds': (L,U)}]
        """
        X = np.asarray(self.X)
        assert X.ndim == 2, "X must be (N, d)"
        N, d = X.shape
        results = []

        # determine bounds
        if bounds is None:
            bounds = [(np.nanmin(X[:, j]), np.nanmax(X[:, j])) for j in range(d)]
            estimated_bounds = True
        else:
            assert len(bounds) == d
            estimated_bounds = False

        for j in range(d):
            xj = X[:, j]
            xj = xj[np.isfinite(xj)]
            n_j = len(xj)
            if n_j == 0:
                results.append({'dim': j, 'stat': np.nan, 'pvalue': np.nan, 'n': 0, 'bounds': None})
                continue

            L, U = bounds[j]
            if not np.isfinite(L) or not np.isfinite(U) or U <= L:
                raise ValueError(f"Bad bounds for dim {j}: {(L,U)}")

            # Define CDF for arcsine on [L,U]: map to [0,1], then use Beta(0.5,0.5).cdf
            a = 0.5
            b = 0.5
            def arcsine_cdf(t):
                z = (t - L) / (U - L)
                return stats.beta.cdf(z, a, b)

            D, p = stats.kstest(xj, arcsine_cdf)
            results.append({'dim': j, 'stat': float(D), 'pvalue': float(p), 'n': int(n_j), 'bounds': (float(L), float(U)),
                            'note': 'bounds estimated' if estimated_bounds else 'bounds provided'})
        return results

    def energy_arcsine_joint(self, bounds=None, n_ref=None, n_perm=199, random_state=0, max_pool=4000):
        """
        Multivariate joint test via Energy Distance (two-sample test)
        H0: X ~ product of independent arcsine (Beta(0.5,0.5)) on each dimension within bounds.
        Procedure:
          - If bounds not given, use sample min/max per dim (p-value is approximate).
          - Simulate reference sample Y ~ independent arcsine on those bounds.
          - Compute energy statistic T.
          - Compute permutation p-value by shuffling labels (using precomputed distance matrix).
        Parameters
        ----------
        bounds : None or list of (L,U)
        n_ref : int or None
            Size of reference sample to simulate. Default = len(X) (possibly downsampled by max_pool).
        n_perm : int
            Number of permutations for p-value (e.g., 199 or 499).
        random_state : int
        max_pool : int
            To bound memory, we cap total pooled size n + m <= max_pool by random subsampling.
        Returns
        -------
        dict with keys:
          {'stat': T, 'pvalue': p, 'n': n_used, 'm': m_used, 'bounds': bounds_used, 'note': str}
        """
        rng = np.random.default_rng(random_state)
        X = np.asarray(self.X)
        assert X.ndim == 2, "X must be (N, d)"
        n, d = X.shape

        # bounds
        if bounds is None:
            bounds_used = [(np.nanmin(X[:, j]), np.nanmax(X[:, j])) for j in range(d)]
            note_bounds = 'bounds estimated from sample'
        else:
            assert len(bounds) == d, "bounds length must equal d"
            bounds_used = bounds
            note_bounds = 'bounds provided'

        # simulate reference Y from independent arcsine on bounds
        if n_ref is None:
            m = n
        else:
            m = int(n_ref)

        # generate Beta(0.5,0.5) on [0,1], then scale to [L,U]
        Z = rng.beta(0.5, 0.5, size=(m, d))
        Y = np.empty_like(Z)
        for j in range(d):
            L, U = bounds_used[j]
            if not np.isfinite(L) or not np.isfinite(U) or U <= L:
                raise ValueError(f"Bad bounds for dim {j}: {(L,U)}")
            Y[:, j] = L + (U - L) * Z[:, j]

        # Optionally subsample to keep total size manageable
        pool_limit = int(max_pool)
        if n + m > pool_limit:
            # proportionally downsample X and Y
            frac = pool_limit / float(n + m)
            n_new = max(100, int(np.round(n * frac)))
            m_new = max(100, int(np.round(m * frac)))
            idxX = rng.choice(n, size=n_new, replace=False)
            idxY = rng.choice(m, size=m_new, replace=False)
            X_use = X[idxX]
            Y_use = Y[idxY]
        else:
            X_use = X
            Y_use = Y
            n_new = n
            m_new = m

        # Build pooled matrix and precompute pairwise distances
        import numpy as _np
        from scipy.spatial.distance import pdist, squareform
        Zpool = _np.vstack([X_use, Y_use])
        # pdist returns condensed vector; squareform makes it full matrix
        D = squareform(pdist(Zpool, metric='euclidean'))  # shape (n_new+m_new, n_new+m_new)
        n_total = n_new + m_new

        # helper to compute energy statistic T given index set A (X) and its complement (Y)
        def energy_T(idxA):
            idxA = _np.array(idxA, dtype=int)
            maskA = _np.zeros(n_total, dtype=bool)
            maskA[idxA] = True
            idxB = _np.where(~maskA)[0]

            # sums
            D_AA = D[_np.ix_(idxA, idxA)]
            D_BB = D[_np.ix_(idxB, idxB)]
            D_AB = D[_np.ix_(idxA, idxB)]

            # sum of off-diagonal distances within A/B (i<j). Use upper triangle.
            sum_AA = _np.sum(_np.triu(D_AA, k=1))
            sum_BB = _np.sum(_np.triu(D_BB, k=1))
            sum_AB = _np.sum(D_AB)

            nA = len(idxA)
            nB = len(idxB)
            # empirical energy distance E_{n,m}
            # careful with normalization: sums over pairs
            term_AB = (2.0 / (nA * nB)) * sum_AB
            term_AA = (2.0 / (nA * (nA - 1))) * sum_AA if nA > 1 else 0.0
            term_BB = (2.0 / (nB * (nB - 1))) * sum_BB if nB > 1 else 0.0
            Enm = term_AB - term_AA - term_BB
            T = (nA * nB / (nA + nB)) * Enm
            return T

        # observed T (first n_new are X)
        idxA0 = _np.arange(n_new)
        T_obs = energy_T(idxA0)

        # permutation p-value
        Ts = _np.empty(int(n_perm), dtype=float)
        for r in range(int(n_perm)):
            perm = rng.permutation(n_total)
            idxA = perm[:n_new]
            Ts[r] = energy_T(idxA)
        pval = (1.0 + _np.sum(Ts >= T_obs)) / (1.0 + len(Ts))

        return {
            'stat': float(T_obs),
            'pvalue': float(pval),
            'n': int(n_new),
            'm': int(m_new),
            'bounds': [(float(L), float(U)) for (L,U) in bounds_used],
            'note': f"{note_bounds}; permutations={int(n_perm)}; pooled={n_new+m_new}/{n+m}"
        }


class Trend_IC_time:
    def __init__(self,data,num_list,distribution,sbml_path=None,start_list=None,end_list=None,time_interval=None,n_steps_list=None,
                 degree_list=None, comb_list=None, method=None, model_name=None, out_root=None, model_kwargs=None):
        self.data = data
        self.num_list = num_list
        self.distribution = distribution
        self.sbml_path = sbml_path
        self.start_list = start_list if start_list is not None else []
        self.end_list = end_list if end_list is not None else []
        self.time_interval = time_interval
        self.n_steps_list = n_steps_list if n_steps_list is not None else []
        self.degree_list = degree_list if degree_list is not None else []
        self.comb_list = comb_list if comb_list is not None else []
        self.method = method
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.out_root = out_root  # optional base directory for outputs when sbml_path is None

        #self.results = self._analysis_summary()
        self.results = self._analysis_summary_parallel()
        # Build wide tables for ill-posed counts per degree (2-comb and 3-comb)
        self.ill2_table, self.ill3_table = self._modify_pattern(degree_start=1, fill=None, keep_max_col=True)

    def _analysis_summary(self):
        results = []
        if self.out_root is not None:
            base_dir = Path(self.out_root)
        elif self.sbml_path is not None:
            base_dir = Path(self.sbml_path).parent
        else:
            base_dir = Path.cwd()

        if self.distribution == 'uniform':
            base_dir = base_dir / "IC_uniform"
        elif self.distribution == 'arcsine':
            base_dir = base_dir / "IC_arcsine"

        # helper to fetch start/end given index in n_steps_list
        def _get_start_end(idx):
            # start
            if len(self.start_list) == len(self.n_steps_list):
                start = self.start_list[idx]
            elif len(self.start_list) == 1:
                start = self.start_list[0]
            else:
                raise ValueError("start_list length must be 1 or match n_steps_list length")
            # end
            if len(self.end_list) == len(self.n_steps_list):
                end = self.end_list[idx]
            elif len(self.end_list) == 1:
                end = self.end_list[0]
            else:
                raise ValueError("end_list length must be 1 or match n_steps_list length")
            return start, end

        # iterate over all combinations of num and n_step
        for num in self.num_list:
            for idx, n_step in enumerate(self.n_steps_list):
                start, end = _get_start_end(idx)

                out_path = base_dir / f"IC_{self.distribution}_n{int(num)}_pts{int(n_step)}"
                out_path.mkdir(parents=True, exist_ok=True)

                Sample = sampling(self.data, int(num), distribution=self.distribution)
                if self.sbml_path is not None:
                    filepaths = Sample.fit_and_save(self.sbml_path, str(out_path), start, end, int(n_step))
                elif self.model_name is not None:
                    filepaths = Sample.fit_and_save_base(model_name=self.model_name,
                                                         out_path=str(out_path),
                                                         start=start, end=end, n_step=int(n_step),
                                                         **(self.model_kwargs or {}))
                else:
                    raise ValueError("Either sbml_path or model_name must be provided in Trend_IC_time.")

                subsample = Sample._subsample(filepaths)

                # Resolve time column robustly: prefer 'time' produced by simulate()
                available_cols = subsample.columns
                if 'time' in available_cols:
                    _time_cols = ['time']
                else:
                    if isinstance(Sample.time_col, (list, tuple)):
                        _time_cols = [c for c in Sample.time_col if c in available_cols]
                    else:
                        _time_cols = [Sample.time_col] if Sample.time_col in available_cols else []

                state_cols = list(_time_cols) + list(Sample.ID)
                subsample_states = subsample[state_cols]

                if self.distribution.lower() == 'uniform':
                    if self.method in ('Monomial', 'Legendre'):
                        summary_result = run_noise_free_analysis(subsample_states, self.degree_list, self.comb_list, self.method)
                    else:
                        raise ValueError("Cannot assign basis for this distribution, try another distribution")
                elif self.distribution.lower() == 'arcsine':
                    if self.method == 'Chebyshev':
                        summary_result = run_noise_free_analysis(subsample_states, self.degree_list, self.comb_list, self.method)
                    else:
                        raise ValueError("Cannot assign basis for this distribution, try another distribution")
                else:
                    raise ValueError(f"Unknown distribution: {self.distribution}")

                results.append({
                    'num': int(num),
                    'start': float(start),
                    'end': float(end),
                    'n_step': int(n_step),
                    'summary': summary_result,
                    'method': self.method
                })
        return results
    
    def _modify_pattern(self, degree_start: int = 1, fill=None, keep_max_col: bool = False):
        """
        Convert per-(num, n_step) summary DataFrames into wide tables.

        Returns
        -------
        ill2_table : pd.DataFrame
            Columns: ['num', 'n_step', 'deg1_ill2', 'deg2_ill2', ..., 'ill2_at_max_degree']
        ill3_table : pd.DataFrame or None
            Same shape for 3-combination column if present; otherwise None.
        """
        rows_2 = []
        rows_3 = []

        # Determine the global set of degrees across all summaries (>= degree_start)
        all_degrees = set()
        for res in self.results:
            df = res['summary']
            if isinstance(df, pd.DataFrame) and 'degree' in df.columns:
                degs = df.loc[df['degree'] >= degree_start, 'degree'].astype(int).tolist()
                all_degrees.update(degs)
        if not all_degrees:
            return pd.DataFrame(columns=['num', 'n_step']), None
        degrees_sorted = sorted(int(d) for d in all_degrees)

        for res in self.results:
            num = int(res['num'])
            n_step = int(res['n_step'])
            df = res['summary']
            if not isinstance(df, pd.DataFrame) or 'degree' not in df.columns:
                # Skip malformed entries
                continue

            # Safe accessors for the two target columns
            has_ill2 = ('# ill-posed 2comb' in df.columns)
            has_ill3 = ('# ill-posed 3comb' in df.columns)

            sub = df.loc[df['degree'] >= degree_start].copy()
            if sub.empty:
                # Construct empty rows with NaNs (or fill)
                base_row = {'num': num, 'n_step': n_step}
                for d in degrees_sorted:
                    base_row[f'deg{d}_ill2'] = fill
                if keep_max_col:
                    base_row['ill2_at_max_degree'] = fill
                rows_2.append(base_row)
                if has_ill3:
                    base_row3 = {'num': num, 'n_step': n_step}
                    for d in degrees_sorted:
                        base_row3[f'deg{d}_ill3'] = fill
                    if keep_max_col:
                        base_row3['ill3_at_max_degree'] = fill
                    rows_3.append(base_row3)
                continue

            sub['degree'] = sub['degree'].astype(int)
            s2 = sub.set_index('degree')[['# ill-posed 2comb']].iloc[:, 0] if has_ill2 else pd.Series(dtype=float)
            s3 = sub.set_index('degree')[['# ill-posed 3comb']].iloc[:, 0] if has_ill3 else pd.Series(dtype=float)

            # Row for ill-posed 2-comb
            row2 = {'num': num, 'n_step': n_step}
            for d in degrees_sorted:
                val = s2.get(d)
                row2[f'deg{d}_ill2'] = (float(val) if pd.notna(val) else fill)
            if keep_max_col:
                row2['ill2_at_max_degree'] = (float(s2.iloc[-1]) if len(s2) > 0 and pd.notna(s2.iloc[-1]) else fill)
            rows_2.append(row2)

            # Row for ill-posed 3-comb (optional)
            if has_ill3:
                row3 = {'num': num, 'n_step': n_step}
                for d in degrees_sorted:
                    val = s3.get(d)
                    row3[f'deg{d}_ill3'] = (float(val) if pd.notna(val) else fill)
                if keep_max_col:
                    row3['ill3_at_max_degree'] = (float(s3.iloc[-1]) if len(s3) > 0 and pd.notna(s3.iloc[-1]) else fill)
                rows_3.append(row3)

        ill2_table = pd.DataFrame(rows_2).sort_values(['num', 'n_step']).reset_index(drop=True)
        ill3_table = (pd.DataFrame(rows_3).sort_values(['num', 'n_step']).reset_index(drop=True)
                      if rows_3 else None)

        return ill2_table, ill3_table

    def plot_illposed_tables(self, which='both', *,
                             label_points: bool = True,
                             label_every: int = 1,
                             label_last_only: bool = False,
                             fmt: str = '{:.2f}',
                             show_value_guides: bool = False,
                             show_delta_t: bool = False):
        """
        Plot ill-posed counts versus n_step and num for both comb-number tables.
        In Jupyter, figures will be displayed (no saving).

        Parameters
        ----------
        which : {'both','ill2','ill3'}
            Which tables to plot. 'both' will attempt ill2 then ill3 if available.
        """
        import matplotlib.patheffects as pe
        def _degree_cols(tbl, comb_tag):
            if tbl is None or tbl.empty:
                return []
            cols = []
            for c in tbl.columns:
                if c.startswith('deg') and c.endswith(f'_{comb_tag}'):
                    cols.append(c)
            def _degnum(name):
                try:
                    return int(name.split('_')[0][3:])  # 'deg3_ill2' -> 3
                except Exception:
                    return 10**9
            cols.sort(key=_degnum)
            return cols

        def _subplot_grid(n):
            import math
            r = int(math.ceil(math.sqrt(max(1, n))))
            c = int(math.ceil(n / r))
            return r, c

        def _color_map_for_degrees(deg_cols):
            # Build a stable color mapping across subplots
            import itertools
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = list(prop_cycle.by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']))
            # repeat if not enough
            col_cycle = list(itertools.islice(itertools.cycle(colors), len(deg_cols)))
            return {dc: col_cycle[i] for i, dc in enumerate(deg_cols)}

        def _marker_map_for_degrees(deg_cols):
            import itertools
            markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '>', '<', 'H', '8', 'p']
            mk_cycle = list(itertools.islice(itertools.cycle(markers), len(deg_cols)))
            return {dc: mk_cycle[i] for i, dc in enumerate(deg_cols)}

        def _fmt_val(val):
            try:
                return fmt.format(float(val))
            except Exception:
                return str(val)

        def _ytick_fontprops(ax):
            # Try to mirror the ytick label font (size/family) for value labels on the axis
            try:
                ticks = ax.yaxis.get_ticklabels()
                for t in ticks:
                    if t.get_text() != '':
                        return t.get_fontproperties()
                if len(ticks) > 0:
                    return ticks[0].get_fontproperties()
            except Exception:
                pass
            # Fallback to axis default
            from matplotlib import font_manager as fm
            fp = fm.FontProperties()
            try:
                fs = plt.rcParams.get('ytick.labelsize', None)
                if fs is not None:
                    fp.set_size(fs)
            except Exception:
                pass
            return fp

        def _is_on_tick(ax, y):
            """Return True if y coincides with an existing y‑tick (within tolerance)."""
            try:
                yt = ax.get_yticks()
                if yt is None or len(yt) == 0:
                    return False
                y_min, y_max = ax.get_ylim()
                tol = 0.01 * (y_max - y_min)  # 1% of axis range
                return np.any(np.isclose(yt, float(y), rtol=0.0, atol=tol))
            except Exception:
                return False

        def _is_zero(y):
            """Return True if y is (numerically or formatted) zero based on current fmt."""
            try:
                yf = float(y)
            except Exception:
                return False
            # numeric near-zero
            if abs(yf) <= 1e-12:
                return True
            # formatted zero under the provided fmt string
            try:
                return float(fmt.format(yf)) == 0.0
            except Exception:
                return False

        # ΔT/time-interval mapping block removed

        def _plot_case_fix_num(tbl, comb_tag):
            if tbl is None or tbl.empty:
                return None
            deg_cols = _degree_cols(tbl, comb_tag)
            if not deg_cols:
                return None
            color_map = _color_map_for_degrees(deg_cols)
            marker_map = _marker_map_for_degrees(deg_cols)
            nums = sorted(tbl['num'].unique())
            r, c = _subplot_grid(len(nums))
            fig, axes = plt.subplots(r, c, figsize=(6.0*c, 4.5*r), squeeze=False, sharex=False, sharey=False)
            for i, num in enumerate(nums):
                ax = axes[i//c][i%c]
                sub = tbl[tbl['num'] == num].sort_values('n_step')
                x_raw = sub['n_step'].values
                x = x_raw - 1  # use n_steps - 1 on the x-axis
                for dc in deg_cols:
                    y = sub[dc].values
                    ax.plot(x, y, marker=marker_map[dc], label=dc.split('_')[0], color=color_map[dc])
                # annotate point values near markers (cleaner than y-axis labels)
                if label_points:
                    fp = _ytick_fontprops(ax)
                    for dc in deg_cols:
                        y = sub[dc].values
                        for j,(xi, yi) in enumerate(zip(x, y)):
                            if label_last_only and j != len(x)-1:
                                continue
                            if (not label_last_only) and (label_every > 1) and (j % int(label_every) != 0):
                                continue
                            if _is_zero(yi):
                                continue
                            ax.annotate(_fmt_val(yi), xy=(xi, yi), xytext=(4, 0), textcoords='offset points',
                                        ha='left', va='center', fontproperties=fp,
                                        color=ax.yaxis.label.get_color(),
                                        path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.85)])

                ax.set_title(f"num={num}")
                ax.set_xlabel('time_step (n_step-1)')
                ax.set_ylabel(f"# ill-posed ({comb_tag})")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            # hide unused axes
            total = r*c
            for j in range(len(nums), total):
                axes[j//c][j%c].axis('off')
            plt.tight_layout()
            plt.show()
            return fig

        def _plot_case_fix_nstep(tbl, comb_tag):
            if tbl is None or tbl.empty:
                return None
            deg_cols = _degree_cols(tbl, comb_tag)
            if not deg_cols:
                return None
            color_map = _color_map_for_degrees(deg_cols)
            marker_map = _marker_map_for_degrees(deg_cols)
            steps = sorted(tbl['n_step'].unique())
            r, c = _subplot_grid(len(steps))
            fig, axes = plt.subplots(r, c, figsize=(6.0*c, 4.5*r), squeeze=False, sharex=False, sharey=False)
            for i, ns in enumerate(steps):
                ax = axes[i//c][i%c]
                sub = tbl[tbl['n_step'] == ns].sort_values('num')
                x = sub['num'].values
                for dc in deg_cols:
                    y = sub[dc].values
                    ax.plot(x, y, marker=marker_map[dc], label=dc.split('_')[0], color=color_map[dc])
                if label_points:
                    fp = _ytick_fontprops(ax)
                    for dc in deg_cols:
                        y = sub[dc].values
                        for j,(xi, yi) in enumerate(zip(x, y)):
                            if label_last_only and j != len(x)-1:
                                continue
                            if (not label_last_only) and (label_every > 1) and (j % int(label_every) != 0):
                                continue
                            if _is_zero(yi):
                                continue
                            ax.annotate(_fmt_val(yi), xy=(xi, yi), xytext=(4, 0), textcoords='offset points',
                                        ha='left', va='center', fontproperties=fp,
                                        color=ax.yaxis.label.get_color(),
                                        path_effects=[pe.withStroke(linewidth=3, foreground='white', alpha=0.85)])

                ax.set_title(f"n_step={ns}")
                ax.set_xlabel('num (initial_conditions)')
                ax.set_ylabel(f"# ill-posed ({comb_tag})")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            # hide unused axes
            total = r*c
            for j in range(len(steps), total):
                axes[j//c][j%c].axis('off')
            plt.tight_layout()
            plt.show()
            return fig

        to_plot = []
        if which in ('both','ill2'):
            to_plot.append(('ill2', self.ill2_table))
        if which in ('both','ill3'):
            to_plot.append(('ill3', self.ill3_table))

        figs = {}
        for comb_tag, table in to_plot:
            figs[(comb_tag,'fix-num')] = _plot_case_fix_num(table, comb_tag)
            figs[(comb_tag,'fix-nstep')] = _plot_case_fix_nstep(table, comb_tag)
        return figs

    def _analysis_summary_parallel(self, n_jobs=-1, backend='loky'):
        if self.out_root is not None:
            base_dir = Path(self.out_root)
        elif self.sbml_path is not None:
            base_dir = Path(self.sbml_path).parent
        else:
            base_dir = Path.cwd()

        def _get_start_end(idx):
            if len(self.start_list) == len(self.n_steps_list):
                start = self.start_list[idx]
            elif len(self.start_list) == 1:
                start = self.start_list[0]
            else:
                raise ValueError("start_list length must be 1 or match n_steps_list length")
            if len(self.end_list) == len(self.n_steps_list):
                end = self.end_list[idx]
            elif len(self.end_list) == 1:
                end = self.end_list[0]
            else:
                raise ValueError("end_list length must be 1 or match n_steps_list length")
            return start, end

        tasks = []
        for num in self.num_list:
            for idx, n_step in enumerate(self.n_steps_list):
                start, end = _get_start_end(idx)
                out_path = base_dir / f"IC_{self.distribution}_n{int(num)}_pts{int(n_step)}"
                tasks.append((num, n_step, start, end, str(out_path)))

        # Parallel execution
        results = Parallel(n_jobs=n_jobs, backend=backend, prefer='processes')([
            delayed(_process_combo)(
                self.data,
                self.distribution,
                self.sbml_path,
                self.model_name,
                self.model_kwargs,
                num,
                n_step,
                start,
                end,
                self.degree_list,
                self.comb_list,
                self.method,
                out_dir_str,
            ) for (num, n_step, start, end, out_dir_str) in tasks
        ])

        self.results = results
        self.ill2_table, self.ill3_table = self._modify_pattern(degree_start=1, fill=None, keep_max_col=True)
        return self.results