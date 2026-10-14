import pandas as pd
import numpy as np
import tellurium as te
import os
from scipy.stats import qmc, truncnorm, chisquare
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy.signal import savgol_filter
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pysindy as ps

from Comparison import standardize_columns, run_noise_free_analysis
from Basis import normalization, normalization_to_01
from Base_test import Lotka_Volterra, CRN


# Helper for parallel analysis
def _process_combo(data, distribution, sbml_path, model_name, model_kwargs,
                   num, n_step, start, end, degree_list, comb_list, method, out_dir_str,
                   pointwise=False):
    """Worker: run one (num, n_step) combo and return summary dict."""
    # Ensure directory exists in worker
    os.makedirs(out_dir_str, exist_ok=True)

    # Build sampler and simulate or evaluate pointwise
    Sample = sampling(data, int(num), distribution=distribution)
    if pointwise:
        # Route A: evaluate RHS at joint-uniform points without integrating trajectories
        if sbml_path is not None:
            filepaths = Sample.evaluate_and_save_points_sbml(sbml_path, out_dir_str,
                                                             filename_prefix="PTS")
        elif model_name is not None:
            mk = model_kwargs or {}
            filepaths = Sample.evaluate_and_save_points_base(model_name=model_name,
                                                             out_path=out_dir_str,
                                                             filename_prefix="PTS",
                                                             **mk)
        else:
            raise ValueError("Either sbml_path or model_name must be provided to evaluate RHS at points.")
        subsample = pd.read_excel(filepaths[0])  # one file of pointwise evaluations
    else:
        # Original: simulate trajectories, then subsample
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
    elif dist_lower == 'gaussian':
        if method == 'Hermite':
            summary_result = run_noise_free_analysis(subsample_states, degree_list, comb_list, method)
        else:
            raise ValueError("Cannot assign basis for this distribution, expected method='Hermite'.")
    elif dist_lower == 'exponential':
        if method == 'Laguerre':
            summary_result = run_noise_free_analysis(subsample_states, degree_list, comb_list, method)
        else:
            raise ValueError("Cannot assign basis for this distribution, expected method='Laguerre'.")
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

def fps_indices(X, M, seed=0):
    """
    Farthest Point Sampling on rows of X (numpy array).
    - Avoids selecting the same index twice.
    - Breaks early if all remaining distances are -inf (unique points exhausted).
    - Adds tiny jitter to break ties.
    Returns indices into X (not deduplicated).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if N == 0:
        return np.array([], dtype=int)
    M = int(min(max(1, M), N))
    chosen = np.empty(M, dtype=int)
    # pick a random start
    chosen[0] = rng.integers(N)
    # distances to the nearest chosen point
    dist = np.linalg.norm(X - X[chosen[0]], axis=1)
    taken = np.zeros(N, dtype=bool)
    taken[chosen[0]] = True
    dist[taken] = -np.inf
    for i in range(1, M):
        # tiny jitter to break ties in a reproducible way
        j = int(np.argmax(dist + 1e-12 * rng.random(N)))
        if not np.isfinite(dist[j]):
            chosen = chosen[:i]
            break
        chosen[i] = j
        taken[j] = True
        # update min-distance to the set
        dnew = np.linalg.norm(X - X[j], axis=1)
        dist = np.minimum(dist, dnew)
        dist[taken] = -np.inf
    return chosen

# Helper: safely widen degenerate bounds for uniform/arcsine sampling
def _widen_bounds(L, U, min_span=1e-6, frac=0.05):
    """
    If a column has U - L ≈ 0 (degenerate in the source data), expand it slightly so
    that sampling does not collapse to a constant. The expansion is centered at the
    observed value and uses the larger of:
      - an absolute floor `min_span`
      - a fraction `frac` of a scale (|center| or 1.0 if near zero).
    L, U: pandas Series (one per column)
    Returns widened (Lw, Uw) as pandas Series with the same index.
    """
    import numpy as _np
    L_arr = _np.asarray(L, dtype=float).copy()
    U_arr = _np.asarray(U, dtype=float).copy()
    for i, (l, u) in enumerate(zip(L_arr, U_arr)):
        # If not finite or non-increasing, or span too small, widen
        bad = (not _np.isfinite(l)) or (not _np.isfinite(u)) or (u - l) <= 0
        span = u - l if _np.isfinite(u - l) else 0.0
        if bad or span < max(min_span, frac * max(1.0, abs(0.5 * (l + u)))):
            center = 0.5 * (l + u) if _np.isfinite(l) and _np.isfinite(u) else (l if _np.isfinite(l) else (u if _np.isfinite(u) else 0.0))
            target_span = max(min_span, frac * max(1.0, abs(center)))
            L_arr[i] = center - 0.5 * target_span
            U_arr[i] = center + 0.5 * target_span
    return pd.Series(L_arr, index=L.index), pd.Series(U_arr, index=U.index)


class create_initial_conditions:
    """
    This class creates a group of initial conditions.
    These conditions are uniformly/arcsine distributed in the original state space
    """
    def __init__(self, data, num, distribution='uniform', seed=None):
        # data: original dataframe including time and states columns, they are not necessary standardized or normalized.
        # num: the number of initial conditions in the group
        # distribution: expected distribution including arcsine and uniform, the defeault setting is uniform
        self.data = data
        self.num = num
        self.distribution = distribution
        self.seed = seed

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
        elif self.distribution.lower() == 'exponential':
            self.sample_orig, self.sample_norm = self._exponential_sample()
        elif self.distribution.lower() == 'gaussian':
            self.sample_orig, self.sample_norm = self._gaussian_sample()

    # uniformly_sample is the way used afterwards
    def _uniformly_sample(self):   
            self.L_orig, self.U_orig = self.data_states.min(), self.data_states.max()
            cols = self.data_states.columns.tolist()
            d = len(cols)

            # Sobol + random shift in [0,1]^d
            u01 = sobol_u01(self.num, d=d, seed=0)

            # Map to original ranges (widened if necessary)
            Lo, Uo = self.L_orig.values, self.U_orig.values
            #Lo, Uo = L_wide.values, U_wide.values
            X_orig = Lo + u01 * (Uo - Lo)

            # Also return a normalized representation spanning [-1, 1] regardless of data ranges
            X_norm = 2.0 * u01 - 1.0
            X_norm = np.clip(X_norm, -1 + 1e-7, 1 - 1e-7)

            return pd.DataFrame(X_orig, columns=cols), pd.DataFrame(X_norm, columns=cols)
    
    def _arcsine_sample(self):
        self.L_orig, self.U_orig = self.data_states.min(), self.data_states.max()
        cols = self.data_states.columns.tolist()
        d = len(cols)

        # Sobol + random shift in [0,1]^d
        u01 = sobol_u01(self.num, d=d, seed=0)

        # arcsine transformation in [-1, 1]
        x = np.cos(np.pi * u01)
        x = np.clip(x, -1 + 1e-7, 1 - 1e-7)

        # Safely widen degenerate columns for mapping back to original scale
        Lw, Uw = _widen_bounds(self.L_orig, self.U_orig, min_span=1e-6, frac=0.05)
        Lo, Uo = Lw.values.astype(float), Uw.values.astype(float)

        # Map arcsine sample x ∈ [-1,1] to original ranges
        X_orig = 0.5 * (x + 1.0) * (Uo - Lo) + Lo
        return pd.DataFrame(X_orig, columns=cols), pd.DataFrame(x, columns=cols)
    
    def _exponential_sample(self):
        # 1) Original per-state hard bounds (NO widening)
        self.L_orig, self.U_orig = self.data_states.min(), self.data_states.max()
        cols = self.data_states.columns.tolist()
        d = len(cols)

        Lo = self.L_orig.values.astype(float)  # lower bounds from data
        Uo = self.U_orig.values.astype(float)  # upper bounds from data

        # 2) Compute per-dimension acceptance probabilities under Exp(1) on [0,+∞)
        #    Effective lower bound for Exp support is max(Lo, 0).
        L_eff = np.maximum(Lo, 0.0)
        U_eff = Uo.copy()

        # Feasibility check: if U_eff <= L_eff in any dim, probability is zero -> impossible.
        bad_dims = np.where(~np.isfinite(L_eff) | ~np.isfinite(U_eff) | (U_eff <= L_eff))[0]
        if len(bad_dims) > 0:
            bad_names = [cols[i] for i in bad_dims]
            raise ValueError(
                "Exponential sampling impossible for dimensions with U <= max(L,0): "
                f"{bad_names}. Please ensure U_i > max(L_i, 0) for all states."
            )

        # Per-dimension acceptance prob p_j = P(L_eff <= X <= U_eff) with X~Exp(1).
        # If U is infinite, p_j = exp(-L_eff).
        p = np.empty(d, dtype=float)
        for j in range(d):
            if np.isfinite(U_eff[j]):
                p[j] = np.exp(-L_eff[j]) - np.exp(-U_eff[j])
            else:
                p[j] = np.exp(-L_eff[j])
        P_all = float(np.prod(p))
        if not np.isfinite(P_all) or P_all <= 0.0:
            raise ValueError("Overall acceptance probability is zero/invalid; cannot sample under the given bounds.")

        # 3) Decide how many total candidates to draw (safety margin to counter variance)
        rng = np.random.default_rng(self.seed)
        safety = 1.25  # 25% headroom
        n_draw = int(np.ceil(self.num / P_all * safety))
        n_draw = max(n_draw, max(1000, 10 * d))  # ensure a reasonable batch size

        accepted = []
        total_generated = 0
        max_rounds = 20

        # 4) Rejection sampling loop with adaptive batch sizing
        while sum(len(a) for a in accepted) < self.num and max_rounds > 0:
            X_cand = rng.exponential(scale=1.0, size=(n_draw, d))  # Exp(1) i.i.d.
            total_generated += n_draw

            # Accept only if each coordinate lies within [Lo, Uo]
            mask = np.ones(n_draw, dtype=bool)
            for j in range(d):
                # Note: if Lo[j] < 0, Exp(1) can't produce negatives, so (X>=Lo) is automatically satisfied.
                mask &= (X_cand[:, j] >= Lo[j]) & (X_cand[:, j] <= Uo[j])
            if np.any(mask):
                accepted.append(X_cand[mask])

            remaining = self.num - sum(len(a) for a in accepted)
            if remaining <= 0:
                break

            # Empirical acceptance rate to refine next batch size
            acc_count = sum(len(a) for a in accepted)
            est_rate = max(1e-12, acc_count / float(total_generated))
            n_draw = int(np.ceil(remaining / est_rate * safety))
            n_draw = max(n_draw, max(1000, 10 * d))
            max_rounds -= 1

        # Final escalation if still short (rare unless P_all is tiny)
        if sum(len(a) for a in accepted) < self.num:
            n_draw = int(np.ceil(self.num / P_all * (safety * 4.0)))
            X_cand = rng.exponential(scale=1.0, size=(n_draw, d))
            mask = np.ones(n_draw, dtype=bool)
            for j in range(d):
                mask &= (X_cand[:, j] >= Lo[j]) & (X_cand[:, j] <= Uo[j])
            if np.any(mask):
                accepted.append(X_cand[mask])

        X = np.vstack(accepted) if accepted else np.empty((0, d))
        if X.shape[0] < self.num:
            raise RuntimeError(
                f"Failed to collect {self.num} accepted Exp(1) samples; only {X.shape[0]} obtained after "
                f"{total_generated} draws. Check bounds: some acceptance probabilities may be extremely small."
            )
        X = X[:self.num]

        # 5) Return also the [-1,1] normalized copy using the SAME (Lo,Uo)
        denom = np.maximum(Uo - Lo, 1e-12)
        X_norm = 2.0 * (X - Lo) / denom - 1.0
        X_norm = np.clip(X_norm, -1 + 1e-7, 1 - 1e-7)

        return pd.DataFrame(X, columns=cols), pd.DataFrame(X_norm, columns=cols)

    def _gaussian_sample(self, sigma_z: float = np.sqrt(0.5)):
        self.L_orig, self.U_orig = self.data_states.min(), self.data_states.max()
        cols = self.data_states.columns.tolist()
        d = len(cols)

        # 1) Sobol + gaussian inverse CDF
        u01 = sobol_u01(self.num, d=d, seed=0)
        u01 = np.clip(u01, 1e-12, 1 - 1e-12)
        z_std = stats.norm.ppf(u01)           # N(0,1)
        z = z_std * float(sigma_z)            # N(0, sigma_z^2)，sigma_z = sqrt(1/2)

        # 2) Align ±3*sigma_z to [L,U] and clip back to the interval.
        Lw, Uw = _widen_bounds(self.L_orig, self.U_orig, min_span=1e-6, frac=0.05)
        Lo, Uo = Lw.values.astype(float), Uw.values.astype(float)
        scale = (Uo - Lo) / (6.0 * float(sigma_z) + 1e-12)
        center = 0.5 * (Lo + Uo)
        X = center + z * scale
        X = np.clip(X, Lo, Uo)

        # 3) [-1,1] Linear scaling (aligned to *the same* Lo, Uo)
        X_norm = 2.0 * (X - Lo) / np.maximum(Uo - Lo, 1e-12) - 1.0
        X_norm = np.clip(X_norm, -1 + 1e-7, 1 - 1e-7)
        return pd.DataFrame(X, columns=cols), pd.DataFrame(X_norm, columns=cols)


class sampling:
    """
    This class is used for sampling under different initial conditions and finally deriving the time series 
    distributed as a specific distribution.
    """
    def __init__(self, data, num, distribution='uniform', noise_level=None):
        # data: original dataframe including time and states columns, 
        # they are not necessary standardized or normalized.
        # num: how many initial conditions are expected to be generated
        # n_step: how many steps are expected to simulated
        self.data = data
        self.num = num
        self.distribution = distribution
        self.noise_level = noise_level # This noise is added to the data after sampling

        Creator = create_initial_conditions(self.data, self.num, self.distribution)
        #self.sample_orig = Creator.sample_orig
        self.L_orig, self.U_orig = Creator.L_orig, Creator.U_orig
        self.sample_orig = Creator.sample_orig
        self.sample_norm = Creator.sample_norm
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
        # Use floating species IDs as the dynamic states (strings only)
        species_ids = [str(s) for s in rr.model.getFloatingSpeciesIds()]
        self.ID = species_ids

        # Build IC table aligned *positionally* to the species order to avoid label mismatches
        if len(self.sample_orig.columns) != len(species_ids):
            raise ValueError(
                f"Column mismatch: {len(self.sample_orig.columns)} columns vs {len(species_ids)} species (IDs: {species_ids})."
                " Make sure your input data has the same number of state columns as floating species."
            )
        ic_df = self.sample_orig.copy()
        # Ensure exact order and names match species_ids
        ic_df.columns = species_ids

        # Selections: time plus species states (use concentration syntax [S])
        species_selections = [f'[{s}]' for s in species_ids]
        rr.timeCourseSelections = ['time'] + species_selections

        # Load parameters dictionary
        param_dict = {}
        for param in rr.model.getGlobalParameterIds():
            val = rr.model[param]
            param_dict[param] = val

        # Prepare CVODE integrator (robust defaults for stiff systems)
        rr.setIntegrator('cvode')
        cvode = rr.getIntegrator()

        filepaths = []
        # ensure integer number of points and include both endpoints (step=5 in the Beer model)
        for k in range(len(ic_df)):
            # Initialize by position to avoid label KeyError; enforce non-negativity for species
            vals = np.array(ic_df.iloc[k].to_numpy(), dtype=float)
            vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), 0.0)
            for s, v in zip(species_ids, vals):
                rr[f'init({s})'] = float(v)

            rr.reset()

            # Configure integrator options *after* reset (some options may reset with the model)
            try:
                cvode.setValue('stiff', True)  # use BDF/Newton for stiff problems
            except Exception:
                pass
            try:
                cvode.setValue('relative_tolerance', 1e-6)
                cvode.setValue('absolute_tolerance', 1e-10)
            except Exception:
                pass
            try:
                cvode.setValue('maximum_num_steps', int(1e6))
                cvode.setValue('variable_step_size', True)
            except Exception:
                pass
            # Limit maximum step size to the output spacing (helps convergence)
            try:
                dt = (float(end) - float(start)) / max(int(n_step) - 1, 1)
                if np.isfinite(dt) and dt > 0:
                    cvode.setValue('maximum_step_size', float(dt))
            except Exception:
                pass

            # Try to simulate; on failure, relax tolerances and shrink max step
            try:
                res = rr.simulate(start, end, n_step)
            except RuntimeError:
                try:
                    cvode.setValue('relative_tolerance', 1e-5)
                    cvode.setValue('absolute_tolerance', 1e-8)
                    if 'dt' in locals() and np.isfinite(dt) and dt > 0:
                        cvode.setValue('maximum_step_size', float(dt) / 10.0)
                except Exception:
                    pass
                res = rr.simulate(start, end, n_step)

            df = pd.DataFrame(res, columns=rr.timeCourseSelections)

            # Rename [S] -> S for consistency
            rename_map = {f'[{s}]': s for s in species_ids}
            df = df.rename(columns=rename_map)
        
            # Sanity check: ensure all species columns are present
            missing_cols = [s for s in species_ids if s not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"simulate() did not return requested species columns: {missing_cols}. "
                    f"Selections were: {rr.timeCourseSelections}"
                )

            # Only save the state trajectories (no derivatives here)
            fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
            df.to_excel(fp, index=False)
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
              - method: Monomial, Legendre and Chebyshev. This input determines
                whether the sampled data need to be normalized or not.
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
            # model_state_names = model_kwargs.get('state_names', ["x","y"])  # order matters
            model_state_names = model_kwargs.get('state_names', ["x", "y1","y2"])  # order matters
            # Build a consistent mapping between model names and standardized names, in order
            model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
            std_in_model_order = [model_to_std[m] for m in model_state_names]  # e.g. ['x1','x2','x3']

            params = model_kwargs.get('params')
            method = model_kwargs.get('method')
            if params is None:
                raise ValueError("'params' must be provided for Lotka-Volterra in model_kwargs.")
            noise_level = model_kwargs.get('noise_level', None)

            # Compute derivatives with central finite difference and use function in pysindy
            cfd = ps.FiniteDifference(order=2)

            for k in range(len(self.sample_orig)):
                row = self.sample_orig.iloc[k]
                # Initial condition in the model's variable order (x,y)
                z0= [float(row[s]) for s in std_in_model_order]

                # Simulate via the Base_test class
                lv = Lotka_Volterra(params=params, t_span=t_span, t_eval=t_eval, z0=z0, noise_level=noise_level)
                df_raw = lv.data_sim.copy()  # columns: ['time','x','y1','y2']

                # Rename model outputs to standardized names for consistency
                rename_map = {m: model_to_std[m] for m in model_state_names}
                rename_map.update({'time': 'time'})
                df = df_raw.rename(columns=rename_map)

                state_cols = [c for c in df.columns if c != 'time']
                t = df['time'].to_numpy()
                L_sample, U_sample = self.L_orig, self.U_orig

                if method in ('Laguerre','Monomial'):
                    X = df[state_cols].to_numpy()
                    Xdot = cfd(X,t)

                    for j,s in enumerate(state_cols):
                        col = f'd{s}/dt'
                        df[col] = Xdot[:,j]                   

                    # Only save the state trajectories 
                    fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                    df.to_excel(fp, index=False)
                elif method in ('Legendre','Chebyshev','Hermite'):
                    X = df[state_cols].to_numpy()
                    df_states_norm = normalization(df[state_cols], L_sample, U_sample)[0]
                    X_norm = df_states_norm.to_numpy()
                    df_norm = pd.concat([df[['time']], df_states_norm], axis=1)
                    Xdot = cfd(X_norm,t)

                    for j,s in enumerate(state_cols):
                        col = f'd{s}/dt'
                        df_norm[col] = Xdot[:,j]

                    # Only save the state trajectories 
                    fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                    df_norm.to_excel(fp, index=False)
                else:
                    raise ValueError(f"Unsupported basis library: {method}")
                filepaths.append(fp)
            
        elif model_name.lower() == 'crn':
            model_state_names = model_kwargs.get('state_names', ["S","E","ES","P"])
            model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
            std_in_model_order = [model_to_std[m] for m in model_state_names]

            k_rates = model_kwargs.get('k_rates')
            method = model_kwargs.get('method')
            if k_rates is None:
                raise ValueError("'k_rates' must be provided for Chemical Reaction Network in model_kwargs.")
            noise_level = model_kwargs.get('noise_level', None) # This noise is added to generate a noisy or not original dataset
            # Compute derivatives with central finite difference and use function in pysindy
            cfd = ps.FiniteDifference(order=2)

            for k in range(len(self.sample_orig)):
                row = self.sample_orig.iloc[k]
                init_cond = [float(row[s]) for s in std_in_model_order]

                # Simulate via the Base_test class
                crn = CRN(k_rates=k_rates, init_cond=init_cond, solvedT=t_eval, noise_level=noise_level)
                df_raw = crn.data_sim.copy()

                # Rename model outputs to standardized names for consistency
                rename_map = {m: model_to_std[m] for m in model_state_names}
                rename_map.update({'time': 'time'})
                df = df_raw.rename(columns=rename_map)

                state_cols = [c for c in df.columns if c != 'time']

                # Use global normalization bounds for all trajectories (avoid mapping endpoints to exactly ±1)
                L_sample, U_sample = self.L_orig, self.U_orig
                
                t = df['time'].to_numpy() 
                df_deriv = df.copy()

                if method in ('Laguerre','Monomial'):
                    X = df_deriv[state_cols].to_numpy()
                    Xdot = cfd(X,t)

                    for j,s in enumerate(state_cols):
                        col = f'd{s}/dt'
                        df[col] = Xdot[:,j]                   

                    # Only save the state trajectories 
                    fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                    df.to_excel(fp, index=False)
                #else:
                elif method in ('Legendre','Chebyshev','Hermite'):
                    X = df[state_cols].to_numpy()
                    df_states_norm = normalization(df[state_cols], L_sample, U_sample)[0]
                    X_norm = df_states_norm.to_numpy()
                    df_norm = pd.concat([df[['time']], df_states_norm], axis=1)
                    Xdot = cfd(X_norm,t)

                    for j,s in enumerate(state_cols):
                        col = f'd{s}/dt'
                        df_norm[col] = Xdot[:,j]

                    # Only save the state trajectories 
                    fp = os.path.join(out_dir, f"{filename_prefix}_{tag}_{k:03d}.xlsx")
                    df_norm.to_excel(fp, index=False)
                #else:
                    #raise ValueError(f"Unsupported basis library: {method}")
                filepaths.append(fp)
 
        return filepaths

    def _subsample(
        self,
        filepaths,
        L=None, U=None,
        n_bins: int = 10,
        k_max: int = 1,
    ):
        """
        Parameters
        ----------
        filepaths : list[str]
            Paths returned by ``fit_and_save`` (each is an Excel file that holds one
            trajectory with states).
        n_bins : int, default 10
            Number of equal‑width bins per dimension (stratified grid).
        k_max : int, default 1
            Maximum number of rows kept per grid cell.

        Returns
        -------
        pd.DataFrame
            Subsampled dataframe containing only time and state columns.
            # NOTE: To compute time derivatives, call `compute_derivatives` on the returned DataFrame.
        """
        # Load and concatenate every trajectory dataframe
        dfs = [pd.read_excel(fp) for fp in filepaths]
        print("Subsample input columns:", dfs[0].columns.tolist())
        data = pd.concat(dfs, ignore_index=True)

        # Drop global start/end time rows to reduce mass at normalization endpoints (±1)
        if 'time' in data.columns:
            tmin = data['time'].min()
            tmax = data['time'].max()
            mask_mid = (data['time'] != tmin) & (data['time'] != tmax)
            if mask_mid.any():
                data = data.loc[mask_mid].reset_index(drop=True)

        X = data[self.ID]
        # Default bounds if not provided
        if L is None or U is None:
            L = X.min()
            U = X.max()
        L, U = _widen_bounds(L, U, min_span=1e-8, frac=0.01)
        X_norm = 2 * (X - L) / (U - L) - 1

        if self.distribution.lower() == 'uniform':
            d = len(self.ID)
            Xn = X.to_numpy()
            # Deduplicate (within rounding tolerance) to avoid repeated identical rows in the output
            Xn_round = np.round(Xn, 8)
            uniq, uniq_idx = np.unique(Xn_round, axis=0, return_index=True)
            n_unique = len(uniq_idx)
            target = min(n_unique, int((n_bins ** d) * k_max), len(data))
            # Run FPS on the unique representatives, then map back
            idx_u = fps_indices(uniq, M=target, seed=0)
            idx = np.asarray(uniq_idx)[idx_u]
            result = data.iloc[idx].reset_index(drop=True)
        elif self.distribution.lower() == 'arcsine':
            eps = 1e-7
            # Clip the edge
            Xn = X
            N, d = Xn.shape

            # density log t(x) = - 0.5 * sum_j log(1-x_j^2)
            log_t = -0.5 * np.log(1.0 - Xn**2).sum(axis=1)

            edges = np.linspace(-1, 1, n_bins+1)
            widths = np.diff(edges)
            log_p = np.zeros(N)
            for j in range(d):
                counts, _ = np.histogram(Xn.to_numpy()[:, j], bins=edges)
                dens = counts / (N * widths[0])
                dens = np.maximum(dens, 1e-12) # floor
                bj = np.clip(np.searchsorted(edges, Xn.to_numpy()[:, j], side='right') - 1, 0, n_bins - 1)
                log_p += np.log(dens[bj])

            log_w = log_t - log_p
            log_w -= log_w.max()
            w = np.exp(log_w)
            w_sum = w.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                w = np.exp(log_t - log_t.max())
                w_sum = w.sum()
            p = w / w_sum

            target = min(len(data), int((n_bins ** d) * k_max))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(data), size=target, replace=False, p=p)
            result = data.iloc[idx].reset_index(drop=True)
        elif self.distribution.lower() in ('gaussian'):
            Ls, Us = pd.Series(L), pd.Series(U)
            Lw, Uw = _widen_bounds(Ls, Us, min_span=1e-6, frac=0.05)
            L_arr, U_arr = Lw.values.astype(float), Uw.values.astype(float)
            sigma = np.sqrt(0.5)
            scale = (U_arr - L_arr) / (6.0 * sigma + 1e-12)
            center = 0.5 * (L_arr + U_arr)
            Z = (X.to_numpy() - center) / scale
            # Target log-density under product exp(-z^2)
            log_t = -np.sum(Z**2, axis=1)
            # Empirical marginal densities via histograms on Z within ±3.5σ
            z_max = 3.5
            edges_list = [np.linspace(-z_max, z_max, int(n_bins)+1) for _ in range(Z.shape[1])]
            counts, _ = np.histogramdd(Z, bins=edges_list)
            widths = [np.diff(e)[0] for e in edges_list]
            dens_per_dim = []
            D = counts.ndim  # equals Z.shape[1]
            for j in range(Z.shape[1]):
                axes_to_sum = tuple(ax for ax in range(D) if ax != j)
                counts_j = counts.sum(axis=axes_to_sum) if len(axes_to_sum) > 0 else counts
                # ensure 1-D marginal of length n_bins
                counts_j = np.asarray(counts_j, dtype=float).reshape(-1)
                dens_j = counts_j / (len(Z) * widths[j])
                dens_j = np.maximum(dens_j, 1e-12)
                bj = np.clip(np.searchsorted(edges_list[j], Z[:, j], side='right') - 1, 0, int(n_bins) - 1)
                dens_per_dim.append(dens_j[bj])
            log_p = np.log(np.column_stack(dens_per_dim)).sum(axis=1)
            log_w = log_t - log_p
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            w_sum = w.sum()
            if not np.isfinite(w_sum) or w_sum <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w_sum
            d = len(self.ID)
            target = min(len(data), int((n_bins ** d) * k_max))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(data), size=target, replace=False, p=w)
            result = data.iloc[idx].reset_index(drop=True)
        elif self.distribution.lower() in ('exponential'):
            # Importance resampling towards product Exp(1) shape per dimension.
            # Map X to canonical z via scale s such that 99% quantile aligns with U (same as IC sampling).
            Ls, Us = pd.Series(L), pd.Series(U)
            Lw, Uw = _widen_bounds(Ls, Us, min_span=1e-6, frac=0.05)
            Lo, Uo = Lw.values.astype(float), Uw.values.astype(float)
            L_trunc = np.maximum(Lo, 0.0)
            q = -np.log(1.0 - 0.99)  # ln(100)
            s = (Uo - L_trunc) / (q + 1e-12)
            s = np.where(s > 0, s, 1.0)
            Z = (np.clip(X.to_numpy(), L_trunc, Uo) - L_trunc) / np.maximum(s, 1e-12)
            # Target log-density for product of Exp(1): -sum(Z)
            log_t = -np.sum(Z, axis=1)
            # Empirical marginal densities via histograms on Z over [0, q]
            edges_list = [np.linspace(0.0, float(q), int(n_bins)+1) for _ in range(Z.shape[1])]
            counts, _ = np.histogramdd(Z, bins=edges_list)
            widths = [np.diff(e)[0] for e in edges_list]
            dens_per_dim = []
            D = counts.ndim
            for j in range(Z.shape[1]):
                axes_to_sum = tuple(ax for ax in range(D) if ax != j)
                counts_j = counts.sum(axis=axes_to_sum) if len(axes_to_sum) > 0 else counts
                counts_j = np.asarray(counts_j, dtype=float).reshape(-1)
                dens_j = counts_j / (len(Z) * widths[j])
                dens_j = np.maximum(dens_j, 1e-12)
                bj = np.clip(np.searchsorted(edges_list[j], Z[:, j], side='right') - 1, 0, int(n_bins) - 1)
                dens_per_dim.append(dens_j[bj])
            log_p = np.log(np.column_stack(dens_per_dim)).sum(axis=1)
            log_w = log_t - log_p
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            w_sum = w.sum()
            if not np.isfinite(w_sum) or w_sum <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w_sum
            d = len(self.ID)
            target = min(len(data), int((n_bins ** d) * k_max))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(data), size=target, replace=False, p=w)
            result = data.iloc[idx].reset_index(drop=True)
        return result

    def compute_derivatives(self, subsample: pd.DataFrame, model=None, model_type=None, model_kwargs=None, sbml_path=None):
        """
        Compute time derivatives for a given subsample DataFrame (after selection).
        This method augments the DataFrame with derivative columns.
        For SBML models, provide sbml_path; for base models, provide model_name and model_kwargs.
        Returns a new DataFrame with derivative columns appended.
        """
        # NOTE: This must be called after subsampling, i.e., after _subsample.
        import copy
        df = subsample.copy()
        # Try to resolve state columns and time column
        time_col = 'time' if 'time' in df.columns else None
        state_cols = [c for c in self.ID if c in df.columns]
        if sbml_path is not None:
            # SBML model: use tellurium
            rr = te.loadSBMLModel(sbml_path)
            species_ids = [str(s) for s in rr.model.getFloatingSpeciesIds()]
            # For each row, set the state and compute rates
            rates_list = []
            for t in range(len(df)):
                state_vals = df.loc[t, species_ids].to_numpy()
                for s, v in zip(species_ids, state_vals):
                    rr[s] = float(v)
                rates = rr.getRatesOfChange()
                rates_list.append(rates)
            derivative_df = pd.DataFrame(rates_list, columns=[f'd{s}/dt' for s in species_ids])
            df = pd.concat([df.reset_index(drop=True), derivative_df], axis=1)
            return df
        elif model_type is not None:
            # Base model: use provided model_type and kwargs
            std_names = state_cols
            if model_type.lower() == 'lotka-volterra':
                model_state_names = model_kwargs.get('state_names', ["x", "y1","y2"])
                if len(model_state_names) != 3:
                    raise ValueError("Lotka-Volterra expects exactly 3 model state names.")
                model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
                params = model_kwargs.get('params')
                lv = Lotka_Volterra(params=params, t_span=(0.0,1.0), t_eval=np.array([0.0,1.0]), z0=[0.0,0.0,0.0], noise_level=None)

                deriv = []
                for i in range(len(df)):
                    t_val = float(df.loc[i, time_col]) if time_col else 0.0
                    state_vec_model_order = [float(df.loc[i, model_to_std[m]]) for m in model_state_names]
                    dstate = lv.lv_rhs_1_2(t_val, state_vec_model_order)
                    deriv.append(dstate)
                deriv_df = pd.DataFrame(deriv, columns=[f"d{model_to_std[m]}/dt" for m in model_state_names])
                df = pd.concat([df.reset_index(drop=True), deriv_df], axis=1)
                return df
            elif model_type.lower() == 'crn':
                model_state_names = model_kwargs.get('state_names', ["S","E","ES","P"])
                model_to_std = {m: std_names[i] for i, m in enumerate(model_state_names)}
                k_rates = model_kwargs.get('k_rates')
                crn = CRN(k_rates=k_rates, init_cond=[0.0]*4, solvedT=np.array([0.0,1.0]), noise_level=None)

                deriv = []
                for i in range(len(df)):
                    t_val = float(df.loc[i, time_col]) if time_col else 0.0
                    state_vec_model_order = [float(df.loc[i, model_to_std[m]]) for m in model_state_names]
                    dstate = crn.toyEnzRHS(state_vec_model_order, t_val)
                    deriv.append(dstate)
                deriv_df = pd.DataFrame(deriv, columns=[f"d{model_to_std[m]}/dt" for m in model_state_names])
                df = pd.concat([df.reset_index(drop=True), deriv_df], axis=1)
                return df
            else:
                raise ValueError(f"Unsupported base model for derivative computation: {model_type}")
        else:
            raise ValueError("Either sbml_path or model_type must be provided to compute derivatives.")

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
    
    def ks_exponential_marginals(self, bounds=None, rate: float = 1.0):
        """
        One-sample K–S test for exponential marginals on each dimension.
        This matches the Laguerre basis weight w(x)=exp(-x) on [0,+∞) (i.e., Exp(rate) with rate=1 by default).
        We test against a *truncated* exponential on [a_eff, b] where:
           a_eff = max(a, 0) to respect the exponential support; if bounds is None we use sample min/max.
        The test uses the exact truncated CDF:
           F_trunc(x) = (F(x) - F(a_eff)) / (F(b) - F(a_eff)),
        where F(x) = 1 - exp(-rate * x) and F(b)=1 if b is infinite.

        Returns
        -------
        results : list of dicts per dimension:
            [{'dim': j, 'stat': D, 'pvalue': p, 'n': N_j, 'bounds': (a_eff, b), 'rate': rate}]
        """
        X = np.asarray(self.X)
        assert X.ndim == 2, "X must be (N, d)"
        N, d = X.shape
        results = []

        # bounds: if None, estimate from data
        if bounds is None:
            bounds = [(np.nanmin(X[:, j]), np.nanmax(X[:, j])) for j in range(d)]
            estimated = True
        else:
            assert len(bounds) == d
            estimated = False

        for j in range(d):
            xj = X[:, j]
            xj = xj[np.isfinite(xj)]
            n_j = len(xj)
            if n_j == 0:
                results.append({'dim': j, 'stat': np.nan, 'pvalue': np.nan, 'n': 0, 'bounds': None, 'rate': rate})
                continue

            a, b = bounds[j]
            if not np.isfinite(a): a = 0.0
            a_eff = max(float(a), 0.0)  # enforce exponential support
            b_eff = float(b)
            if not np.isfinite(b_eff):
                Fb = 1.0
            else:
                if b_eff <= a_eff:
                    raise ValueError(f"Bad bounds for dim {j}: U <= max(L,0): {(a, b)}")
                Fb = 1.0 - np.exp(-rate * b_eff)

            Fa = 1.0 - np.exp(-rate * a_eff)
            denom = max(Fb - Fa, 1e-15)

            def cdf_trunc(t):
                t = np.asarray(t, dtype=float)
                # below lower support -> 0; above upper bound -> 1
                if np.isscalar(t):
                    if t <= a_eff: return 0.0
                    if np.isfinite(b_eff) and t >= b_eff: return 1.0
                Ft = 1.0 - np.exp(-rate * t)
                return (Ft - Fa) / denom

            D, p = stats.kstest(xj, cdf_trunc)
            results.append({
                'dim': j,
                'stat': float(D),
                'pvalue': float(p),
                'n': int(n_j),
                'bounds': (float(a_eff), float(b_eff)),
                'rate': float(rate),
                'note': 'bounds estimated' if estimated else 'bounds provided'
            })
        return results

    def energy_exponential_joint(self, bounds=None, rate: float = 1.0, n_ref=None, n_perm=199, random_state=0, max_pool=4000):
        """
        Multivariate joint test via Energy Distance for exponential target (product of independent exponentials).
        Target per-dimension distribution: truncated Exp(rate) on [a_eff, b], where a_eff=max(a,0) and
        b is the provided upper bound (can be +∞). This aligns with the Laguerre basis weight exp(-x).

        Procedure:
          - If bounds is None, use sample min/max per dim (p-values approximate).
          - Simulate reference Y from the product of (truncated) Exp(rate) with inverse-CDF sampling.
          - Compute energy statistic T and a permutation p-value.

        Returns a dict:
          {'stat': T, 'pvalue': p, 'n': n_used, 'm': m_used, 'bounds': bounds_used, 'rate': rate, 'note': str}
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

        # helper: sample from truncated Exp(rate) on [a_eff, b]
        def sample_trunc_exp(size):
            Z = np.empty((size, d), dtype=float)
            for j in range(d):
                a, b = bounds_used[j]
                a_eff = max(float(a), 0.0)
                Fa = 1.0 - np.exp(-rate * a_eff)
                if np.isfinite(b):
                    Fb = 1.0 - np.exp(-rate * float(b))
                else:
                    Fb = 1.0
                u = rng.random(size)
                # target cdf in [Fa, Fb]: u' = Fa + u*(Fb-Fa)
                u_prime = Fa + u * (Fb - Fa)
                # inverse CDF for Exp(rate): x = -ln(1 - u')/rate
                Z[:, j] = -np.log(np.maximum(1.0 - u_prime, 1e-15)) / rate
            return Z

        # reference sample size
        m = int(n if n_ref is None else n_ref)
        Y = sample_trunc_exp(m)

        # Optionally subsample pooled size
        pool_limit = int(max_pool)
        if n + m > pool_limit:
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

        # pairwise distances
        Zpool = np.vstack([X_use, Y_use])
        Dmat = squareform(pdist(Zpool, metric='euclidean'))
        n_total = n_new + m_new

        def energy_T(idxA):
            idxA = np.asarray(idxA, dtype=int)
            maskA = np.zeros(n_total, dtype=bool)
            maskA[idxA] = True
            idxB = np.where(~maskA)[0]
            DA = Dmat[np.ix_(idxA, idxA)]
            DB = Dmat[np.ix_(idxB, idxB)]
            DAB = Dmat[np.ix_(idxA, idxB)]
            sum_AA = np.sum(np.triu(DA, k=1))
            sum_BB = np.sum(np.triu(DB, k=1))
            sum_AB = np.sum(DAB)
            nA, nB = len(idxA), len(idxB)
            term_AB = (2.0 / (nA * nB)) * sum_AB
            term_AA = (2.0 / (nA * (nA - 1))) * sum_AA if nA > 1 else 0.0
            term_BB = (2.0 / (nB * (nB - 1))) * sum_BB if nB > 1 else 0.0
            Enm = term_AB - term_AA - term_BB
            return (nA * nB / (nA + nB)) * Enm

        idxA0 = np.arange(n_new)
        T_obs = energy_T(idxA0)
        Ts = np.empty(int(n_perm), dtype=float)
        for r in range(int(n_perm)):
            perm = rng.permutation(n_total)
            idxA = perm[:n_new]
            Ts[r] = energy_T(idxA)
        pval = (1.0 + np.sum(Ts >= T_obs)) / (1.0 + len(Ts))

        return {
            'stat': float(T_obs),
            'pvalue': float(pval),
            'n': int(n_new),
            'm': int(m_new),
            'bounds': [(float(max(a,0.0)), float(b)) for (a,b) in bounds_used],
            'rate': float(rate),
            'note': f"{note_bounds}; permutations={int(n_perm)}; pooled={n_new+m_new}/{n+m}"
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