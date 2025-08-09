import pandas as pd
import numpy as np
import tellurium as te
import os
from scipy.stats import qmc
from scipy.stats import chisquare
from scipy.spatial.distance import pdist, squareform
from scipy import stats

from Comparison import standardize_columns
from Basis import normalization

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


class sampling():
    """
    This class is used for sampling under different initial conditions and finally deriving the time series 
    distributed as a specific distribution.
    """
    def __init__(self,data,num,distribution='uniform'):
        self.data = data
        self.num = num
        self.distribution = distribution

        Creator = create_initial_conditions(self.data,self.num,self.distribution)
        self.sample_orig = Creator.sample_orig
        self.data_states = Creator.data_states
        self.state_cols = self.data_states.columns.tolist()
    
    def fit_and_save(self,sbml_path:str,out_path:str,start,end,n_points,
                     filename_prefix="IC"):
        os.makedirs(out_path,exist_ok=True)
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
            res = rr.simulate(start, end, n_points)
            df = pd.DataFrame(res, columns=rr.timeCourseSelections)

            rates_list = []
            for t in range(len(df)):
                for j, s in enumerate(species_ids):
                    rr[s] = df.loc[t, s]
                rates = rr.getRatesOfChange()
                rates_list.append(rates)

            derivative_df = pd.DataFrame(rates_list, columns=[f'd{s}/dt' for s in species_ids])
            data_augmented = pd.concat([df, derivative_df], axis=1)

            fp = os.path.join(out_path, f"{filename_prefix}_{k:03d}.xlsx")
            data_augmented.to_excel(fp, index=False)
            filepaths.append(fp)

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
    

class distribution_test():
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

