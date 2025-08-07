import pandas as pd
import numpy as np
import tellurium as te
import os

from Comparison import standardize_columns
from Basis import normalization
    

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
        # Initial conditions in the form of num * n_features
        random_initial_original = np.random.uniform(low=L_orig, high=U_orig, size=(self.num,self.n_features))
        random_initial_normalized = np.random.uniform(low=L_norm, high=U_norm, size=(self.num,self.n_features))
        return pd.DataFrame(random_initial_original,columns=self.data_states.columns), pd.DataFrame(random_initial_normalized,columns=self.data_states.columns)
    
    def _arcsine_sample(self):
        L_orig, U_orig = self.data_states.min(), self.data_states.max()
        L_norm, U_norm = self.data_norm.min(), self.data_norm.max()
        L_orig_array, U_orig_array = L_orig.values, U_orig.values
        L_norm_array, u_norm_array = L_norm.values, U_norm.values
        # Initial conditions in the form of num * n_features
        u = np.random.uniform(low=L_norm_array,high=u_norm_array,size=(self.num,self.n_features))
        x = np.cos(np.pi * u)
        arcsine_X_original = 0.5 * (x + 1) * (U_orig_array - L_orig_array) + L_orig_array
        return pd.DataFrame(arcsine_X_original,columns=self.data_states.columns), pd.DataFrame(x,columns=self.data_states.columns)


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
            # Assign every point to a grid cell
            idx = ((X_norm + 1) * (n_bins / 2)).astype(int).clip(0, n_bins - 1)
            idx_cols = [f"b{i}" for i in range(len(self.ID))]
            bins = pd.DataFrame(idx.values, columns=idx_cols)

            # Stratified sampling: keep ≤ k_max rows per cell
            chosen = []
            for _, g in data.join(bins).groupby(idx_cols, sort=False):
                chosen.append(g.sample(min(k_max, len(g)), random_state=0))
            result = pd.concat(chosen, ignore_index=True)
        elif self.distribution.lower() == 'arcsine':
            n_target = int(len(data) / 2)
            eps = 1e-12
            w = 1.0 / np.sqrt(1 - X_norm.pow(2) + eps)
            w = w.prod(axis=1)

            result = data.sample(n=n_target,weights=w,random_state=27).reset_index(drop=True)
        return result