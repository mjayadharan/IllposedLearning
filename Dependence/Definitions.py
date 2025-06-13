import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import warnings

def DataStandard(D):
    # D: Input functions value dataframe
    scaler = StandardScaler()
    arr = scaler.fit(D).transform(D)
    scaled_data = pd.DataFrame(arr,columns=D.columns,index=D.index)
    # Return a dataframe
    return scaled_data


def preprocess_for_stable(data, method='None'):
    """
    Preprocess dataframe for numerically stable SVD analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with potentially small values
    method : str
        Preprocessing method: 'standardize', 'normalize', 'center', or 'scale'
    
    Returns:
    --------
    pandas.DataFrame, dict
        Preprocessed dataframe and preprocessing info
    """
    
    preprocessing_info = {
        'original_stats': {
            'min': data.min().min(),
            'max': data.max().max(),
            'mean': data.mean().mean(),
            'std': data.std().mean()
        },
        'method': method
    }
    
    if method == 'standardize':
        # Z-score standardization: (X - mean) / std
        processed_data = (data - data.mean()) / data.std()
        preprocessing_info['means'] = data.mean()
        preprocessing_info['stds'] = data.std()
        
    elif method == 'normalize':
        # Min-max normalization to [0, 1]
        processed_data = (data - data.min()) / (data.max() - data.min())
        preprocessing_info['mins'] = data.min()
        preprocessing_info['ranges'] = data.max() - data.min()
        
    elif method == 'center':
        # Mean centering only
        processed_data = data - data.mean()
        preprocessing_info['means'] = data.mean()
        
    elif method == 'scale':
        # Scale by standard deviation only
        processed_data = data / data.std()
        preprocessing_info['stds'] = data.std()
        
    else:
        processed_data = data.copy()
    
    # Handle potential NaN/inf values
    processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
    if processed_data.isnull().any().any():
        warnings.warn("NaN values detected after preprocessing. Consider different preprocessing.")
    
    preprocessing_info['processed_stats'] = {
        'min': processed_data.min().min(),
        'max': processed_data.max().max(),
        'mean': processed_data.mean().mean(),
        'std': processed_data.std().mean()
    }
    
    return processed_data, preprocessing_info


def verify_triangle(Distance):
    # Distance: Input distance dataframe of candidate functions
    mx = 0.0
    bad = None
    list = Distance.columns
    for a,b,c in itertools.combinations(list,3):
        dist_ab = Distance.at[a,b]
        dist_ac = Distance.at[a,c]
        dist_bc = Distance.at[b,c]

        tol = 1e-9
        cond1 = (dist_ab <= dist_ac + dist_bc + tol)
        cond2 = (dist_ac <= dist_ab + dist_bc + tol)
        cond3 = (dist_bc <= dist_ab + dist_ac + tol)

        if not (cond1 and cond2 and cond3):
            return False
    return True

def verify_relative(Distance):
    # split terms like x1x2 and x2^2 into {"x1","x2"} and {"x2"}
    def parse_vars(name):
        comps = name.split()
        return {comp.split('^')[0] for comp in comps}
    
    cols = Distance.columns
    var_sets = {c: parse_vars(c) for c in cols}
    violations = []
    for a,b,c in itertools.combinations(cols,3):
        vsa, vsb, vsc = var_sets[a], var_sets[b], var_sets[c]

        # Denote strict subsets by < and normal subsets by <=.
        if vsb < vsa and not (vsb <= vsc):
            dab = Distance.at[a, b]
            dbc = Distance.at[b, c]
            if not (dab < dbc):
                violations.append((a, b, c, dab, dbc))
    
    return len(violations) == 0
    #return len(violations) == 0, len(violations), violations

def verify_bound(Distance,b):
    threshold = np.sqrt(math.log(2,b))
    mask = Distance <= threshold
    all_ok = mask.values.all()
    return all_ok

def distance_Pearson(D):
    # Distance is defined based on Pearson correlation coefficient
    Distance = 1 - D.copy().abs()  # As a small Pearson coefficient refers to less relevant and it ranges from [-1,1]
    # Return the distance matrix in the form of dataframe
    return Distance

