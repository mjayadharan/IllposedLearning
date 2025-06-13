import itertools
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import Definitions
import ordpy

def KDE_distribution(D,range_index,grid_points=1000):
    # D: Input functions value dataframe
    # std_index: Need to standardize the input dataframe ['True','None']
    # range_index: Need tackle extreme range differences with interquartile grids ['True','None']  
    length = D.shape[0]
    list = D.columns

    # Define the function of generating global grid
    def generate_grid():
        min_val = np.min(D.values) - 1e-5
        max_val = np.max(D.values) + 1e-5
        if range_index == 'True':     # Quantile grid
            D_all_data = D.values.flatten()
            quantiles = np.linspace(0,1,grid_points)
            grid = np.percentile(D_all_data,quantiles)
        else:
            grid = np.linspace(min_val,max_val,grid_points)
        return grid
    grid = generate_grid()

    KDE = pd.DataFrame()
    for col in list:
        fun_values = np.array(D[col])
        fun_values.reshape(-1,1)
        if length > 1:
            # Gaussian kernel
            kde = gaussian_kde(fun_values,bw_method='silverman')
            pdf = kde(grid)
            #kde = KernelDensity(kernel='epanechnikov',bandwidth='silverman').fit(fun_values)
            #pdf = kde.score_samples(grid)
        else:
            pdf = np.zeros_like(grid)
            if length == 1:
                # Individual data points are set as pulse functions
                idx = np.argmin(np.abs(grid - fun_values[0]))
                pdf[idx] = 1.0
        pdf /= np.trapz(pdf,grid)
        KDE[col] = pdf
    # Return a DataFrame with pdf value at each grid point for every candidate function
    return KDE,grid


def KLD(P,Q,grid,eps=1e-12): 
    # This function is used to define Kullback–Leibler (KL) divergence
    # P/Q: Input dataframe of two candidate functions
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()
    dx = np.diff(grid, append = grid[-1])
    D_KL = np.sum(P * np.log(P / Q) * dx)
    return float(D_KL)


def distance_JS(Distribution,grid,eps=1e-12,tiny=1e-12):
    # Jenson-Shannon distance is defined based on Jennson-Shannon divergence
    # Distribution: Input the distribution dataframe of all candidate functions
    cols = Distribution.columns
    JSD = pd.DataFrame(
        np.zeros((len(cols), len(cols))), index=cols, columns=cols, dtype=float
    )
    for P, Q in itertools.combinations_with_replacement(cols, 2):
        dis_P = Distribution[P].values + eps
        dis_Q = Distribution[Q].values + eps
        dis_P /= dis_P.sum()
        dis_Q /= dis_Q.sum()

        dis_M = 0.5 * (dis_P + dis_Q)

        js_div = 0.0 if P == Q else 0.5 * (KLD(dis_P, dis_M,grid) + KLD(dis_Q, dis_M,grid))
        js_div = np.nan_to_num(js_div, nan=0.0, neginf=0.0)   # Replace NaN and -inf with 0
        js_div = max(js_div,0.0)

        jsd = 0.0 if js_div < tiny else np.sqrt(js_div)
        JSD.at[P, Q] = JSD.at[Q, P] = jsd      # Symmetrical distance matrix
    # Return the distance matrix in the form of dataframe
    return JSD


def weight_calculation(Distribution,grid,eps=1e-12):
    # Distribution: Input the distribution dataframe of all candidate functions
    cols = Distribution.columns
    weights = pd.DataFrame(
        np.zeros((len(cols), len(cols))), index=cols, columns=cols, dtype=float
    )
    dx = np.diff(grid, append=grid[-1])
    for P,Q in itertools.combinations_with_replacement(cols, 2):
        dis_P = Distribution[P].values + eps
        dis_Q = Distribution[Q].values + eps
        dis_P /= dis_P.sum()
        dis_Q /= dis_Q.sum()

        entropy_P = -np.sum(dis_P * np.log(dis_P) * dx)
        entropy_Q = -np.sum(dis_Q * np.log(dis_Q) * dx)
        if P == Q:
            weights.at[P,Q] = weights.at[Q,P] = 0
        else:
            # A low entropy means less information contained, means one thing is more likely to be predicted
            # For two distributions P and Q, first normalize to ensure w_P + w_Q = 1,
            # then, if P's entropy value is smaller, we tend to conclude that a small weight should be given to P
            den = entropy_Q + entropy_Q
            if den < 1e-16:
                w = 0.5
            else:
                w = entropy_P / den
            # weight from P to Q, which is the weight correspong to the distribution P
            weights.at[P,Q] = w
            # weight from Q to P, which is the weight correspong to the distribution Q
            weights.at[Q,P] = 1 - w
    return weights



def distance_JS_weight(Distribution,weights,eps=1e-12,tiny=1e-12):
    # The difference of this function is about weights, we first use Shannon entropy to calculate the weights between two candidate functions and then define the
    # distance based on the original definition of Jenson-Shannon divergence.

    # Jenson-Shannon distance is defined based on Jennson-Shannon divergence
    # Distribution: Input the distribution dataframe of all candidate functions
    # weights: Input the weight dataframe
    cols = Distribution.columns
    JSD = pd.DataFrame(
        np.zeros((len(cols), len(cols))), index=cols, columns=cols, dtype=float
    )
    for P, Q in itertools.combinations_with_replacement(cols, 2):
        dis_P = Distribution[P].values + eps
        dis_Q = Distribution[Q].values + eps
        dis_P /= dis_P.sum()
        dis_Q /= dis_Q.sum()

        w_P = weights.at[P,Q]
        w_Q = weights.at[Q,P]

        if P == Q:
            js_div = 0.0
        else:
            js_div = w_P * np.sum(dis_P * np.log(dis_P / (w_P * dis_P + w_Q * dis_Q))) + w_Q * np.sum(dis_Q * np.log(dis_Q / (w_P * dis_P + w_Q * dis_Q)))
        js_div = np.nan_to_num(js_div, nan=0.0, neginf=0.0)   # Replace NaN and -inf with 0
        js_div = max(js_div,0.0)

        jsd = 0.0 if js_div < tiny else np.sqrt(js_div)
        JSD.at[P, Q] = JSD.at[Q, P] = jsd      # Symmetrical distance matrix
    # Return the distance matrix in the form of dataframe
    return JSD


def PJSD(D,dx,taux,tiny=1e-12):
    # D: Input functions values dataframe
    # std_index: Need to standardize the input dataframe ['True','None']
    # dx: Embedding dimension (horizontal axis) (default: 3)
    # taux: Embedding delay (horizontal axis)(default: 1)
    cols = D.columns
    PJSD = pd.DataFrame(
        np.zeros((len(cols), len(cols))), index=cols, columns=cols, dtype=float
    )
    for P,Q in itertools.combinations_with_replacement(cols,2):
        data = D[[P,Q]].values.T
        pjsd = ordpy.permutation_js_distance(data,dx=dx,dy=1,taux=taux,tauy=1,base='e',normalized=None,
                                                 tie_precision=None)
            # Here, we just need to care about dx and taux since we are defining distance between two time series
            # For practical purposes, value range of dx is suugested to be [3,7] and taux is suggested to be 1
        pjsd = 0.0 if pjsd < tiny else pjsd
        PJSD.at[P,Q] = PJSD.at[Q,P] = pjsd
    return PJSD