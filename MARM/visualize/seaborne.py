import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr

from .common import new_rc_params

mpl.rcParams.update(new_rc_params)

SEABORNE_FIGWIDTH = 15


def corrcoef(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    r = np.ones(shape=(rows, rows))
    p = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i+1, rows):
            r_, p_ = pearsonr(matrix[i], matrix[j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


def get_parameter_array(result, fval_cutoff):
    if fval_cutoff is None:
        fval_cutoff = 1e-1

    return np.vstack([
        np.array(res['x']) for res in result.optimize_result.list if
        res.fval < result.optimize_result.list[0].fval + fval_cutoff
    ])


def plot_parameter_correlations(result, fval_cutoff=None, std_threshold=1e-1):
    pars = get_parameter_array(result, fval_cutoff)

    par_std = pars.std(axis=0)

    corr, pval = corrcoef(pars.T)
    corr[pval > 0.05/(np.power(corr.shape[0],2)-corr.shape[0])/2] = 0.0
    idx = np.where((par_std > std_threshold) & (np.sum(np.abs(corr) > 0, axis=1) > 0))[0]
    corr = corr[np.ix_(idx, idx)]
    names = [result.problem.x_names[ix] for ix in idx]
    par_corr = pd.DataFrame(corr, columns=names, index=names)

    plt.figure()
    sns.clustermap(par_corr, cmap='bwr', vmin=-1, vmax=1,
                   figsize=(SEABORNE_FIGWIDTH, SEABORNE_FIGWIDTH),
                   yticklabels=True, xticklabels=True)


def get_consistent_parameter_estimates(result, fval_cutoff=None,
                                      std_threshold=1e-1):
    pars = get_parameter_array(result, fval_cutoff)

    par_std = pars.std(axis=0)
    par_mean = pars.mean(axis=0)
    par_name = result.problem.x_names
    df_pars = pd.DataFrame(
        {'name': par_name, 'mean': par_mean, 'std': par_std})
    return df_pars[df_pars['std'] < std_threshold]


def plot_parameter_scatterplot(result, parameter_names, fval_cutoff=None):
    par_df = pd.DataFrame(
        get_parameter_array(result, fval_cutoff),
        columns=result.problem.x_names
    )
    sns.pairplot(par_df[parameter_names])
