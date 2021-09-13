import pypesto.visualize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from MARM.estimation import get_problem, get_result, get_model
from MARM.parameters import save_parameters
from MARM.paths import get_figure_dir
from MARM.visualize import (
    plot_parameter_correlations,
    plot_and_save_fig, SEABORNE_FIGWIDTH
)
from MARM.analysis import N_RUNS

model_name = sys.argv[1]
variant = sys.argv[2]
dataset = sys.argv[3]

problem = get_problem(model_name, variant, dataset, 0, 1)
model = get_model(model_name, variant, dataset)
result = get_result(model_name, variant, dataset)
result.optimize_result.sort()
result.optimize_result.list = result.optimize_result.list[:N_RUNS]

figdir = get_figure_dir(model_name, variant, dataset)
pypesto.visualize.waterfall(result, scale_y='lin')
plot_and_save_fig(figdir, 'waterfall.pdf')

pypesto.visualize.parameters(result)
plot_and_save_fig(figdir, 'parameters.pdf')

pypesto.visualize.optimizer_history(
    result, scale_y='lin'
)
plot_and_save_fig(figdir, 'optimizer_trace.pdf')

pypesto.visualize.optimizer_convergence(result)
plot_and_save_fig(figdir, 'optimizer_convergence.pdf')

parameter_df = save_parameters(result, model_name, variant, dataset)

x_names = [problem.x_names[ix] for ix in problem.x_free_indices]
idx = sorted(range(len(problem.ub)), key=lambda k: problem.ub[k])

try:
    fig = plt.figure(figsize=(SEABORNE_FIGWIDTH*4, SEABORNE_FIGWIDTH))
    sns.clustermap(
        parameter_df.loc[:N_RUNS].apply(np.log10)[x_names],
        z_score=1,
        xticklabels=True, yticklabels=False,
    )
    plot_and_save_fig(figdir, 'clustermap.pdf')
except:
    pass

fig = plt.figure(figsize=(SEABORNE_FIGWIDTH, SEABORNE_FIGWIDTH/2))
ax = sns.boxplot(
    data=parameter_df.apply(np.log10)[[
        x_names[ix] for ix in idx
    ]],
    color='gray'
)
ax.plot([problem.ub[ix] for ix in idx], linestyle='dotted', color='black')
ax.plot([problem.lb[ix] for ix in idx], linestyle='dotted', color='black')
plt.xticks(rotation=90)
plot_and_save_fig(figdir, 'boxplot.pdf')

log_parameter_df = parameter_df[x_names].apply(np.log10)

plot_parameter_correlations(result, fval_cutoff=np.inf, std_threshold=1e-1)
plot_and_save_fig(figdir, 'parameter_correlations.pdf')
