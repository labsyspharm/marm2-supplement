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
x_names = sorted(x_names, key=lambda x: x.split('_')[-1])

parameter_df_log = parameter_df.copy()
for x_name in x_names:
    if x_name.endswith(('_phi', '_dG', '_ddG')):
        continue
    parameter_df_log[x_name] = np.log10(parameter_df[x_name])

try:
    fig = plt.figure(figsize=(SEABORNE_FIGWIDTH*4, SEABORNE_FIGWIDTH))
    sns.clustermap(
        parameter_df_log.loc[:N_RUNS, x_names],
        z_score=1,
        xticklabels=True, yticklabels=False,
    )
    plot_and_save_fig(figdir, 'clustermap.pdf')
except:
    pass

fig = plt.figure(figsize=(SEABORNE_FIGWIDTH, SEABORNE_FIGWIDTH/2))
ax = sns.boxplot(
    data=parameter_df_log.loc[:, x_names],
    color='gray'
)
ax.plot(problem.ub, linestyle='dotted', color='black')
ax.plot(problem.lb, linestyle='dotted', color='black')
plt.xticks(rotation=90)
plot_and_save_fig(figdir, 'boxplot.pdf')

plot_parameter_correlations(result, fval_cutoff=np.inf, std_threshold=1e-1)
plot_and_save_fig(figdir, 'parameter_correlations.pdf')
