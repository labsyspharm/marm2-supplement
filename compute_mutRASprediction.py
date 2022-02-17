import sys
import os

import numpy as np
import pandas as pd

from MARM.analysis import (
    read_settings, load_parameters_as_dataframe, run_and_store_simulation,
    get_analysis_results_file
)
from MARM.estimation import get_problem

import pypesto
import fides
import logging

sxs = read_settings(sys.argv)
cell_line = sys.argv[6]


optim_options = {
    fides.Options.XTOL: 1e-12,
    fides.Options.GATOL: 1e-4,
}

optimizer = pypesto.optimize.FidesOptimizer(
    options=optim_options,
    verbose=logging.INFO
)

df_parameters = load_parameters_as_dataframe(sxs['model_name'],
                                             sxs['variant'],
                                             sxs['dataset'])

problem = get_problem(sxs['model_name'], 'nrasq61mut',
                      f'MEKi_PRAFi_RAFi_{cell_line}_mutrastraining',
                      sxs['threads'])

free_pars = ['q61_RAS_gtp_kcat']

fixed_pars = [
    problem.x_names.index(c)
    for c in df_parameters.columns
    if c not in ['chi2'] + free_pars
    and c in problem.x_names
]
fixed_vals = [
    df_parameters.loc[sxs['index'], problem.x_names[ip]]
    if problem.x_scales[ip] == 'lin'
    else np.log10(df_parameters.loc[sxs['index'], problem.x_names[ip]])
    for ip in fixed_pars
]

print(dict(zip(fixed_pars, fixed_vals)))

problem.fix_parameters(fixed_pars, fixed_vals)

problem.lb_full[problem.x_names.index('q61_RAS_gtp_kcat')] = -1
problem.lb_init_full[problem.x_names.index('q61_RAS_gtp_kcat')] = -1
problem.ub_full[problem.x_names.index('q61_RAS_gtp_kcat')] = 4
problem.ub_init_full[problem.x_names.index('q61_RAS_gtp_kcat')] = 4
problem.normalize()

optimize_options = pypesto.optimize.optimize.OptimizeOptions(
    startpoint_resample=True,
    allow_failed_starts=True,
)
result = pypesto.optimize.minimize(
    problem,
    optimizer,
    n_starts=5,
    startpoint_method=pypesto.startpoint.uniform,
    options=optimize_options,
)

par_dict = {}
for name, value, scale in zip(problem.x_names,
                              result.optimize_result.list[0]['x'],
                              problem.x_scales):
    if scale == 'log10':
        par_dict[name] = np.power(10, value)
    else:
        par_dict[name] = value

for col in df_parameters.columns:
    if col not in par_dict and col != 'chi2':
        par_dict[col] = df_parameters.loc[sxs['index'], col]

df_par = pd.DataFrame(par_dict, index=[sxs['index']])

file = get_analysis_results_file(sxs['model_name'], sxs['variant'],
                                 sxs['dataset'], f'mutRASpars_{cell_line}',
                                 sxs['index'])
os.makedirs(os.path.dirname(file), exist_ok=True)
df_par.to_csv(file)

run_and_store_simulation(sxs, f'mutRASprediction_{cell_line}',
                         par_dict=par_dict)

run_and_store_simulation(sxs, f'mutRASprediction_{cell_line}_combo',
                         par_dict=par_dict)
