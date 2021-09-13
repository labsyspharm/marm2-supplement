import sys
import os
import logging
import fides

import pypesto.optimize
import pickle

from MARM.estimation import get_problem
from MARM.paths import get_results_path, get_traces_path

model_name = sys.argv[1]
variant = sys.argv[2]
dataset = sys.argv[3]
index = sys.argv[4]
n_starts = int(sys.argv[5])
n_threads = int(sys.argv[6])

dataset = '_'.join(sorted(dataset.split('_')))

optim_options = {
    fides.Options.XATOL: 1e-12,
    fides.Options.GATOL: 1e-4,
}

optimizer = pypesto.optimize.FidesOptimizer(
    options=optim_options,
    verbose=logging.INFO
)

problem = get_problem(model_name, variant, dataset, n_threads)

optimize_options = pypesto.optimize.OptimizeOptions(
    startpoint_resample=True,
    allow_failed_starts=True,
)

history_options = pypesto.HistoryOptions(
    trace_record=True,
    trace_record_hess=False,
    trace_record_res=False,
    trace_record_sres=False,
    trace_record_schi2=False,
    storage_file=os.path.join(
        get_traces_path(model_name, variant),
        f'{dataset}_{index}_{{id}}.csv',
    ),
    trace_save_iter=1
)
result = pypesto.optimize.minimize(
    problem,
    optimizer,
    n_starts=n_starts,
    startpoint_method=pypesto.startpoint.uniform,
    options=optimize_options,
    history_options=history_options
)

outdir = os.path.join(get_results_path(model_name, variant))
logdir = os.path.join('logs', 'cluster')

if not os.path.exists(outdir):
    os.makedirs(outdir)

outfile = os.path.join(outdir, f'{dataset}-{index}.pickle')
with open(outfile, 'wb') as f:
    pickle.dump(result.optimize_result, f)
