import sys
import os
import time
import pypesto.startpoint
import pandas as pd
import random

from MARM.estimation import get_problem
from MARM.paths import get_profile_dir, get_multimodel_speedup_result_file

model_name = sys.argv[1]
variant = sys.argv[2]
dataset = sys.argv[3]
index = sys.argv[4]

dataset = '_'.join(sorted(dataset.split('_')))


def time_objective(objective, startpoint):
    tstart = time.perf_counter()
    fval, grad = objective(startpoint, (0, 1, ))
    return time.perf_counter() - tstart, fval


problem_multi = get_problem(model_name, variant, dataset, 1,
                            multimodel=True)


problem_single = get_problem(model_name, variant, dataset, 1,
                             multimodel=False)

random.seed(index)

sampler = pypesto.startpoint.UniformStartpoints(
    check_fval=True, check_grad=False,
)

x = sampler.sample(
    n_starts=1, lb=problem_multi.lb, ub=problem_multi.ub,
)
x = sampler.check_and_resample(x, problem_multi.lb, problem_multi.ub,
                               problem_multi.objective)

t_multi, fval_multi = time_objective(problem_multi.objective, x)
print(f'multimodel: {t_multi}')
t_single, fval_single = time_objective(problem_single.objective, x)
print(f'singlemodel: {t_single}')

os.makedirs(get_profile_dir(model_name, variant), exist_ok=True)

pd.DataFrame({
    't_single': t_single, 't_multi': t_multi,
    'fval_single': fval_single, 'fval_multi': fval_multi,
    't_ratio': t_single/t_multi,
}, index=[index]).to_csv(get_multimodel_speedup_result_file(
    model_name, variant, dataset, index
))
