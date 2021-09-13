import re
import sys
import os
import pickle
import shutil
import pandas as pd
import pypesto
import numpy as np
import datetime
from MARM.estimation import get_model
from MARM.paths import get_results_path, get_traces_path
from MARM.estimation import get_problem
from pypesto.optimize.optimizer import read_result_from_file


if __name__ == "__main__":
    model_name = sys.argv[1]
    variant = sys.argv[2]
    dataset = sys.argv[3]

    problem = get_problem(model_name, variant, dataset, 0, 1)

    results_path = get_results_path(model_name, variant)

    result_files = os.listdir(results_path)

    traces_path = get_traces_path(model_name, variant)

    trace_files = os.listdir(traces_path)

    optimizer_results = []

    parsed_results = []

    par_names = []

    for file in trace_files:
        if re.match(dataset + '_[0-9]+_[0-9]+\.csv', file):
            sfile = file.replace('.', '_')
            splitted = [int(s) for s in sfile.split('_') if s.isdigit()]
            run = splitted[0]
            start = splitted[1]

            rfile = f'{dataset}-{run}.pickle'
            if rfile in result_files and rfile not in parsed_results:
                print(f'loading full results for run {run}')
                with open(os.path.join(results_path, rfile), 'rb') as f:
                    result = pickle.load(f)
                    # thin results
                    optimizer_results += result.list
                    for r in result.list:
                        if r['x'] is None:
                            continue
                        if len(r['x']) < len(problem.x_names):
                            r.update_to_full(problem)
                        par_names.append([
                            problem.x_names[ix] 
                            if ix in problem.x_fixed_indices
                            else result.list[0].history._trace['x'].columns[
                                problem.x_free_indices.index(ix)
                            ]
                            for ix in range(problem.dim_full)
                        ])
                parsed_results.append(rfile)

            elif rfile not in result_files:
                print(f'loading partial results for run {run}, start {start}')

                history_options = pypesto.HistoryOptions(
                    trace_record=True,
                    trace_record_hess=False,
                    trace_record_res=False,
                    trace_record_sres=False,
                    trace_record_schi2=False,
                    storage_file=os.path.join(
                        get_traces_path(model_name, variant),
                        f'{dataset}_{run}_{{id}}.csv',
                    ),
                    trace_save_iter=1
                )
                result = read_result_from_file(problem, history_options,
                                               str(start))
                par_names.append([
                    problem.x_names[ix] 
                    if ix in problem.x_fixed_indices
                    else result.history._trace['x'].columns[
                        problem.x_free_indices.index(ix)
                    ]
                    for ix in range(problem.dim_full)
                ])

                optimizer_results.append(result)

    outfile = os.path.join(results_path, f'{dataset}.pickle')

    print(sorted([
        r['fval']
        for r in optimizer_results
    ])[0:min(5, len(optimizer_results))])

    with open(outfile, 'wb') as f:
        pickle.dump((optimizer_results, par_names), f)

    shutil.copyfile(outfile, outfile.replace(
        '.pickle',
        '{date:%Y-%m-%d %H:%M:%S}.pickle'.format(date=datetime.datetime.now())
    ))
