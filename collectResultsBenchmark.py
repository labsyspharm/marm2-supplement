import sys
import os
import re

import pandas as pd

from MARM.paths import get_profile_dir, get_results_path


if __name__ == "__main__":
    model_name = sys.argv[1]
    variant = sys.argv[2]
    dataset = sys.argv[3]

    profile_dir = get_profile_dir(model_name, variant)

    benchmark_files = os.listdir(profile_dir)
    results = []
    for file in benchmark_files:
        if re.search(r'multimodel_objective_[0-9]+\.csv$', file) and \
                not file.startswith('.'):
            results.append(pd.read_csv(os.path.join(profile_dir, file)))

    result = pd.concat(results)
    result.to_csv(os.path.join(
        get_results_path(model_name, variant),
        f'{dataset}_multimodel_benchmark.csv'
    ))


