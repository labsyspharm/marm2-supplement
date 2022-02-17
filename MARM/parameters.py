import amici
import os
import pandas as pd
import numpy as np

from .paths import get_parameters_file


def specialise_par_name(name, panrafi, rafi, meki):

    if panrafi is not None:
        name = name.replace('PRAFi', panrafi)

    if rafi is not None:
        name = name.replace('RAFi', rafi)

    if meki is not None:
        name = name.replace('MEKi', meki)

    return name


def load_parameters(model, settings, prafi, rafi, meki, index=0,
                    allow_missing_pars=False):
    scales = amici.parameterScalingFromIntVector([
        amici.ParameterScaling_none
        for _ in model.getParameterNames()
    ])
    model.setParameterScale(scales)

    df_parameters = load_parameters_as_dataframe(settings['model_name'],
                                                 settings['variant'],
                                                 settings['dataset'])

    par = []
    for name in model.getParameterNames():
        if specialise_par_name(name, prafi, rafi, meki) in df_parameters:
            val = df_parameters.loc[index,
                                    specialise_par_name(name, prafi, rafi,
                                                        meki)]
            model.setParameterByName(
                name,
                val
            )
        elif allow_missing_pars:
            val = model.getParameterByName(name)
        else:
            raise ValueError(f'Missing value for parameter {name}')

        par.append(val)

    return par


def load_pysb_parameters(model, model_name, variant, dataset, index=0,
                         allow_missing_pars=False):

    df_parameters = load_parameters_as_dataframe(model_name, variant, dataset)

    for par_name in model.parameters.keys():
        if par_name in df_parameters:
            model.parameters[par_name].value = \
                df_parameters[par_name].values[index]
        elif allow_missing_pars:
            raise RuntimeError(f'Model parameter {par_name} was not estimated')


def load_parameters_as_dataframe(model_name, variant, dataset):
    base_dir = os.path.dirname(__file__)
    parameter_file = os.path.join(base_dir, 'parameters',
                                  f'{model_name}_{variant}_{dataset}.csv')
    return pd.read_csv(parameter_file, index_col=0)


def save_parameters(result, model_name, variant, dataset):
    pars = np.vstack(
        [res['x'] for res in result.optimize_result.list]
    )
    parameter_dict = {
        'fval': [np.nanmin(res.history.get_fval_trace())
                 for res in result.optimize_result.list]
    }
    parameter_dict.update({
        result.problem.x_names[i]: pars[:, i]
        if result.problem.x_names[i].endswith('_phi')
        else pow(10, pars[:, i])
        for i in range(result.problem.dim_full)
    })
    parameters_export = pd.DataFrame(parameter_dict)
    parameters_export.to_csv(get_parameters_file(model_name, variant, dataset))

    return parameters_export
