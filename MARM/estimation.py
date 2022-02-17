import amici
import amici.petab_objective
import amici.parameter_mapping
import pypesto
import itertools
import copy
import os
import pickle
import pandas as pd
import numpy as np
from .paths import (
    get_model_name_dataset, get_results_path, get_model_module_dir_dataset
)
from .parameters import specialise_par_name

RAFI = ['Vemurafenib', 'Dabrafenib', 'PLX8394']
PANRAFI = ['LY3009120', 'AZ_628']
MEKI = ['Cobimetinib', 'Trametinib', 'Selumetinib', 'Binimetinib', 'PD0325901']


def get_fixed_parameters(model, condition, overwrite):
    cond = copy.deepcopy(condition)
    for field in overwrite:
        cond[field] = overwrite[field]
    return [cond[name] for name in model.getFixedParameterNames()]


def get_solver(model):
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    solver.setNewtonMaxSteps(int(0))
    solver.setMaxSteps(int(1e6))
    solver.setAbsoluteTolerance(1e-11)
    solver.setRelativeTolerance(1e-9)
    solver.setAbsoluteToleranceSteadyState(1e-9)
    solver.setRelativeToleranceSteadyState(1e-7)

    return solver


def get_instances(instance_vars):
    instance_vars = [var for var in instance_vars if
                     var not in ['EGFR', 'mutrastraining', 'engineered']]
    instances = []
    for r in range(len(instance_vars) + 1):
        instances.extend(list(itertools.combinations(instance_vars, r)))
    return [
        "_".join(sorted(instance)) for instance in instances
    ]


def get_objective(model_name, variant, dataset, n_threads, multimodel=True,
                  modifications=None, datafile=None):
    full_model = get_model(model_name, variant, dataset, modifications)

    full_parameters = set()
    for param in full_model.getParameterIds():
        for rafi, prafi, meki in itertools.product(RAFI, PANRAFI, MEKI):
            full_parameters.add(specialise_par_name(param, prafi, rafi, meki))
    full_parameters = sorted(list(full_parameters))

    instance_vars = dataset.split('_')

    instances = get_instances(instance_vars)

    objectives = []

    for instance in instances:
        if multimodel:
            model = get_model(model_name, variant, instance, modifications)
        else:
            model = full_model

        solver = get_solver(model)

        edatas = get_edata(dataset if datafile is None else datafile,
                           instance, model)

        if len(edatas) > 0:
            parameter_mapping = amici.parameter_mapping.ParameterMapping([
                amici.petab_objective.ParameterMappingForCondition(
                    map_sim_var={
                        model_param: specialise_par_name(
                            model_param,
                            'LY3009120' if prafi is None and not multimodel
                            else prafi,
                            'Vemurafenib' if rafi is None and not multimodel
                            else rafi,
                            'Cobimetinib' if meki is None and not multimodel
                            else meki,
                        )
                        for model_param in model.getParameterNames()
                    },
                    scale_map_sim_var={
                        par_id: amici.parameter_mapping.amici_to_petab_scale(
                            par_scale
                        )
                        for par_id, par_scale in zip(
                            model.getParameterIds(), model.getParameterScale()
                        )
                    }
                )
                for prafi, rafi, meki, _ in edatas
            ])

            objectives.append(pypesto.objective.AmiciObjective(
                model,
                solver,
                [e[3] for e in edatas],
                x_ids=full_parameters,
                parameter_mapping=parameter_mapping,
                max_sensi_order=None,
                guess_steadystate=False,
                n_threads=n_threads,
            ))

    return pypesto.objective.AggregatedObjective(
        objectives=objectives,
        x_names=full_parameters
    )


def get_problem(model_name, variant, dataset, n_threads, multimodel=True):

    objective = get_objective(model_name, variant, dataset, n_threads,
                              multimodel)

    names = objective.x_names

    ub = np.ones(len(names)) * 3
    lb = np.ones(len(names)) * -2

    for par in names:
        # initial values
        # scale: log10
        # unit: molecules
        if par.endswith('_0'):
            if par == 'GRB2_0':
                lb[list(names).index(par)] = 4
                ub[list(names).index(par)] = 5
            elif par == 'SOS1_0':
                lb[list(names).index(par)] = 3.5
                ub[list(names).index(par)] = 4.5
            elif par == 'RAS_0':
                lb[list(names).index(par)] = 4
                ub[list(names).index(par)] = 5
            elif par == 'BRAF_0':
                lb[list(names).index(par)] = 2.5
                ub[list(names).index(par)] = 3.5
            elif par == 'CRAF_0':
                lb[list(names).index(par)] = 3.5
                ub[list(names).index(par)] = 4.5
            elif par == 'MEK_0':
                lb[list(names).index(par)] = 4.5
                ub[list(names).index(par)] = 5.5
            elif par == 'ERK_0':
                lb[list(names).index(par)] = 4.5
                ub[list(names).index(par)] = 5.5
            elif par == 'CBL_0':
                lb[list(names).index(par)] = 3.5
                ub[list(names).index(par)] = 4.5
            else:
                lb[list(names).index(par)] = 2
                ub[list(names).index(par)] = 8

        elif par.endswith('_eq'):
            if par == 'SPRY_eq':
                lb[list(names).index(par)] = 3.5
                ub[list(names).index(par)] = 5.5
            if par == 'DUSP_eq':
                lb[list(names).index(par)] = 1.5
                ub[list(names).index(par)] = 3.5
            if par == 'EGFR_eq':
                lb[list(names).index(par)] = 3
                ub[list(names).index(par)] = 5
            if par == 'mSPRY_eq':
                lb[list(names).index(par)] = 0.5
                ub[list(names).index(par)] = 1.5
            if par == 'mDUSP_eq':
                lb[list(names).index(par)] = 1.0
                ub[list(names).index(par)] = 2.0
            if par == 'mEGFR_eq':
                lb[list(names).index(par)] = 0.5
                ub[list(names).index(par)] = 1.5

        elif par.endswith('_gexpslope'):
            lb[list(names).index(par)] = 0
            ub[list(names).index(par)] = 5

        # thermodynamic constants
        # scale: linear
        # unit: unitless
        elif par.endswith('_phi'):
            lb[list(names).index(par)] = 0
            ub[list(names).index(par)] = 1
        # energy differences
        # scale: log10
        # unit: unitless?
        elif par.endswith('_deltaG'):
            if par in [
                'ep_SOS1S1134p_GRB2_deltaG',
            ]:
                lb[list(names).index(par)] = 0
                ub[list(names).index(par)] = 6
            if par in [

                'ep_MEKphosphop_MEKi_deltaG',
            ]:
                lb[list(names).index(par)] = 0
                ub[list(names).index(par)] = 4
            elif par in [
                'ep_RAF_RAF_mod_RAFi_double_deltaG'.replace('RAFi', rafi)
                for rafi in RAFI
            ]:
                lb[list(names).index(par)] = 0
                ub[list(names).index(par)] = 5
            elif par in [
                'ep_EGFR_EGFR_mod_EGF_single_deltaG',
                *['ep_RAF_RAF_mod_RAFi_single_deltaG'.replace('RAFi', rafi)
                  for rafi in RAFI],
            ]:
                lb[list(names).index(par)] = -5
                ub[list(names).index(par)] = 0
            elif par in ['ep_RAS_RAF_mod_RAFi_single_deltaG'.replace('RAFi', rafi)
                  for rafi in RAFI]:
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 0
            elif par == 'ep_RAF_RAF_mod_RASstategtp_double_deltaG':
                lb[list(names).index(par)] = -10
                ub[list(names).index(par)] = -2
            else:
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 4
        # kon
        # scale: log10
        # unit: 1/uM*1/h
        elif par.endswith('_kf'):
            if par in ['bind_pEGFR_GRB2_kf', 'bind_EGF_EGFR_kf',
                       'bind_EGFR_EGFR_kf', 'bind_GRB2_SOS1_kf',
                       'bind_DUSP_ERKphosphop_kf', 'bind_RASstategtp_RAF_kf',
                       'bind_CBL_GRB2_kf', 'bind_MEK_ERKphosphou_kf',
                       'bind_SOS1_RAS_kf', ]:
                lb[list(names).index(par)] = 2
                ub[list(names).index(par)] = 7
            else:
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 4
        # koff
        # scale: log10
        # unit: 1/h
        elif par.endswith('_kD'):
            if par in [
                'bind_EGFR_EGFR_kD',
            ]:
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = 2
            elif par == 'bind_RAF_RAF_kD':
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 4
            elif par == 'bind_RAFrafANY_MEKphosphou_kD':
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = 2
            elif par == 'bind_DUSP_ERKphosphop_kD':
                lb[list(names).index(par)] = -7
                ub[list(names).index(par)] = -2
            elif par == 'bind_MEK_ERKphosphou_kD':
                lb[list(names).index(par)] = -6
                ub[list(names).index(par)] = 0
            elif par == 'bind_Vemurafenib_RAF_kD':
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 0
            elif par == 'bind_Trametinib_MEK_kD':
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = -1
            elif par == 'bind_Selumetinib_MEK_kD':
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = -1
            elif par == 'bind_PLX8394_RAF_kD':
                lb[list(names).index(par)] = -3
                ub[list(names).index(par)] = -1
            elif par == 'bind_PD0325901_MEK_kD':
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = -2
            elif par == 'bind_LY3009120_RAF_kD':
                lb[list(names).index(par)] = -3
                ub[list(names).index(par)] = -1
            elif par == 'bind_Dabrafenib_RAF_kD':
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = -2
            elif par == 'bind_Cobimetinib_MEK_kD':
                lb[list(names).index(par)] = -3
                ub[list(names).index(par)] = -1
            elif par == 'bind_Binimetinib_MEK_kD':
                lb[list(names).index(par)] = -3
                ub[list(names).index(par)] = -1
            elif par == 'bind_AZ_628_RAF_kD':
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = -1
            else:
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = 0

        # kcat
        # scale: log10
        # unit: 1/h
        elif par.endswith('_deg_kcat'):
            lb[list(names).index(par)] = 0
            ub[list(names).index(par)] = 3
        # keff
        # scale: log10
        # unit: 1/h
        elif par.endswith('_kcatr'):
            if par == 'catalyze_PP2A_MEK_u_kcatr':
                lb[list(names).index(par)] = -6
                ub[list(names).index(par)] = 0
            elif par in [
                'catalyze_RAFrafiNone_MEKmeki_MEKi_p_kcatr'.replace('MEKi',
                                                                    meki)
                for meki in MEKI
            ]:
                lb[list(names).index(par)] = -6
                ub[list(names).index(par)] = 0
            else:
                lb[list(names).index(par)] = -4
                ub[list(names).index(par)] = 0

        elif par.endswith('_kcat'):
            lb[list(names).index(par)] = 1
            ub[list(names).index(par)] = 5

        elif par.endswith('_kM'):
            lb[list(names).index(par)] = -3
            ub[list(names).index(par)] = 1

        # kdeg
        # scale: log10
        # unit: 1/uM*1/h
        elif par.endswith('_kdeg'):
            if par.startswith('p'):
                lb[list(names).index(par)] = -3
                ub[list(names).index(par)] = 0
            elif par.startswith('m'):
                lb[list(names).index(par)] = -2
                ub[list(names).index(par)] = 1

        elif par.endswith('_kbase'):
            lb[list(names).index(par)] = -3
            ub[list(names).index(par)] = 1

        elif par.endswith('IF_scale'):
            lb[list(names).index(par)] = 0
            ub[list(names).index(par)] = 2

        elif par.endswith('IF_offset'):
            lb[list(names).index(par)] = -2
            ub[list(names).index(par)] = -0.5

    fixed_idx = []
    fixed_vals = []

    fixed_vals += [0.0 for idx, name in enumerate(names)
                   if name.endswith('_phi') and idx not in fixed_idx]
    fixed_idx += [idx for idx, name in enumerate(names)
                  if name.endswith('_phi') and idx not in fixed_idx]

    # paradox breaker
    fixed_vals.append(0.0)
    fixed_idx.append(names.index('ep_RAF_RAF_mod_PLX8394_single_deltaG'))

    for val, idx in zip(fixed_vals, fixed_idx):
        print(f'fixing {names[idx]} to {val}')

    x_scales = ['lin' if name.endswith('_phi') else 'log10'
                for name in names]

    return pypesto.problem.Problem(
        objective, lb, ub,
        x_fixed_indices=fixed_idx,
        x_fixed_vals=fixed_vals,
        x_scales=x_scales,
        x_names=names,
    )


def par_alias(name, model_name, dataset):

    alias = {}
    return alias.get((model_name, dataset), {}).get(name, name)


def get_result(model_name, variant, dataset):
    problem = get_problem(model_name, variant, dataset, 1)
    rfile = os.path.join(
        get_results_path(model_name, variant),
        f'{dataset}.pickle'
    )
    with open(rfile, 'rb') as f:
        optimize_result, par_names = pickle.load(f)

    result = pypesto.Result(problem)
    for opt_result, names in zip(optimize_result, par_names):
        if opt_result['x'] is None:
            continue

        sorted_par_idx = [
            names.index(name)
            for name in problem.x_names
        ]
        x_sorted = [opt_result['x'][sorted_par_idx[ix]] for ix in
                    range(len(problem.x_names))]
        opt_result['x'] = x_sorted
        result.optimize_result.list.append(opt_result)

    result.optimize_result.list = [
        r for r in result.optimize_result.list
        if np.isfinite(r['fval'])
    ]

    result.optimize_result.sort()

    return result


def load_fixed_parameters(model_name, variant, dataset):
    model = get_model(model_name, variant, dataset)
    result = get_result(model_name, variant, dataset)

    pars = np.vstack(
        [np.array(res['x']) for res in result.optimize_result.list
         if res.fval < result.optimize_result.list[0].fval + 0.2]
    )

    par_std = pars.std(axis=0)
    par_mean = pars.mean(axis=0)
    par_name = model.getParameterNames()

    fixed_idx = [idx for idx in range(len(par_std))
                 if par_std[idx] < 0.1]
    fixed_vals = [par_mean[idx] for idx in fixed_idx]
    fixed_names = [par_name[idx] for idx in fixed_idx]

    print(f'loading values for parameters from model {model_name} trained on '
          f'{dataset} data:')
    for name, val in zip(fixed_names, fixed_vals):
        print(f'{name}: {val}')

    return fixed_names, fixed_vals


def get_model(model_name, variant, dataset, modifications=None):

    full_name = get_model_name_dataset(model_name, variant, dataset,
                                       modifications)
    model_output_dir = get_model_module_dir_dataset(model_name, variant,
                                                    dataset, modifications)
    model_module = amici.import_model_module(full_name, model_output_dir)

    model = model_module.getModel()

    scales = amici.parameterScalingFromIntVector([
        amici.ParameterScaling.none if par.endswith('_phi') else
        amici.ParameterScaling.log10
        for par in model.getParameterNames()
    ])

    model.setParameterScale(scales)
    model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.newtonOnly
    )
    return model


def get_edata(dataset, instance, model):
    base_dir = os.path.dirname(__file__)
    data_file = os.path.join(base_dir, 'data', f'processed_{dataset}.csv')
    exp_data = pd.read_csv(data_file, index_col=0)

    instances = instance.lower().split('_')

    if 'egf' not in instances:
        exp_data = exp_data[exp_data.EGF_0 == 0]
    else:
        exp_data = exp_data[exp_data.EGF_0 > 0]

    if 'rafi' not in instances:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in RAFI]] == 0).all(axis=1)
        ]
    else:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in RAFI]] > 0).any(axis=1)
        ]

    if 'prafi' not in instances:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in PANRAFI]] == 0).all(axis=1)
        ]
    else:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in PANRAFI]] > 0).any(axis=1)
        ]

    if 'meki' not in instances:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in MEKI]] == 0).all(axis=1)
        ]
    else:
        exp_data = exp_data[
            (exp_data[[f'{drug}_0' for drug in MEKI]] > 0).any(axis=1)
        ]

    if len(exp_data):
        edatas_labeled = []
        for prafi, rafi, meki in itertools.product(
                PANRAFI + [None], RAFI + [None], MEKI + [None]
        ):
            combo_data = exp_data[
                pd.concat([
                    exp_data[f'{drug}_0'] > 0
                    if drug is not None
                    else pd.concat([
                        exp_data[f'{drug_zero}_0'] == 0
                        for drug_zero in zero_drugs
                    ], axis=1).all(axis=1)
                    for drug, zero_drugs in zip([prafi, rafi, meki],
                                                [PANRAFI, RAFI, MEKI])
                ], axis=1).all(axis=1)
            ]
            for suffix in ['', '_preeq', '_presim']:
                for drug, drug_init in zip([prafi, rafi, meki],
                                           ['PRAFi_0', 'RAFi_0', 'MEKi_0']):
                    if drug is None:
                        val = 0.0
                    else:
                        val = combo_data[f'{drug}_0{suffix}']
                    combo_data[drug_init + suffix] = val

            if len(combo_data):
                edatas = amici.getEdataFromDataFrame(model, combo_data)
                for edata in edatas:
                    edata.reinitializeFixedParameterInitialStates = True
                    edatas_labeled.append((prafi, rafi, meki, edata))

        return edatas_labeled
    else:
        return []
