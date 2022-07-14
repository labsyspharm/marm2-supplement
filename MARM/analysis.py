import os
import re
import pandas as pd
import numpy as np
import amici
import importlib
import copy
import datetime

from sklearn.metrics import auc

from .visualize.common import get_value_dict
from .estimation import (
    get_model, get_solver, get_model_name_dataset, get_objective, RAFI,
    MEKI, PANRAFI
)
from .parameters import load_parameters, load_parameters_as_dataframe
from .paths import get_analysis_results_file, get_directory

N_RUNS = 50


def read_settings(argv, index=True, threads=True):
    s = dict()
    s['model_name'], s['variant'], s['dataset'] = argv[1:4]
    if index:
        s['index'] = int(argv[4])
    if threads:
        s['threads'] = int(argv[5])
    return s


def load_model_solver(s, prafi=None, rafi=None, meki=None, mods=None):
    instance_vars = ['EGF']
    if prafi is not None:
        instance_vars.append('PRAFi')
    if rafi is not None:
        instance_vars.append('RAFi')
    if meki is not None:
        instance_vars.append('MEKi')

    instance = '_'.join(sorted(instance_vars))
    model = get_model(s['model_name'], s['variant'], instance, mods)
    full_name = get_model_name_dataset(s['model_name'], s['variant'],
                                       instance, mods)
    solver = get_solver(model)
    solver.setSensitivityOrder(amici.SensitivityOrder_none)

    par = load_parameters(model, s, prafi, rafi, meki, index=s['index'])
    return par, model, solver, full_name


def extend_datapoints(e, tps):
    tps = sorted(list(e.getTimepoints())
                 + list(tps))

    current_data = np.reshape(e.getObservedData(),
                              (e.nt(), e.nytrue()))
    current_data_std = np.reshape(e.getObservedDataStdDev(),
                                  (e.nt(), e.nytrue()))
    new_data = np.zeros((len(tps), e.nytrue())) * np.NaN
    new_data_std = np.zeros((len(tps), e.nytrue())) * np.NaN
    for val in np.unique(e.getTimepoints()):
        for count in range(e.getTimepoints().count(val)):
            new_data[tps.index(val, count), :] = \
                current_data[e.getTimepoints().index(val, count), :]
            new_data_std[tps.index(val, count), :] = \
                current_data_std[e.getTimepoints().index(val, count), :]

    e.setTimepoints(tps)
    e.setObservedData(new_data.flatten())
    e.setObservedDataStdDev(new_data_std.flatten())


def run_and_store_simulation(sxs, filename, par_dict=None,
                             channel_specific_dusp_binding=False):
    datasets = {
        'trainingdata': sxs['dataset'],
        'finepulse': sxs['dataset'],
        'singleprediction':
            'EGF_EGFR_MEKi_PRAFi_RAFi_singleprediction',
        'comboprediction':
            'EGF_EGFR_MEKi_PRAFi_RAFi_comboprediction',
        'panrafcomboprediction':
            'EGF_EGFR_MEKi_PRAFi_RAFi_panrafcomboprediction',
        'mutRASprediction_engineered':
            'MEKi_NRAS_PRAFi_RAFi_engineered_mutrasprediction',
        'mutRASprediction_engineered_combo':
            'MEKi_NRAS_PRAFi_RAFi_engineered_mutrascomboprediction',
        'ht29': 'EGF_EGFR_MEKi_PRAFi_RAFi_ht29'
    }
    if channel_specific_dusp_binding:
        modifications = 'channelcf_monoobs'
    else:
        modifications = 'channel_monoobs'
    if filename.startswith('mutRAS'):
        perts = sxs['dataset'].split('_')
        for v in ['EGF', 'EGFR']:
            if v in perts:
                perts.remove(v)
        perts += ['NRAS']
        dataset = '_'.join(sorted(perts))
    else:
        dataset = sxs['dataset']
    obj = get_objective(sxs['model_name'], sxs['variant'],
                        dataset, sxs['threads'],
                        multimodel=True, modifications=modifications,
                        datafile=datasets[filename])
    if par_dict is None:
        df_parameters = load_parameters_as_dataframe(sxs['model_name'],
                                                     sxs['variant'],
                                                     sxs['dataset'])
        par = df_parameters.loc[sxs['index'], obj.x_names].values

        if filename == 'ht29':
            ccle_abundances = pd.read_csv(os.path.join(
                get_directory(), 'data', 'ccle_abundances.csv',
            ), index_col=[0]).T

            # find control condition (3 params are volume/EGF mol weight and
            # avogadro constant)
            cobj = next(
                o
                for o in obj._objectives
                if len(o.edatas) == 1
                and len(o.edatas[0].fixedParameters) == 3
            )

            # simulate control condition
            sim_ref = cobj([
                val if name.endswith(('_phi', '_dG', '_ddG'))
                else np.log10(val)
                for val, name in zip(par, cobj.x_names)
            ], sensi_orders=(0,), return_dict=True,
            amici_reporting=amici.RDataReporting.full)['rdatas'][0]

            # simulate control conditions, correct for offset & scaling
            pERK_ref = (sim_ref.y[
                0, cobj.amici_model.getObservableNames().index('pERK_IF_obs')
            ] - par[obj.x_names.index('pERK_IF_offset')]) \
                / par[obj.x_names.index('pERK_IF_scale')]

            diff = ccle_abundances.HT29 - ccle_abundances.A375

            aliases = {
                'CRAF': ['RAF1'],
                'ERK': ['MAPK1', 'MAPK3'],
                'MEK': ['MAP2K1', 'MAP2K2'],
                'mDUSP': ['mDUSP4', 'mDUSP6'],
                'mSPRY': ['mSPRY2', 'mSPRY4'],
                'DUSP': ['DUSP4', 'DUSP6'],
                'SPRY': ['SPRY2', 'SPRY4'],
                'RAS': ['HRAS', 'KRAS', 'NRAS'],
            }

            for ix, xname in enumerate(obj.x_names):
                if xname.endswith('_0'):
                    x_diff = diff[
                        aliases.get(xname[:-2], [xname[:-2]])
                    ].mean()
                else:
                    continue

                par_diff = np.exp(x_diff)
                print(f'multiplying parameter {xname} by {par_diff} for '
                      f'cell line ht29')
                par[ix] *= par_diff

            #par[obj.x_names.index('DUSP_eq')] /= 1000

            sim_mod = cobj([
                val if name.endswith(('_phi', '_dG', '_ddG'))
                else np.log10(val)
                for val, name in zip(par, cobj.x_names)
            ], sensi_orders=(0,), return_dict=True,
            amici_reporting=amici.RDataReporting.full)['rdatas'][0]

            pERK_mod = (sim_mod.y[
                0, cobj.amici_model.getObservableNames().index('pERK_IF_obs')
            ] - par[obj.x_names.index('pERK_IF_offset')]) \
                / par[obj.x_names.index('pERK_IF_scale')]

            for ix, xname in enumerate(obj.x_names):
                if xname.endswith('_eq'):
                    gene_name = xname[:-3].replace("m", "")
                    kM = par[obj.x_names.index(
                        f'synthesize_pERK_{gene_name}_ERK_kM'
                    )]
                    # correct for difference in baseline pERK
                    gexprfactor = (pERK_ref / (pERK_ref + kM)) / (
                                pERK_mod / (pERK_mod + kM))

                    if xname.startswith('m'):
                        x_diff = diff[
                            aliases.get(xname[:-3], [xname[:-3]])
                        ].mean() + gexprfactor
                    else:
                        x_diff = diff[
                            aliases.get(xname[:-3], [xname[:-3]])
                        ].mean() + gexprfactor

                else:
                    continue

                print(f'multiplying parameter {xname} by {np.exp(x_diff)} for '
                      f'cell line ht29')
                par[ix] *= np.exp(x_diff)

            crdata = cobj([
                    val if name.endswith(('_phi', '_dG', '_ddG'))
                    else np.log10(val)
                    for val, name in zip(par, obj.x_names)
            ], sensi_orders=(0,), return_dict=True)['rdatas']

            par[obj.x_names.index('pERK_IF_scale')] *= (
                1 - par[obj.x_names.index('pERK_IF_offset')]
            ) / (crdata[0].y[
                0, cobj.amici_model.getObservableNames().index('pERK_IF_obs')
            ] - par[obj.x_names.index('pERK_IF_offset')])

    else:
        par = [par_dict[name] for name in obj.x_names]

    par = [
        val if name.endswith(('_phi', '_dG', '_ddG'))
        else np.log10(val)
        for val, name in zip(par, obj.x_names)
    ]

    for objective in obj._objectives:
        model = objective.amici_model
        if filename == 'comboprediction':
            for e in objective.edatas:
                extend_datapoints(e, np.logspace(-4, 1, 51))

        elif filename == 'finepulse':
            selected_e = [ic for ic, e in enumerate(objective.edatas)
                          if len(e.getTimepoints()) > 2
                          and e.t_presim == 0
                          and 'EGF_0' in model.getFixedParameterNames()]
            objective.edatas = [
                objective.edatas[ic] for ic in selected_e
            ]
            objective.parameter_mapping.parameter_mappings = [
                objective.parameter_mapping.parameter_mappings[ic]
                for ic in selected_e
            ]
            for e in objective.edatas:
                extend_datapoints(e, np.linspace(0, 2.0, 51))

    obj._objectives = [objective for objective in obj._objectives
                       if len(objective.edatas) > 0]

    rdatas = obj(par, sensi_orders=(0,), return_dict=True,
                 amici_reporting=amici.RDataReporting.full)['rdatas']
    edatas = [edata
              for objective in obj._objectives
              for edata in objective.edatas]
    models = [objective.amici_model
              for objective in obj._objectives
              for _ in objective.edatas]
    mappings = [mapping
                for objective in obj._objectives
                for mapping in objective.parameter_mapping.parameter_mappings]

    drugs = [
        tuple([
            re.search(
                fr'bind_([\w_]+)_{target}_dG',
                mapping.map_sim_var[f'bind_{inh}_{target}_dG']
            ).group(1)
            if f'bind_{inh}_{target}_dG' in mapping.map_sim_var
            else None
            for inh, target in zip(['PRAFi', 'RAFi', 'MEKi'],
                                   ['RAF', 'RAF', 'MEK'])
        ])
        for mapping in mappings
    ]

    dfs = []
    for rdata, edata, model, (prafi, rafi, meki) in zip(rdatas, edatas, models,
                                                        drugs):
        df_rdata = amici.getSimulationObservablesAsDataFrame(model, edata,
                                                             rdata)
        df_edata = amici.getDataObservablesAsDataFrame(model, edata)

        df_instance = pd.concat([df_edata, df_rdata])
        rename_and_fill_drug_columns(df_instance, prafi, rafi, meki)
        for suffix in ['', '_preeq', '_presim']:
            colname = 'EGF_0' + suffix
            if colname not in df_instance.columns:
                df_instance[colname] = 0.0
        dfs.append(df_instance)

    df = pd.concat(dfs)
    if channel_specific_dusp_binding:
        filename += '_cf'
    write_analysis_dataframe(df, sxs, filename)


def rename_and_fill_drug_columns(df, prafi=None, rafi=None, meki=None):
    for suffix in ['', '_preeq', '_presim']:
        for drug, init in zip([prafi, rafi, meki],
                              ['PRAFi_0', 'RAFi_0', 'MEKi_0']):
            if drug is not None:
                df.rename(
                    columns={f'{init}{suffix}': f'{drug}_0{suffix}'},
                    inplace=True
                )
            elif f'{init}{suffix}' in df.columns:
                df.drop(columns=f'{init}{suffix}', inplace=True)
        for drug in PANRAFI + RAFI + MEKI:
            colname = f'{drug}_0' + suffix
            if colname not in df.columns:
                df[colname] = 0.0


def load_model_aux(full_name):
    pysb_model = importlib.import_module(
        f'.pysb_flat.{full_name}', 'MARM',
    ).model

    observable = importlib.import_module(
        f'.observables.{full_name}_observable', 'MARM',
    ).observable

    return pysb_model, observable


def write_analysis_dataframe(df, s, filename):
    file = get_analysis_results_file(s['model_name'], s['variant'],
                                     s['dataset'], filename, s['index'])
    os.makedirs(os.path.dirname(file), exist_ok=True)
    df.to_csv(file)


def read_analysis_dataframe(s, filename, index=None):
    file = get_analysis_results_file(s['model_name'], s['variant'],
                                     s['dataset'], filename,
                                     index if index is not None
                                     else s['index'])
    os.makedirs(os.path.dirname(file), exist_ok=True)
    return pd.read_csv(file, index_col=0)


def read_all_analysis_dataframes(s, filename, tps=None):
    dfs = []
    for index in range(N_RUNS):
        try:
            df_index = read_analysis_dataframe(s, filename, index)
            df_index['par_index'] = index
            if tps is not None:
                df_index = df_index[df_index.time.apply(lambda x: x in tps)]
            dfs.append(df_index)
        except FileNotFoundError as err:
            print(f'missing data for index {index} ({str(err)})')

    return pd.concat(dfs)


def process_rdata(rdatas, fun, mode, p):
    funvals = []
    for rdata in rdatas:
        for it, x in enumerate(rdata['x']):
            val = fun(x.transpose(), p)

            value_dict = get_value_dict(val, mode)

            funvals.append(value_dict)

    df = pd.DataFrame.from_dict(
        {key: [funval[key] for funval in funvals] for key in funvals[0]},
        orient='columns',
    )
    return df


def get_entropy(df):
    df_local = df.loc[:, df.min(axis=0) > 0]
    return df_local.div(
        df_local.sum(axis=1), axis=0
    ).apply(
        lambda x: -x * np.log2(x)
    ).sum(axis=1).values


def get_kldivergence(df, ref_index):
    prob = df.div(
        df.sum(axis=1), axis=0
    )
    return np.asarray([
        row.div(
            prob.loc[ref_index]
        ).apply(np.log2)[row > 0].mul(
            row[row > 0]
        ).sum()
        for ir, row in prob.iterrows()
    ])


def extend_drug_adapted(df, t_adaption):
    df_drug_adapted = copy.deepcopy(df[(df.time == 0) &
                                       (df.datatype == 'simulation')])

    df_drug_adapted.time = -t_adaption
    return pd.concat([df, df_drug_adapted])


def get_obs_df(df, model):
    # alias for RAFi_0 == 0 to plot it in a logplot
    drug_zeros = dict()
    for drug in RAFI + PANRAFI + MEKI:
        concs = df[f'{drug}_0'].unique()
        if sum(concs > 0) > 1:
            drug0 = np.min(concs[concs > 0])/10
            df.loc[df[f'{drug}_0'] == 0, f'{drug}_0'] = drug0
            drug_zeros[drug] = drug0

    obs = list(model.getObservableNames())
    obs = [ob for ob in obs
           if ob not in [
               'tEGF_obs', 'tRAFi_obs', 'tMEKi_obs', 'tPRAFi_obs', 'pMEK_obs',
           ]]

    # phosphorylated
    if_obs = [ob for ob in obs if ob.endswith('IF_obs')]
    # total abundances
    t_obs = [ob for ob in obs if ob.startswith('t')]
    # channel
    a_obs = ['pMEK_phys_obs', 'pMEK_onco_obs', 'pERK_phys_obs',
             'pERK_onco_obs']
    # fraction phosphorylated
    p_obs = [ob for ob in obs
             if ob.startswith('p') and not ob.endswith('IF_obs')
             and ob not in a_obs and not ob == 'pEGFR_obs']

    data_filter = (df.datatype == 'data') & \
                  (df.par_index == df.par_index.min())
    # process data for plotting as lines and errorbars

    id_vars = [f'{drug}_0{suffix}'
               for drug in RAFI + PANRAFI + MEKI
               for suffix in ['', '_preeq', '_presim']] + [
        'EGF_0', 'EGFR_crispr', 'datatype', 'time', 't_presim',
        'NRAS_Q61mut', 'par_index',
    ]

    id_vars = [var for var in id_vars if var in df.columns]

    df_data_obs = pd.melt(
        df[data_filter],
        id_vars=id_vars,
        value_vars=obs
    )

    df_data_std = pd.melt(
        df[data_filter],
        id_vars=id_vars,
        value_vars=[ob + '_std' for ob in obs]
    )

    df_data_obs['ymin'] = df_data_obs.value - df_data_std.value
    df_data_obs['ymax'] = df_data_obs.value + df_data_std.value

    df_sim_obs = pd.melt(
        df[(df.datatype == 'simulation')],
        id_vars=id_vars,
        value_vars=obs
    )

    for df in [df_sim_obs, df_data_obs]:
        # add 200 as extra value to get right colors
        df.EGF_0 = pd.Categorical(df.EGF_0, ordered=True,
                                  categories=[0.0, 100.0, 200.0])
    return df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zeros


def get_signal_deconvolution_df(df, iterator, par_index):
    df = copy.deepcopy(df[df.par_index == par_index])

    plots = node_species(df, concs=True)

    node = node_order()

    channel = node_channel()

    for label, col in plots.items():
        df[label] = df[col].sum(axis=1)

    df_melt = pd.melt(df, id_vars=[iterator, 'time'],
                      value_vars=list(channel.keys()))

    df_melt['step'] = df_melt.variable.apply(lambda x: node[x])
    df_melt['channel'] = df_melt.variable.apply(lambda x: channel[x])
    return df_melt


def signaling_steps():
    def conc_min(x):
        return x.values[0]

    def conc_max(x):
        return x.values[-1]

    return[
        ('RASgtp', 'active EGFR',  {'type': 'phys', 'normfun': conc_max}),
        ('phys pMEK', 'RASgtp',    {'type': 'phys', 'normfun': conc_min}),
        ('phys pERK', 'phys pMEK', {'type': 'phys', 'normfun': conc_min}),
        ('onco pMEK', 'BRAF600E',  {'type': 'onco', 'normfun': conc_min}),
        ('onco pERK', 'onco pMEK', {'type': 'onco', 'normfun': conc_min}),
    ]


def nodes(df):
    return {
        node: {
            'species': species,
            'step': node_order()[node],
            'channel': node_channel()[node],
        }
        for node, species in node_species(df).items()
    }


def node_species(df, concs=False):
    return {
        'active EGFR': [col for col in df.columns
                        if col.startswith('expl_EGFR') and 'Tyrp' in col]
        if concs else ['activeEGFR_obs'],
        'BRAF600E': ['tBRAF']
        if concs else ['tBRAF_obs'],
        'RASgtp': ['gtpRAS']
        if concs else ['gtpRAS_obs'],
        'onco pMEK': [col for col in df.columns
                      if col.startswith('expl_MEK') and 'channelonco' in col]
        if concs else ['pMEK_onco_obs'],
        'phys pMEK': [col for col in df.columns
                      if col.startswith('expl_MEK') and 'channelphys' in col]
        if concs else ['pMEK_phys_obs'],
        'onco pERK': [col for col in df.columns
                      if col.startswith('expl_ERK') and 'channelonco' in col]
        if concs else ['pERK_onco_obs'],
        'phys pERK': [col for col in df.columns
                      if col.startswith('expl_ERK') and 'channelphys' in col]
        if concs else ['pERK_phys_obs'],
    }


def node_order():
    return {
        'active EGFR': 0,
        'BRAF600E': 1,
        'RASgtp': 1,
        'onco pMEK': 2,
        'phys pMEK': 2,
        'onco pERK': 3,
        'phys pERK': 3,
        # 'DUSP': 4,
        # 'competent SOS1': 4,
    }


def node_channel():
    return {
        'active EGFR': 'phys',
        'BRAF600E': 'onco',
        'RASgtp': 'phys',
        'onco pMEK': 'onco',
        'phys pMEK': 'phys',
        'onco pERK': 'onco',
        'phys pERK': 'phys',
        # 'DUSP': 'onco',
        # 'competent SOS1': 'phys',
    }


def get_signal_transduction_df(sxs, filename, frame_filter, iterators, mode):
    if mode == 'peak':
        peakfun = 'max'

        def signal_act(data, species):
            return getattr(data[species].sum(axis=1),
                           peakfun)()
    elif mode == 'int_log10':
        def signal_act(data, species):
            df_subsetted = data[
                (data.time > 0) & (data.time != 0.0833) & (data.time != 8)
            ]
            if len(df_subsetted):
                return auc(
                    df_subsetted.time.apply(np.log10),
                    df_subsetted[species].sum(axis=1)
                )
            else:
                return 0.0
    else:
        ValueError(f'invalid mode {mode}.')

    transductions = []
    groupvars = iterators + ['datatype', 'EGF_0', 'EGFR_crispr']
    for par_index in range(N_RUNS):
        try:
            df = read_analysis_dataframe(sxs, filename, par_index)
            df = df[frame_filter(df)]
            signal_activity = {
                node: [
                    signal_act(data, species)
                    for _, data in df.groupby(groupvars)
                ]
                for node, species in node_species(df).items()
            }
            concentrations = {
                var: [
                    vals[idx] for vals, _ in df.groupby(groupvars)
                ]
                for idx, var in enumerate(groupvars)
            }
            df_trans = pd.DataFrame(dict(**concentrations, **signal_activity))
            df_trans.sort_values(by=iterators,
                                 ascending=True, inplace=True)
            df_trans['par_index'] = par_index

            for u, v, data in signaling_steps():
                df_trans['_to_'.join((v, u))] = df_trans[u].div(df_trans[v])

            for node in node_species(df).keys():
                df_trans[node] = df_trans[node] / df_trans[node].max()

            for u, v, data in signaling_steps():
                df_trans['_to_'.join((v, u))] /= data['normfun'](
                    df_trans['_to_'.join((v, u))]
                )

            df_trans['active EGFR_to_phys pERK'] = \
                df_trans['active EGFR_to_RASgtp'] * \
                df_trans['RASgtp_to_phys pMEK'] * \
                df_trans['phys pMEK_to_phys pERK']

            transductions.append(df_trans)
        except FileNotFoundError as err:
            print(f'missing data for index {par_index} ({str(err)}')

    df_transduction = pd.concat(transductions)

    steps = dict({
        '_to_'.join((v, u)): {
            'step': node_order()[v],
            'channel': node_channel()[v]
        }
        for u, v, data in signaling_steps()
    }, **nodes(df))

    df_melt = pd.melt(df_transduction, id_vars=groupvars,
                      value_vars=[col for col in df_transduction.columns
                                  if col not in ['par_index', *groupvars]])

    for var in ['step', 'channel']:
        df_melt[var] = df_melt.variable.apply(
            lambda x: steps.get(x, {'step': 4, 'channel': 'phys'})[var]
        )

    return df_melt


def average_over_par_index(df, groupvars):
    return pd.DataFrame([
        dict(value=values.value.median(), **dict(zip(groupvars, cond)))
        for cond, values in df.groupby(groupvars)
    ])


def write_timestamp(figdir, rfile):
    sttime = datetime.datetime.utcnow().strftime('%Y%m%d_%H:%M:%S - ')
    with open(os.path.join(figdir, rfile), 'a') as file:
        file.write(f'{sttime}\n')
