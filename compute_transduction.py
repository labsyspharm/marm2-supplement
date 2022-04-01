from MARM.visualize.common import get_pysb_parameters
from MARM.analysis import (
    write_analysis_dataframe, process_rdata, read_settings, load_model_solver,
    load_model_aux, rename_and_fill_drug_columns
)

import itertools
import amici
import pandas as pd
import numpy as np
import sys

settings = read_settings(sys.argv)

drug_name = sys.argv[6]
perturbations = ''

drugs = {
    'Vemurafenib': {
        'conc': 'RAFi_0',
        'range': np.logspace(-4, 1, 19),
        'type': 'rafi'
    },
    'Dabrafenib': {
        'conc': 'RAFi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'rafi'
    },
    'AZ_628': {
        'conc': 'RAFi_0',
        'range': np.logspace(-4, 1, 19),
        'type': 'rafi'
    },
    'PLX8394': {
        'conc': 'RAFi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'rafi'
    },
    'LY3009120': {
        'conc': 'RAFi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'rafi'
    },
    'Cobimetinib': {
        'conc': 'MEKi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'meki'
    },
    'Trametinib': {
        'conc': 'MEKi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'meki'
    },
    'Binimetinib': {
        'conc': 'MEKi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'meki'
    },
    'Selumetinib': {
        'conc': 'MEKi_0',
        'range': np.logspace(-4, 1, 19),
        'type': 'meki'
    },
    'PD0325901': {
        'conc': 'MEKi_0',
        'range': np.logspace(-5, 0, 19),
        'type': 'meki'
    }
}

drug = drugs[drug_name]
par, model, solver, full_name = load_model_solver(
    settings, **{drug['type']: drug_name}, mods='channel_monoobs'
)
pysb_model, observable = load_model_aux(full_name)


model.setTimepoints([0] + list(np.logspace(-4, 1, 201)))
model.setFixedParameterByName('EGF_0', 100.0)
if 'MEKi_0' in model.getFixedParameterNames():
    model.setFixedParameterByName('MEKi_0', 0.0)
if 'RAFi_0' in model.getFixedParameterNames():
    model.setFixedParameterByName('RAFi_0', 0.0)
model.setFixedParameterByName('EGFR_crispr', 1.0)

edata = amici.ExpData(model.get())
edata.reinitializeFixedParameterInitialStates = True
edata.t_presim = 0.0
fp = list(edata.fixedParameters)
fp[model.getFixedParameterNames().index('EGF_0')] = 0.0
edata.fixedParametersPreequilibration = fp

edatas = []
for conc, egfr in itertools.product(drug['range'], ['a', 'wt']):
    egfr_crispr = 1.0 if egfr == 'wt' else 10.0
    edata_drug = amici.ExpData(edata)
    fp = list(edata_drug.fixedParameters)
    fp[model.getFixedParameterNames().index(drug['conc'])] = conc
    fp[model.getFixedParameterNames().index('EGFR_crispr')] = egfr_crispr
    edata_drug.fixedParameters = fp

    fp_preeq = list(edata_drug.fixedParametersPreequilibration)
    fp_preeq[model.getFixedParameterNames().index(drug['conc'])] = conc
    fp_preeq[model.getFixedParameterNames().index('EGFR_crispr')] = egfr_crispr
    edata_drug.fixedParametersPreequilibration = fp_preeq

    edatas.append(edata_drug)

rdatas = amici.runAmiciSimulations(model, solver, edatas,
                                   num_threads=settings['threads'])

# load parameter for pysb
p = get_pysb_parameters(model, pysb_model)

df = pd.concat([
    amici.getSimulationObservablesAsDataFrame(model, edatas, rdatas),
    process_rdata(rdatas, observable, 'monomer', p)
], axis=1)

rename_and_fill_drug_columns(df, **{drug['type']: drug_name})

df.drop(columns=[
    col for col in df.columns
    if col.startswith('expl') and not (
        'channelmbraf' in col or
        'channeldbraf' in col or
        'channelcraf' in col or
        'Tyrp' in col
    )
], inplace=True)

write_analysis_dataframe(
    df, settings, f'transduction__{drug_name}__{perturbations}'
)
