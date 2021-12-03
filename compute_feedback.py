from MARM.analysis import (
    write_analysis_dataframe, read_settings, load_model_solver,
    load_model_aux, rename_and_fill_drug_columns
)

import amici
import sys
import pandas as pd
import numpy as np

settings = read_settings(sys.argv)

par, model, solver, full_name = load_model_solver(
    settings, rafi='Vemurafenib', mods='channel_monoobs'
)

pysb_model, observable = load_model_aux(full_name)

model.setFixedParameterByName('EGF_0', 100.0)
model.setFixedParameterByName('RAFi_0', 10.0)
model.setFixedParameterByName('EGFR_crispr', 1.0)

edata_ref = amici.ExpData(model.get())
edata_ref.reinitializeFixedParameterInitialStates = True
edata_ref.t_presim = 0.0
fp = list(edata_ref.fixedParameters)
fp[model.getFixedParameterNames().index('EGF_0')] = 0.0
edata_ref.fixedParametersPreequilibration = fp

for cond in ['preequilibration', 'observed', 'log', 'egfra']:
    edata = amici.ExpData(edata_ref)

    if cond == 'preequilibration':
        edata.setTimepoints(np.logspace(-7, 5, 51))

        fp = list(edata.fixedParameters)
        fp[model.getFixedParameterNames().index('EGF_0')] = 0.0
        edata.fixedParameters = fp

        fp = list(edata.fixedParametersPreequilibration)
        fp[model.getFixedParameterNames().index('RAFi_0')] = 0.0
        edata.fixedParametersPreequilibration = fp

    if cond == 'observed':
        edata.setTimepoints(np.linspace(0, 2, 51))

    if cond == 'log':
        edata.setTimepoints(np.logspace(-7, 2, 51))

    if cond == 'egfra':
        model.setFixedParameterByName('EGFR_crispr', 9.19)
        edata.setTimepoints(np.linspace(0, 8, 51))

    edatas = [edata]

    # run simulations
    rdatas = amici.runAmiciSimulations(model, solver, edatas,
                                       num_threads=settings['threads'])

    df = pd.concat([
        amici.getSimulationObservablesAsDataFrame(model, edatas, rdatas),
    ], axis=1)
    rename_and_fill_drug_columns(df, rafi='Vemurafenib')
    write_analysis_dataframe(
        df, settings, f'feedback_analysis_{cond}'
    )
