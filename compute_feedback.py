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

pysb_model, _ = load_model_aux(full_name)

model.setFixedParameterByName('EGF_0', 100.0)
model.setFixedParameterByName('RAFi_0', 10.0)
model.setFixedParameterByName('EGFR_crispr', 1.0)

edata_ref = amici.ExpData(model.get())

edata_ref.reinitialization_state_idxs_sim = [
    model.getStateNames().index(state_name)
    for state_name in [
        'PRAFi(raf=None) ** CP',
        'RAFi(raf=None) ** CP',
        'MEKi(mek=None) ** CP',
        'EGF(rtk=None) ** CP'
    ]
    if state_name in model.getStateNames()
]


edata_ref.t_presim = 0.0
fp = list(edata_ref.fixedParameters)
fp[model.getFixedParameterNames().index('EGF_0')] = 0.0
edata_ref.fixedParametersPreequilibration = fp

for cond in ['preequilibration', 'observed', 'log', 'egfra', 'egfra_long']:
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
        fp = list(edata.fixedParameters)
        fp[model.getFixedParameterNames().index('EGFR_crispr')] = 9.19
        edata.fixedParameters = fp

        fp = list(edata.fixedParametersPreequilibration)
        fp[model.getFixedParameterNames().index('EGFR_crispr')] = 9.19
        edata.fixedParametersPreequilibration = fp

        edata.setTimepoints(np.linspace(0, 8, 51))

    if cond == 'egfra_long':
        fp = list(edata.fixedParameters)
        fp[model.getFixedParameterNames().index('EGFR_crispr')] = 9.19
        edata.fixedParameters = fp

        fp = list(edata.fixedParametersPreequilibration)
        fp[model.getFixedParameterNames().index('EGFR_crispr')] = 9.19
        edata.fixedParametersPreequilibration = fp
        edata.setTimepoints(np.logspace(-7, 2, 51))

    rafis = np.logspace(-4, 1, 51)
    edatas = [
        amici.ExpData(edata) for rafi in rafis
    ]
    for rafi, edata in zip(rafis, edatas):
        fp = list(edata.fixedParameters)
        fp[model.getFixedParameterNames().index('RAFi_0')] = rafi
        edata.fixedParameters = fp

        if cond != 'preequilibration':
            fp = list(edata.fixedParametersPreequilibration)
            fp[model.getFixedParameterNames().index('RAFi_0')] = rafi
            edata.fixedParametersPreequilibration = fp

    # run simulations
    print(f'Running simulations for condition {cond}')
    rdatas = amici.runAmiciSimulations(model, solver, edatas,
                                       num_threads=settings['threads'])

    df = pd.concat([
        amici.getSimulationObservablesAsDataFrame(model, edatas, rdatas),
    ], axis=1)
    rename_and_fill_drug_columns(df, rafi='Vemurafenib')
    write_analysis_dataframe(
        df, settings, f'feedback_analysis_{cond}'
    )
