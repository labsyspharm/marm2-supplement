import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

from MARM.visualize import (
    IFLABEL, RASLABEL, DUSPLABEL,
    plot_simdata_heatmap, plot_synergies, plot_isobolograms
)
from MARM.paths import get_figure_dir
from MARM.estimation import get_model
from MARM.analysis import (
    read_settings, get_obs_df, read_all_analysis_dataframes,
    average_over_par_index, get_signal_transduction_df, write_timestamp
)

sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

df = read_all_analysis_dataframes(sxs, 'comboprediction', tps=[0.0833, 8.0])

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zero = \
    get_obs_df(df, model)


rafi = 'Vemurafenib'
meki = 'Cobimetinib'
rafi_0 = f'{rafi}_0'
meki_0 = f'{meki}_0'
rafi_label = f'{rafi.replace("_","")} [$\mu$M]'
meki_label = f'{meki.replace("_","")} [$\mu$M]'

for egfr_state, time, egfr_label in zip(
        sorted(df_data_obs.EGFR_crispr.unique()),
        [0.0833, 8],
        ['EGFRwt', 'EGFRa']
):
    plotspecs = [
        {
            'obs': 'pERK_IF_obs',
            'label': IFLABEL,
            'zlims': (0, 2.0) if egfr_label == 'EGFRwt' else (0, 1.2),
            'cmap': 'viridis',
            'id': 'pERK',
        },
        {
            'obs': 'pERK_onco_obs',
            'label': IFLABEL,
            'zlims': (0, 2.0) if egfr_label == 'EGFRwt' else (0, 1.2),
            'cmap': 'viridis',
            'id': 'pERK_onc',
        },
        {
            'obs': 'pERK_phys_obs',
            'label': IFLABEL,
            'zlims': (0, 2.0) if egfr_label == 'EGFRwt' else (0, 1.2),
            'cmap': 'viridis',
            'id': 'pERK_phys',
        },
        {
            'obs': 'pMEK_IF_obs',
            'label': IFLABEL,
            'zlims': (0, 2),
            'cmap': 'plasma',
            'id': 'pMEK',
        },
        {
            'obs': 'pMEK_onco_obs',
            'label': IFLABEL,
            'zlims': (0, 2),
            'cmap': 'plasma',
            'id': 'pMEK_onc',
        },
        {
            'obs': 'pMEK_phys_obs',
            'label': IFLABEL,
            'zlims': (0, 2),
            'cmap': 'plasma',
            'id': 'pMEK_phys',
        },
        {
            'obs': 'gtpRAS_obs',
            'label': RASLABEL,
            'zlims': (0, 10),
            'cmap': 'inferno',
            'id': 'RASgtp',
        },
        {
            'obs': 'tDUSP_obs',
            'label': DUSPLABEL,
            'zlims': (0, 10),
            'cmap': 'plasma',
            'id': 'tDUSP',
        },
    ]

    if egfr_label == 'EGFRwt':
        mode = 'peak'
    else:
        mode = 'int_log10'

    df_gain = get_signal_transduction_df(
        sxs,
        'comboprediction',
        lambda frame: (frame.EGFR_crispr == egfr_state) &
                      (frame.EGF_0 == 100) &
                      (frame.datatype == 'simulation'),
        [rafi_0, meki_0],
        mode
    )
    df_gain = average_over_par_index(
        df_gain, [c for c in df_gain.columns if c != 'value']
    )
    for gain in [var for var in df_gain.variable.unique()
                 if len(var.split('_to_')) == 2]:
        plot_simdata_heatmap(
            df_gain[df_gain.variable == gain],
            [gain],
            meki_0, meki_label, rafi_0, rafi_label, gain,
            logx=True, logy=True, rows='EGF_0', zlims=(0, 2),
            cmap='PRGn',
            figdir=figdir,
            filename=f'doseresponse_{rafi}_{meki}_{egfr_label}'
                     f'_transduction_{gain}.pdf',
        )

    del df_gain

    def condition_filter(frame):
        return frame[
            (frame['EGFR_crispr'] == egfr_state) &
            (frame['time'] == time)
        ]

    groupvars = [rafi_0, meki_0, 'EGF_0', 'EGFR_crispr', 'variable',
                 'datatype', 'time']
    df_sim_melt = average_over_par_index(condition_filter(df_sim_obs),
                                         groupvars)
    df_melt = pd.concat([df_sim_melt,
                         condition_filter(df_data_obs)[groupvars + ['value']]])
    del df_sim_melt
    for plotspec in plotspecs:
        #if egfr_label == 'EGFRa' and plotspec['obs'].startswith('pERK'):
        #    plotspec['zlims'] = (0, 1.2)

        plot_simdata_heatmap(
            df_melt[df_melt.variable == plotspec['obs']],
            [plotspec['obs']],
            meki_0, meki_label, rafi_0, rafi_label,
            plotspec['label'],
            logx=True, logy=True, rows='EGF_0', zlims=plotspec['zlims'],
            cmap=plotspec['cmap'],
            figdir=figdir,
            filename=f'doseresponse_{rafi}_{meki}_{egfr_label}'
                     f'_{plotspec["id"]}.pdf',
        )
    del df_melt

    df_cond = condition_filter(df)
    df_edata = df_cond[df_cond.datatype == 'data']
    df_rdata = df_cond[df_cond.datatype == 'simulation']

    for frame in [df_edata, df_rdata]:
        for drug_0, name in zip([rafi_0, meki_0], [rafi, meki]):
            frame.loc[frame[drug_0] == drug_zero[name], drug_0] = 0.0

    plot_isobolograms(df_edata, df_rdata, egf=0.0, time=time,
                      rafi_0=rafi_0, meki_0=meki_0,
                      vmax=1.2 if egfr_label == 'EGFRa' else 2.0)
    plt.savefig(os.path.join(
        figdir, f'doseresponse_{rafi}_{meki}_{egfr_label}'
                f'_isobolograms_woEGF.pdf'
        )
    )
    plot_isobolograms(df_edata, df_rdata, egf=100, time=time,
                      rafi_0=rafi_0, meki_0=meki_0,
                      vmax=1.2 if egfr_label == 'EGFRa' else 2.0)
    plt.savefig(os.path.join(
        figdir, f'doseresponse_{rafi}_{meki}_{egfr_label}'
                f'_isobologram_wEGF.pdf'
        )
    )
    del df_edata
    del df_rdata

df_edata = df[df.datatype == 'data']
df_rdata = df[df.datatype == 'simulation']

for frame in [df_edata, df_rdata]:
    for drug_0, name in zip([rafi_0, meki_0], [rafi, meki]):
        frame.loc[frame[drug_0] == drug_zero[name], drug_0] = 0.0

for measure in ['bliss', 'hsa']:
    for marker in ['pERK_IF_obs', 'pERK_onco_obs', 'pERK_phys_obs']:
        plot_synergies(
            df_edata, df_rdata,
            kind='bliss_nn' if measure == 'bliss' and marker == 'pERK_phys_obs'
            else measure,
            rafi_0=rafi_0, meki_0=meki_0, marker=marker)
        plt.savefig(os.path.join(
            figdir, f'doseresponse_{rafi}_{meki}_{measure}_{marker}.pdf'
        ))

write_timestamp(figdir, 'comboprediction')
