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
    average_over_par_index, write_timestamp
)

sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

prafi = 'LY3009120'
prafi_0 = f'{prafi}_0'
prafi_label = fr'{prafi.replace("_", "")} [$\mu$M]'

df = read_all_analysis_dataframes(sxs, 'panrafcomboprediction')

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zero = \
    get_obs_df(df, model)

groupvars = ['time', 'LY3009120_0', 'Cobimetinib_0', 'Vemurafenib_0',
             'EGF_0', 'EGFR_crispr', 'variable', 'datatype']
df_sim_melt = average_over_par_index(df_sim_obs, groupvars)
df_melt = pd.concat([df_sim_melt, df_data_obs[groupvars + ['value']]])
df_melt = df_melt[df_melt.time == 8.0]

for drugb in ['Vemurafenib', 'Cobimetinib']:
    drugb_0 = f'{drugb}_0'
    drugb_label = f'{drugb.replace("_", "")} [$\mu$M]'

    df_combo = df_melt.copy()
    if drugb == 'Vemurafenib':
        df_combo = df_combo[
            df_combo['Cobimetinib_0'] == drug_zero['Cobimetinib']
        ]
    else:
        df_combo = df_combo[
            df_combo['Vemurafenib_0'] == drug_zero['Vemurafenib']
        ]

    for obs, zlabel, zlims, cmap, id in zip(
            ['pERK_IF_obs', 'gtpRAS_obs', 'tDUSP_obs'],
            [IFLABEL, RASLABEL, DUSPLABEL],
            [(0, 1.2), (0, 5e3), (0, 1e4)],
            ['viridis', 'inferno', 'plasma'],
            ['pERK', 'RASgtp', 'tDUSP']
    ):
        plot_simdata_heatmap(
            df_combo,
            [obs], drugb_0, drugb_label, prafi_0, prafi_label, zlabel,
            logx=True, logy=True, rows='EGF_0', zlims=zlims, cmap=cmap,
            figdir=figdir,
            filename=f'doseresponse_{prafi}_{drugb}_EGFRa_{id}.pdf',
        )

    df_edata = df[df.datatype == 'data'].copy()
    df_rdata = df[df.datatype == 'simulation'].copy()

    for frame in [df_edata, df_rdata]:
        for drug_0, name in zip([prafi_0, drugb_0], [prafi, drugb]):
            frame.loc[frame[drug_0] == drug_zero[name], drug_0] = 0.0

    for measure in ['bliss', 'hsa']:
        plot_synergies(df_edata, df_rdata, kind=measure,
                       rafi_0=prafi_0, meki_0=drugb_0)
        plt.savefig(os.path.join(
            figdir, f'{prafi}_{drugb}_{measure}_EGFRa.pdf')
        )

    plot_isobolograms(df_edata, df_rdata, rafi_0=prafi_0, meki_0=drugb_0,
                      vmax=1.2)
    plt.savefig(os.path.join(
        figdir, f'{prafi}_{drugb}_isobolograms_EGFRa.pdf')
    )

    write_timestamp(figdir, 'panrafcomboprediction')
