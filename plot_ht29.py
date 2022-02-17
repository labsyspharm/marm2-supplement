import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

from MARM.visualize import (
    IFLABEL, plot_simdata_heatmap, plot_synergies, plot_isobolograms
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

rafi = 'Vemurafenib'
meki = 'Cobimetinib'
rafi_0 = f'{rafi}_0'
meki_0 = f'{meki}_0'
rafi_label = f'{rafi.replace("_","")} [$\mu$M]'
meki_label = f'{meki.replace("_","")} [$\mu$M]'

df = read_all_analysis_dataframes(sxs, 'ht29')

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zero = \
    get_obs_df(df, model)

groupvars = ['time', rafi_0, meki_0, 'EGF_0', 'EGFR_crispr', 'variable',
             'datatype']
df_sim_melt = average_over_par_index(df_sim_obs, groupvars)
df_melt = pd.concat([df_sim_melt, df_data_obs[groupvars + ['value']]])

for marker in ['pERK_IF_obs', 'pERK_onco_obs', 'pERK_phys_obs']:
    plot_simdata_heatmap(
        df_melt,
        [marker], meki_0, meki_label, rafi_0, rafi_label, IFLABEL,
        logx=True, logy=True, rows='EGF_0', zlims=(0, 1.2), cmap='viridis',
        figdir=figdir,
        filename=f'doseresponse_Vemurafenib_Cobimetinib_ht29_{marker}.pdf',
    )

df_edata = df[df.datatype == 'data']
df_rdata = df[df.datatype == 'simulation']

for frame in [df_edata, df_rdata]:
    for drug_0, name in zip([rafi_0, meki_0], [rafi, meki]):
        frame.loc[frame[drug_0] == drug_zero[name], drug_0] = 0.0

for measure in ['bliss', 'hsa']:
    plot_synergies(df_edata, df_rdata, kind=measure,
                   rafi_0=rafi_0, meki_0=meki_0)
    plt.savefig(os.path.join(
        figdir, f'Vemurafenib_Cobimetinib_{measure}_ht29.pdf')
    )

plot_isobolograms(df_edata, df_rdata, rafi_0=rafi_0, meki_0=meki_0, vmax=1.2)
plt.savefig(os.path.join(
    figdir, f'Vemurafenib_Cobimetinib_isobolograms_ht29.pdf')
)

write_timestamp(figdir, 'ht29')
