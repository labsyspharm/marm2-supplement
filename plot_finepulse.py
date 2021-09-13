import sys
import numpy as np

from MARM.estimation import get_model
from MARM.analysis import (
    read_settings, get_obs_df, read_all_analysis_dataframes,
    extend_drug_adapted, write_timestamp
)
from MARM.paths import get_figure_dir
from MARM.visualize import (
    IFLABEL, TIMELABEL,
    plot_simdata_grid
)

sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

df = read_all_analysis_dataframes(sxs, 'finepulse')
df = extend_drug_adapted(df, 0.2)

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zero = \
    get_obs_df(df, model)

for drug, zero_drug in zip(['Vemurafenib', 'Cobimetinib'],
                           ['Cobimetinib', 'Vemurafenib']):

    def filter_tc(df_filter):
        return df_filter[
            (df_filter['time'] <= 2.0) &
            (df_filter['t_presim'] == 0.0) &
            (df_filter['EGFR_crispr'] == 1.0) &
            (df_filter['EGF_0'] == 100.0) &
            (df_filter[f'{zero_drug}_0'] == drug_zero[zero_drug]) &
            (df_filter[f'{drug}_0'].apply(
                lambda x: np.mod(np.round(np.log10(x), 1)*10, 5)
            ) == 0.0) &
            (df_filter[f'{drug}_0'] > 0)
        ]

    plot_simdata_grid(
        filter_tc(df_data_obs), filter_tc(df_sim_obs),
        if_obs, 'time', TIMELABEL, IFLABEL,
        ylimits=(-.2, 2), logx=False, logy=False, logz=True,
        group=f'{drug}_0', color=f'{drug}_0',
        expand=False,
        figdir=figdir,
        filename=f'timecourse_wEGF_dr{drug}_tc.pdf',
    )

write_timestamp(figdir, 'finepulse')
