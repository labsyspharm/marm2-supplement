import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from MARM.estimation import get_model, RAFI, MEKI, PANRAFI
from MARM.analysis import (
    read_settings, get_obs_df, read_all_analysis_dataframes,
    write_timestamp, get_analysis_results_file, N_RUNS,
    average_over_par_index
)
from MARM.paths import get_figure_dir
from MARM.visualize import (
    IFLABEL, plot_simdata_grid, plot_simdatadecomp_grid, plot_simdata_heatmap,
    plot_synergies
)

sxs = read_settings(sys.argv, index=False, threads=False)

model = get_model(sxs['model_name'], 'nrasq61mut', sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

cell_line = 'engineered'
YLIM = (-0.15, 2.25)

df = read_all_analysis_dataframes(sxs, f'mutRASprediction_{cell_line}')

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zeros = \
    get_obs_df(df, model)

for obs in if_obs:
    for index in range(N_RUNS):
        obs_name = obs.split('_')[0]
        offset = df_sim_obs[(df_sim_obs.variable == f'{obs_name}_onco_obs') &
                            (df_sim_obs.datatype == 'simulation') &
                            (df_sim_obs.par_index == index)]
        if len(offset):
            par_file = get_analysis_results_file(sxs['model_name'],
                                                 sxs['variant'],
                                                 sxs['dataset'],
                                                 f'mutRASpars_{cell_line}',
                                                 index)
            par = pd.read_csv(par_file, index_col=0)

            offset = df_sim_obs[
                (df_sim_obs.variable == f'{obs_name}_onco_obs') &
                (df_sim_obs.datatype == 'simulation') &
                (df_sim_obs.par_index == index)
            ]
            offset.value = par.loc[index, f'{obs_name}_IF_offset']
            offset.variable = f'{obs_name}_background_obs'
            df_sim_obs = pd.concat([df_sim_obs, offset], ignore_index=True)


def apply_filters(df_filter, generic, specific):
    return df_filter[generic(df_filter) & specific(df_filter)]

filter_generic = lambda f: (f['time'] == 0.0) & \
                           (f['LY3009120_0'] == 0.0)

for drug in ['Vemurafenib', 'Dabrafenib', 'Selumetinib', 'Binimetinib',
             'Trametinib', 'Cobimetinib', 'AZ_628']:

    label = rafi_label = f'{drug.replace("_", "")} [$\mu$M]'

    filter_specific = lambda f: pd.concat([
        (f[f'{zdrug}_0'] == drug_zeros[zdrug])
        for zdrug in RAFI + PANRAFI + MEKI
        if zdrug != drug and zdrug in drug_zeros
    ], axis=1).all(axis=1)

    plot_simdata_grid(
        df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
        df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
        obs=if_obs, xindex=f'{drug}_0', xlabel=label, ylabel=IFLABEL,
        ylimits=YLIM, logx=True, logy=False, group='NRAS_Q61mut',
        color='NRAS_Q61mut',
        figdir=figdir, ec50=True, ecmax=True,
        filename=f'prediction_NRASmut_{cell_line}_dr{drug}.pdf',
    )

    filter_specific = lambda f: pd.concat([
        (f[f'{zdrug}_0'] == drug_zeros[zdrug])
        for zdrug in RAFI + PANRAFI + MEKI
        if zdrug != drug and zdrug in drug_zeros
    ], axis=1).all(axis=1) & (f['NRAS_Q61mut'] == 1.0)

    plot_simdatadecomp_grid(
        apply_filters(df_data_obs, filter_generic, filter_specific),
        apply_filters(df_sim_obs, filter_generic, filter_specific),
        if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
        xindex=f'{drug}_0', xlabel=label, ylabel=IFLABEL,
        ylimits=YLIM, logx=True, logy=False,
        figdir=figdir,
        filename=f'prediction_NRASmut_{cell_line}_dr{drug}_decomp.pdf',
    )

filter_specific = lambda f: \
    (f['Vemurafenib_0'] == 1.0) & (f['Cobimetinib_0'] > 0)

plot_simdata_grid(
    df_data=apply_filters(df_data_obs, filter_generic,
                          filter_specific),
    df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
    obs=if_obs, xindex=f'Cobimetinib_0', xlabel=label, ylabel=IFLABEL,
    ylimits=YLIM, logx=True, logy=False, group='NRAS_Q61mut',
    color='NRAS_Q61mut',
    figdir=figdir,
    filename=f'prediction_NRASmu_{cell_line}_drComibimetinib_'
             f'Vemurafenib1.0.pdf',
)

filter_specific = lambda f: \
    (f['Vemurafenib_0'] == 1.0) & (f['Cobimetinib_0'] > 0) \
    & (f['NRAS_Q61mut'] == 1.0)

plot_simdatadecomp_grid(
    apply_filters(df_data_obs, filter_generic, filter_specific),
    apply_filters(df_sim_obs, filter_generic, filter_specific),
    if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
    xindex=f'Cobimetinib_0', xlabel=label, ylabel=IFLABEL,
    ylimits=YLIM, logx=True, logy=False,
    figdir=figdir,
    filename=f'prediction_NRASmut_{cell_line}_drComibimetinib_'
             f'Vemurafenib1.0_decomp.pdf',
)

df = read_all_analysis_dataframes(sxs,
                                  f'mutRASprediction_{cell_line}_combo')

df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zeros = \
    get_obs_df(df, model)

groupvars = ['time', 'Vemurafenib_0', 'Cobimetinib_0', 'LY3009120_0',
             'variable', 'datatype', 'NRAS_Q61mut']

df_sim_melt = average_over_par_index(df_sim_obs, groupvars)
df_melt = pd.concat([df_sim_melt, df_data_obs[groupvars + ['value']]])

drugb = 'Cobimetinib'
drugb_0 = f'{drugb}_0'
drugb_label = fr'{drugb.replace("_", "")} [$\mu$M]'

for druga in ['Vemurafenib', 'LY3009120']:
    druga_0 = f'{druga}_0'
    druga_label = f'{druga.replace("_", "")} [$\mu$M]'

    if druga == 'Vemurafenib':
        fixdrug = 'LY3009120'
        drugfix = drug_zeros[fixdrug]
    else:
        fixdrug = 'Vemurafenib'
        drugfix = 1.0

    for marker in ['pERK_IF_obs', 'pERK_onco_obs', 'pERK_phys_obs']:
        plot_simdata_heatmap(
            df_melt[df_melt[f'{fixdrug }_0'] == drugfix],
            [marker], drugb_0, drugb_label, druga_0, druga_label, IFLABEL,
            logx=True, logy=True, rows='NRAS_Q61mut', zlims=(0, 1.2),
            cmap='viridis', figdir=figdir,
            filename=f'doseresponse_{druga}_{drugb}_NRASQ61K_{marker}.pdf',
        )

    df_edata = df[(df.datatype == 'data') &
                  (df[f'{fixdrug}_0'] == drugfix)]
    df_rdata = df[(df.datatype == 'simulation') &
                  (df[f'{fixdrug}_0'] == drugfix)]

    for frame in [df_edata, df_rdata]:
        for drug_0, name in zip([druga_0, drugb_0], [druga, drugb]):
            frame.loc[frame[drug_0] == drug_zeros[name], drug_0] = 0.0

    cobi_concs = [
        val
        for val, count in df_rdata['Cobimetinib_0'].value_counts().items()
        if count > 50
    ]
    df_rdata = df_rdata[
        df_rdata['Cobimetinib_0'].apply(lambda x: x in cobi_concs)
    ]
    df_edata = df_edata[
        df_edata['Cobimetinib_0'].apply(lambda x: x in cobi_concs)
    ]

    for measure in ['bliss', 'hsa']:
        plot_synergies(df_edata, df_rdata, kind=measure,
                       rafi_0=druga_0, meki_0=drugb_0, mode='nras')
        plt.savefig(os.path.join(
            figdir, f'{druga}_{drugb}_{measure}_NRASQ61K.pdf')
        )

write_timestamp(figdir, f'mutRASprediction')
