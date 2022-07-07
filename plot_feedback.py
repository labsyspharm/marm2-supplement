import pandas as pd
import numpy as np
import sys

from plotnine import *

from MARM.visualize.ggplot import plot_and_save_fig, PLOTNINE_FIGWIDTH
from MARM.estimation import get_model
from MARM.analysis import (
    read_settings, read_analysis_dataframe, write_timestamp, N_RUNS
)
from MARM.paths import get_figure_dir


def remove_prefix(x, prefix):
    return x.rename(lambda y: y.replace(prefix, ''), axis=0)


sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

prot_obs = ['tDUSP_obs', 'tSPRY_obs', 'tEGFR_obs']
mrna_obs = ['tmDUSP_obs', 'tmSPRY_obs', 'tmEGFR_obs']
abundance_obs = mrna_obs + prot_obs


dfs = []
for condition in ['observed', 'preequilibration', 'log', 'egfra',
                  'egfra_long']:
    df = read_analysis_dataframe(sxs, f'feedback_analysis_{condition}', 0)

    abundance_entries = []
    for index in range(N_RUNS):
        try:
            df = read_analysis_dataframe(sxs,
                                         f'feedback_analysis_{condition}', index)
            df['par_index'] = index
            abundance_entries += [
                df[
                    abundance_obs + ['pERK_IF_obs', 'Vemurafenib_0', 'time', 'par_index']
                ]
            ]
        except FileNotFoundError:
            print(f'missing data for index {index}')

    abundances = pd.concat(abundance_entries, ignore_index=True)

    df = pd.melt(abundances,
                 id_vars=['time', 'par_index', 'pERK_IF_obs', 'Vemurafenib_0'],
                 value_vars=abundance_obs)

    df.variable = pd.Categorical(
        df.variable, ordered=True, categories=[
            'tDUSP_obs', 'tmDUSP_obs', 'tSPRY_obs', 'tmSPRY_obs',
            'tEGFR_obs', 'tmEGFR_obs'
    ])
    df.value = df.value.apply(np.exp)

    plot = (
        ggplot(df[df.Vemurafenib_0 == 1.0],
               aes(x='time', y='value', group='variable', color='variable',
                   fill='variable'))
        + stat_summary(fun_y=np.median, geom='line', size=1)
        + xlab('time [h]')
        + ylab('abundance [molecule/cell]/[FPKM]')
        + theme('minimal', figure_size=(PLOTNINE_FIGWIDTH / 3,
                                        PLOTNINE_FIGWIDTH / 3 * 2 / 3))
        + scale_color_cmap_d(name='tab20')
        + scale_fill_cmap_d(name='tab20')
        + scale_y_log10(limits=(1e0, 5e5))
    )



    for quantile_range in np.linspace(0.2, 0.8, 4):
        up = 0.5 + quantile_range / 2
        lp = 0.5 - quantile_range / 2
        plot += stat_summary(fun_y=np.median,
                             fun_ymin=lambda x: np.quantile(x, lp),
                             fun_ymax=lambda x: np.quantile(x, up),
                             geom='ribbon', alpha=0.2, color=None)

    if condition == 'preequilibration':
        plot += scale_x_log10(limits=(1e-1, 1e2))
    elif condition not in ['observed', 'egfra']:
        plot += scale_x_log10()

    plot_and_save_fig(plot, figdir,
                      f'feedback_abundances_{condition}.pdf')

    df['condition'] = condition
    if condition == 'egfra':
        dfs.append(df[df.time == 8.0])
    if condition == 'preequilibration':
        dfs.append(df[df.time == df.time.max()])
    if condition == 'observed':
        tt = df.time.unique()
        five_min = tt[np.argmin(np.abs(tt - 0.083))]
        dfs.append(df[df.time == five_min])

df_all = pd.concat(dfs)
df_all.condition = pd.Categorical(df_all.condition)
df_all.variable = pd.Categorical(
    df_all.variable, ordered=True, categories=prot_obs + mrna_obs
)
for condition in df_all.condition.unique():
    lb_pERK, ub_pERK = df_all[df_all.condition == condition].pERK_IF_obs.quantile([0.01,0.99])
    print(f'{condition}: {lb_pERK} - {ub_pERK} pERK')
    df_all = df_all[(df_all.condition != condition) | ((df_all.pERK_IF_obs > lb_pERK) & (df_all.pERK_IF_obs < ub_pERK))]


n_bins = 20
plot = (
        ggplot(df_all,
               aes(x='pERK_IF_obs', y='value', group='condition',
                   color='condition', fill='condition'))
        + facet_wrap('variable', ncol=3, scales='free_y')
        + stat_summary_bin(fun_y=np.median, geom='line', size=1,
                           bins=n_bins)
        + xlab('pERK level [au]')
        + ylab('expression level [molecule/cell]')
        + theme('minimal', figure_size=(PLOTNINE_FIGWIDTH ,
                                        PLOTNINE_FIGWIDTH  * 2/3))
        + scale_color_cmap_d(name='Dark2')
        + scale_fill_cmap_d(name='Dark2')
        + scale_y_log10()
)

for quantile_range in np.linspace(0.2, 0.8, 4):
    up = 0.5 + quantile_range / 2
    lp = 0.5 - quantile_range / 2
    plot += stat_summary_bin(fun_y=np.median,
                             fun_ymin=lambda x: np.quantile(x, lp),
                             fun_ymax=lambda x: np.quantile(x, up),
                             geom='ribbon', alpha=0.2, color=None,
                             bins=n_bins)

plot_and_save_fig(plot, figdir,
                  f'feedback_pERK_phase_diagram.pdf')

write_timestamp(figdir, 'feedback')