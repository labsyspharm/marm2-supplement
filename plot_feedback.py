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


for condition in ['observed', 'preequilibration', 'log', 'egfra']:
    df = read_analysis_dataframe(sxs, f'feedback_analysis_{condition}', 0)

    abundance_entries = []
    for index in range(N_RUNS):
        try:
            df = read_analysis_dataframe(sxs,
                                         f'feedback_analysis_{condition}', index)
            times = [time for time in df.time.unique()]
            abundance_entries += [
                df[(df.time == time)][
                    abundance_obs
                ].max().append(pd.Series({'time': time, 'par_index': index}))
                for time in times
            ]
        except FileNotFoundError:
            print(f'missing data for index {index}')

    abundances = pd.DataFrame(abundance_entries)
    df = pd.melt(abundances,
                 id_vars=['time', 'par_index'],
                 value_vars=abundance_obs)

    df.variable = pd.Categorical(
        df.variable, ordered=True, categories=[
            'tDUSP_obs', 'tmDUSP_obs', 'tSPRY_obs', 'tmSPRY_obs',
            'tEGFR_obs', 'tmEGFR_obs'
    ])
    df.value = df.value.apply(np.exp)

    print(df)
    plot = (
        ggplot(df,
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

    if condition not in ['observed', 'egfra']:
        plot += scale_x_log10()

    plot_and_save_fig(plot, figdir,
                      f'feedback_abundances_{condition}.pdf')

write_timestamp(figdir, 'feedback')
