import sys
import pandas as pd
import numpy as np

from MARM.estimation import get_model, RAFI, MEKI, PANRAFI
from MARM.analysis import (
    read_settings, get_obs_df, read_all_analysis_dataframes, load_parameters,
    write_timestamp, N_RUNS
)
from MARM.paths import get_figure_dir
from MARM.visualize import (
    IFLABEL, TIMELABEL,
    plot_simdata_grid, plot_simdatadecomp_grid, plot_simdata_wrap
)

sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

YLIM = (-0.15, 2.25)

for suffix in '', '_cf':
    df = read_all_analysis_dataframes(sxs, 'trainingdata' + suffix)

    df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zeros = \
        get_obs_df(df, model)

    matches = 0
    for ir, row in df_data_obs.iterrows():
        if np.isnan(row.value):
            continue
        conds = [
            'Vemurafenib_0', 'Vemurafenib_0_preeq', 'Vemurafenib_0_presim',
            'Dabrafenib_0', 'Dabrafenib_0_preeq', 'Dabrafenib_0_presim',
            'PLX8394_0', 'PLX8394_0_preeq', 'PLX8394_0_presim', 'LY3009120_0',
            'LY3009120_0_preeq', 'LY3009120_0_presim', 'AZ_628_0',
            'AZ_628_0_preeq', 'AZ_628_0_presim', 'Cobimetinib_0',
            'Cobimetinib_0_preeq',
            'Cobimetinib_0_presim', 'Trametinib_0', 'Trametinib_0_preeq',
            'Trametinib_0_presim', 'Selumetinib_0', 'Selumetinib_0_preeq',
            'Selumetinib_0_presim', 'Binimetinib_0', 'Binimetinib_0_preeq',
            'Binimetinib_0_presim', 'PD0325901_0', 'PD0325901_0_preeq',
            'PD0325901_0_presim', 'EGF_0', 'EGFR_crispr', 't_presim', 'time'
        ]
        filter = df_sim_obs.variable == row.variable
        for cond in conds:
            filter = np.logical_and(filter, df_sim_obs[cond] == row[cond])
        x = df_sim_obs.loc[filter, 'value']
        matches += (row.ymin < np.quantile(x, 0.1)) and \
                   (np.quantile(x, 0.9) < row.ymax)
    print(str(matches/sum(np.logical_not(np.isnan(df_data_obs.value)))))

    for obs in if_obs:
        for index in range(N_RUNS):
            par = load_parameters(model, sxs, rafi=None, meki=None, prafi=None,
                                  index=index, allow_missing_pars=True)
            obs_name = obs.split('_')[0]
            offset = df_sim_obs[
                (df_sim_obs.variable == f'{obs_name}_onco_obs') &
                (df_sim_obs.datatype == 'simulation') &
                (df_sim_obs.par_index == index)
            ]
            offset.value = model.getParameterByName(f'{obs_name}_IF_offset')
            offset.variable = f'{obs_name}_background_obs'
            df_sim_obs = pd.concat([df_sim_obs, offset], ignore_index=True)


    def apply_filters(df_filter, generic, specific):
        return df_filter[generic(df_filter) & specific(df_filter)]


    for drug in MEKI + RAFI + PANRAFI:
        label = rafi_label = f'{drug.replace("_","")} [$\mu$M]'

        def filter_generic(df_filter):
            return (
                    (df_filter['t_presim'] == 0) &
                    (df_filter['EGFR_crispr'] == 1.0) &
                    pd.concat([
                        (df_filter[f'{zdrug}_0'] == drug_zeros[zdrug])
                        for zdrug in RAFI + MEKI + PANRAFI
                        if zdrug != drug and zdrug in drug_zeros
                    ], axis=1).all(axis=1)
            )

        if drug not in PANRAFI:
            def filter_specific(df_filter):
                return ((df_filter['time'] == 0.0833) |
                        (df_filter['time'] == 0.083)) \
                        & (df_filter['EGF_0'] == 100.0)
            plot_simdatadecomp_grid(
                apply_filters(df_data_obs, filter_generic, filter_specific),
                apply_filters(df_sim_obs, filter_generic, filter_specific),
                if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
                f'{drug}_0', label, IFLABEL,
                ylimits=YLIM, logx=True, logy=False,
                figdir=figdir,
                filename=f'training{suffix}_wEGF_dr{drug}_decomp.pdf',
            )

            def filter_specific(df_filter):
                return (df_filter['time'] >= 0.0) & \
                       (df_filter['EGF_0'] == 100.0) & \
                       (((df_filter[f'{drug}_0'] != 1.0) |
                         (df_filter['time'] == 0.0833) |
                         (df_filter['time'] == 0.5) |
                         (df_filter['time'] == 2) |
                         (df_filter['time'] == 8) |
                         (df_filter['time'] == 24))
                        | (drug != 'Vemurafenib'))
            plot_simdata_grid(
                apply_filters(df_data_obs, filter_generic, filter_specific),
                apply_filters(df_sim_obs, filter_generic, filter_specific),
                if_obs, f'{drug}_0', label, IFLABEL,
                ylimits=YLIM, logx=True, logy=False, group='time',
                color='time', figdir=figdir,
                filename=f'training{suffix}_wEGF_tcdr{drug}.pdf',
            )

        def filter_specific(df_filter):
            return (df_filter['time'] >= 0.0) & (df_filter['EGF_0'] == 0.0)
        plot_simdata_grid(
            apply_filters(df_data_obs, filter_generic, filter_specific),
            apply_filters(df_sim_obs, filter_generic, filter_specific),
            if_obs, f'{drug}_0', label, IFLABEL,
            ylimits=YLIM, logx=True, logy=False, group='time', color='time',
            figdir=figdir,
            filename=f'training{suffix}_woEGF_tcdr{drug}.pdf',
        )

    filter_generic = lambda f: ((f['time'] == 8.0) & (f['EGFR_crispr'] > 1.0))

    for drug in ['Vemurafenib', 'Dabrafenib', 'LY3009120',
                 'Trametinib', 'Cobimetinib']:

        label = f'{drug.replace("_", "")} [$\mu$M]'

        filter_specific = lambda f: pd.concat([
            (f[f'{zdrug}_0'] == drug_zeros[zdrug])
            for zdrug in RAFI + PANRAFI + MEKI
            if zdrug != drug and zdrug in drug_zeros
        ], axis=1).all(axis=1)

        plot_simdata_grid(
            df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
            df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
            obs=if_obs, xindex=f'{drug}_0', xlabel=label, ylabel=IFLABEL,
            ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
            figdir=figdir, ec50=True, ecmax=True,
            filename=f'training{suffix}_EGFRa_dr{drug}.pdf',
        )

        filter_specific = lambda f: pd.concat([
            (f[f'{zdrug}_0'] == drug_zeros[zdrug])
            for zdrug in RAFI + PANRAFI + MEKI
            if zdrug != drug and zdrug in drug_zeros
        ], axis=1).all(axis=1) & (f['EGF_0'] == 100.0)

        plot_simdatadecomp_grid(
            apply_filters(df_data_obs, filter_generic, filter_specific),
            apply_filters(df_sim_obs, filter_generic, filter_specific),
            if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
            f'{drug}_0', xlabel=label, ylabel=IFLABEL,
            ylimits=YLIM, logx=True, logy=False,
            figdir=figdir,
            filename=f'training{suffix}_EGFRa_dr{drug}_decomp.pdf',
        )

    rafi = 'Vemurafenib'
    rafi_label = f'Vemurafenib [$\mu$M]'
    rafi_val = 1.0
    rafi_0 = f'{rafi}_0'


    def filter_timecourse(df_filter):
        return df_filter[
            (df_filter['time'] > 0) &
            (df_filter['t_presim'] == 0) &
            (df_filter['EGFR_crispr'] == 1.0) &
            (df_filter[rafi_0] == rafi_val) &
            (df_filter[f'{rafi_0}_preeq'] == 1.0)
        ]


    plot_simdata_grid(
        filter_timecourse(df_data_obs), filter_timecourse(df_sim_obs),
        if_obs, 'time', TIMELABEL, IFLABEL,
        ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
        figdir=figdir,
        filename=f'training{suffix}_EGFRwt_tc_{rafi}{rafi_val}.pdf',
    )


    def filter_timecourse(df_filter):
        return df_filter[
            (df_filter['time'] > 0) &
            (df_filter['t_presim'] == 0) &
            (df_filter['EGFR_crispr'] == 1.0) &
            (df_filter[rafi_0] == rafi_val) &
            (df_filter[f'{rafi_0}_preeq'] == 1.0) &
            (df_filter['EGF_0'] == 100.0)
        ]


    plot_simdatadecomp_grid(
        filter_timecourse(df_data_obs), filter_timecourse(df_sim_obs),
        if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
        'time', TIMELABEL, IFLABEL,
        ylimits=YLIM, logx=True, logy=False,
        figdir=figdir,
        filename=f'training{suffix}_EGFRwt_tc_{rafi}{rafi_val}_decomp.pdf',
    )

    filter_generic = lambda f: ((f['EGF_0'] == 100.0) &
                                (f['EGFR_crispr'] != 1.0))
    filter_specific = lambda f: f[rafi_0] == rafi_val

    plot_simdata_grid(
        df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
        df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
        obs=if_obs, xindex=f'time', xlabel=TIMELABEL, ylabel=IFLABEL,
        ylimits=YLIM, logx=False, logy=False, group='EGFR_crispr',
        color='EGFR_crispr',
        figdir=figdir,
        filename=f'training{suffix}_EGFRai_tc_{rafi}{rafi_val}.pdf',
    )

    def filter_rechallenge(df_filter):
        return df_filter[
            (df_filter['time'] > 0) &
            (df_filter['t_presim'] > 0) &
            (df_filter[rafi_0] == 1.0)
        ]

    plot_simdata_grid(
        filter_rechallenge(df_data_obs), filter_rechallenge(df_sim_obs),
        if_obs, 'time', TIMELABEL, IFLABEL,
        ylimits=YLIM, logx=True, logy=False, group='t_presim',
        color='t_presim', figdir=figdir,
        filename=f'training{suffix}_rechallenge_{rafi}{rafi_val}.pdf',
    )

    tp_obs = [o for o in t_obs if not o.startswith('tm')]
    tm_obs = [o for o in t_obs if o.startswith('tm')]

    plot_simdata_wrap(
        df_data_obs.query('time == 0').query('EGFR_crispr == 1.0').query('t_presim == 0.0'),
        df_sim_obs.query('time == 0').query('EGFR_crispr == 1.0').query('t_presim == 0.0'),
        tp_obs, rafi_0, rafi_label,
        'log10(protein abundance) [molecules/cell]',
        ylimits=(5, 13), logy=False, height_scale=0.5,
        figdir=figdir,
        filename=f'training{suffix}_proteomics_dr{rafi}.pdf'
    )

    plot_simdata_wrap(
        df_data_obs.query('time == 0').query('EGFR_crispr == 1.0').query('t_presim == 0.0'),
        df_sim_obs.query('time == 0').query('EGFR_crispr == 1.0').query('t_presim == 0.0'),
        tm_obs, rafi_0, rafi_label, 'log10(transcript abundance) [FPKM]',
        ylimits=(1, 6), logy=False, height_scale=0.5,
        figdir=figdir,
        filename=f'training{suffix}_transcriptomics_dr{rafi}.pdf'
    )

    plot_simdata_wrap(
        df_data_obs.query(f'{rafi_0} == 1.0').query('EGFR_crispr == 1.0').query(
            'EGF_0 == 100').query('t_presim == 0.0'),
        df_sim_obs.query(f'{rafi_0} == 1.0').query('EGFR_crispr == 1.0').query(
            'EGF_0 == 100').query('t_presim == 0.0'),
        tm_obs, 'time', TIMELABEL, 'log10(transcript abundance) [FPKM]',
        ylimits=(1, 6), xlimits=(0, 10), logy=False, logx=False,
        height_scale=0.5, figdir=figdir,
        filename=f'training{suffix}_transcriptomics_tc.pdf'
    )

    plot_simdata_wrap(
        df_data_obs.query('time == 0'), df_sim_obs.query('time == 0'),
        p_obs, rafi_0, rafi_label, 'protein phosphorylation [fraction]',
        ylimits=(0, 1.2), logy=False, height_scale=0.5,
        figdir=figdir,
        filename=f'training{suffix}_phoshoproteomics_dr{rafi}.pdf'
    )

write_timestamp(figdir, 'trainingdata')
