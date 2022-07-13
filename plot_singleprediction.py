import sys
import pandas as pd

from MARM.estimation import get_model, RAFI, MEKI, PANRAFI
from MARM.analysis import (
    read_settings, get_obs_df, read_all_analysis_dataframes, load_parameters,
    write_timestamp, N_RUNS
)
from MARM.paths import get_figure_dir
from MARM.visualize import IFLABEL, plot_simdata_grid, plot_simdatadecomp_grid

sxs = read_settings(sys.argv, index=False, threads=False)
model = get_model(sxs['model_name'],
                  sxs['variant'],
                  sxs['dataset'],
                  'channel_monoobs')
figdir = get_figure_dir(sxs['model_name'], sxs['variant'], sxs['dataset'])

YLIM = (-0.2, 2.25)

for suffix in '', '_cf':
    df = read_all_analysis_dataframes(sxs, 'singleprediction' + suffix)

    df_data_obs, df_sim_obs, if_obs, t_obs, p_obs, a_obs, drug_zeros = \
        get_obs_df(df, model)

    for obs in if_obs:
        for index in range(N_RUNS):
            par = load_parameters(model, sxs, rafi=None, meki=None, prafi=None,
                                  index=index, allow_missing_pars=True)
            obs_name = obs.split('_')[0]
            offset = df_sim_obs[(df_sim_obs.variable == f'{obs_name}_onco_obs') &
                                (df_sim_obs.datatype == 'simulation') &
                                (df_sim_obs.par_index == index)]
            offset.value = model.getParameterByName(f'{obs_name}_IF_offset')
            offset.variable = f'{obs_name}_background_obs'
            df_sim_obs = pd.concat([df_sim_obs, offset], ignore_index=True)


    def apply_filters(df_filter, generic, specific):
        return df_filter[generic(df_filter) & specific(df_filter)]


    filter_generic = lambda f: ((f['time'] == 0.0833) & (f['EGFR_crispr'] == 1.0))
    for drug in PANRAFI:
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
            ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
            figdir=figdir,
            filename=f'prediction{suffix}_EGFRwt_dr{drug}.pdf',
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
            filename=f'prediction{suffix}_EGFRwt_dr{drug}_decomp.pdf',
        )


    filter_generic = lambda f: ((f['time'] == 8.0) & (f['EGFR_crispr'] > 1.0))
    for drug in ['LY3009120']:
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
            ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
            figdir=figdir,
            filename=f'prediction{suffix}_EGFRa_dr{drug}.pdf',
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
            filename=f'prediction{suffix}_EGFRa_dr{drug}_decomp.pdf',
        )


    filter_generic = lambda f: ((f['time'] == 0.0833) & (f['EGFR_crispr'] == 1.0))
    for meki, rafi, rafi_val in zip(
            ['Cobimetinib', 'Cobimetinib', 'Cobimetinib', 'Trametinib'],
            ['Vemurafenib', 'LY3009120', 'AZ_628', 'Dabrafenib'],
            [1.0, 1.0, 1.0, 0.1],
    ):

        meki_0 = f'{meki}_0'
        meki_label = f'{meki.replace("_", "")} [$\mu$M]'
        rafi_0 = f'{rafi}_0'

        filter_specific = lambda f: f[rafi_0] == rafi_val


        plot_simdata_grid(
            df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
            df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
            obs=if_obs, xindex=meki_0, xlabel=meki_label, ylabel=IFLABEL,
            ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
            figdir=figdir,
            filename=f'prediction{suffix}_EGFRwt_dr{meki}_{rafi}{rafi_val}.pdf',
        )

        filter_specific = lambda f: (f[rafi_0] == rafi_val) & (f['EGF_0'] == 100.0)

        plot_simdatadecomp_grid(
            apply_filters(df_data_obs, filter_generic, filter_specific),
            apply_filters(df_sim_obs, filter_generic, filter_specific),
            if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
            meki_0, xlabel=meki_label, ylabel=IFLABEL,
            ylimits=YLIM, logx=True, logy=False,
            figdir=figdir,
            filename=f'prediction{suffix}_EGFRwt_dr{meki}_{rafi}{rafi_val}_decomp.pdf',
        )

    rafi = 'Vemurafenib'
    meki = 'Cobimetinib'
    rafi_val = 1.0
    meki_0 = f'{meki}_0'
    meki_label = f'{meki.replace("_", "")} [$\mu$M]'
    rafi_0 = f'{rafi}_0'

    filter_generic = lambda f: ((f['time'] == 8.0) & (f['EGFR_crispr'] > 1.0))
    filter_specific = lambda f: f[rafi_0] == rafi_val

    plot_simdata_grid(
        df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
        df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
        obs=if_obs, xindex=meki_0, xlabel=meki_label, ylabel=IFLABEL,
        ylimits=YLIM, logx=True, logy=False, group='EGF_0', color='EGF_0',
        figdir=figdir,
        filename=f'prediction{suffix}_EGFRa_dr{meki}_{rafi}{rafi_val}.pdf',
    )

    filter_specific = lambda f: (f[rafi_0] == rafi_val) & \
                                (f['EGF_0'] == 100.0)

    plot_simdatadecomp_grid(
        apply_filters(df_data_obs, filter_generic, filter_specific),
        apply_filters(df_sim_obs, filter_generic, filter_specific),
        if_obs, a_obs + ['pMEK_background_obs', 'pERK_background_obs'],
        meki_0, xlabel=meki_label, ylabel=IFLABEL,
        ylimits=YLIM, logx=True, logy=False,
        figdir=figdir,
        filename=f'prediction{suffix}_EGFRa_dr{meki}_{rafi}{rafi_val}_decomp.pdf',
    )

    filter_generic = lambda f: ((f['EGF_0'] == 100.0) & (f['EGFR_crispr'] == 1.0))
    filter_specific = lambda f: f[rafi_0] == rafi_val

    plot_simdata_grid(
        df_data=apply_filters(df_data_obs, filter_generic, filter_specific),
        df_sim=apply_filters(df_sim_obs, filter_generic, filter_specific),
        obs=if_obs, xindex=meki_0, xlabel=meki_label, ylabel=IFLABEL,
        ylimits=YLIM, logx=True, logy=False, group='time', color='time',
        figdir=figdir,
        filename=f'prediction{suffix}_EGFRwt_tc_dr{meki}_{rafi}{rafi_val}.pdf',
    )

write_timestamp(figdir, 'singleprediction')
