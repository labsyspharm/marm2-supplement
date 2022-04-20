import amici
import scipy
import math
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import MARM

from .common import plot_and_save_fig, get_rule_modifies_site_state, \
    get_rule_removes_site

from .common import (
    check_mode, process_preequilibration, get_dose_response_edatas,
    process_rdata_doseresponse, process_rdata_timecourse,
    get_figure_with_subplots
)

THRESHOLD_MONO_FRACTIONS = 0.025
MAX_MONO_FRACTIONS = 10
THRESHOLD_FLUXES = 0.025
MAX_FLUXES = 10

LEGEND_FONTSIZE = 8


def get_steady_state_obs(model, solver, obs):
    model.setTimepoints([np.inf])
    edata = amici.ExpData(model.get())

    rdata = amici.runAmiciSimulation(model, solver, edata)
    rdata_df = amici.getSimulationObservablesAsDataFrame(model, [edata],
                                                         [rdata])
    return rdata_df[obs]


def getSteadyStateAndInitObs(model, solver, obs):
    model.setTimepoints([0.0, np.inf])
    edata = amici.ExpData(model.get())

    rdata = amici.runAmiciSimulation(model, solver, edata)
    rdata_df = amici.getSimulationObservablesAsDataFrame(model,[edata],[rdata])
    return rdata_df[obs]


def replace_syms(expr, rdata_series):
    symbols = expr.free_symbols
    for sym in symbols:
        expr = expr.subs(sym, rdata_series[str(sym)])
    return expr


def label_thresh_truncated(labels, values, threshold):
    return [label if value > threshold else ''
            for value, label in zip(values, labels)]


def pct_thresh_truncated(pct):
    return ('%1.1f%%' % pct) if pct > THRESHOLD_MONO_FRACTIONS * 100 else ''


def get_subplots(model, mode):
    check_mode(mode)

    if mode == 'monomer':
        names = [mono.name for mono in model.monomers]
    elif mode == 'flux':
        names = [f'{mono.name}::{site}'
                 for mono in model.monomers
                 for site in model.monomers[mono.name].sites]
    elif mode == 'reaction':
        names = [rule.name for rule in model.rules]
    elif mode == 'pattern':
        names = [
            f'{rule.name} E{icp}'
            for rule in model.rules
            for icp, cp in enumerate(rule.reactant_pattern.complex_patterns)
        ]

    ncols = 3
    nrows = math.ceil(len(names) / 3)

    fig, axes_list = get_figure_with_subplots(nrows, ncols)

    return fig, axes_list, names


def plot_mode_dependent(name, model, sim, axes_list, fig, logx, logy, mode):
    check_mode(mode)
    if mode == 'monomer':
        return plot_monomer_fractions(name, model, sim, axes_list, fig, logx,
                                      logy)

    elif mode == 'flux':
        return plot_reaction_fluxes(name, model, sim, axes_list, fig, logx,
                                    logy)

    elif mode == 'reaction':
        return plot_rule_reactions(name, model, sim, axes_list, fig, logx,
                                   logy)

    elif mode == 'pattern':
        return plot_pattern_fluxes(name, model, sim, axes_list, fig, logx,
                                    logy)


def plot_monomer_fractions(name, model, sim, axes_list, fig, logx, logy):
    mono_name = name

    obs_df = get_monomer_fraction_data_frame(sim, mono_name)

    total = obs_df.sum(axis=1)
    other = total * 0

    # reorder according to max fraction
    obs_df = obs_df.loc[:, obs_df.max().sort_values(ascending=True).index]

    num_fractions = len(obs_df.columns)

    # filter according to thresholds
    for icol, col in enumerate(obs_df):
        if num_fractions - icol >= MAX_MONO_FRACTIONS or all(
                obs_df[col] / total < THRESHOLD_MONO_FRACTIONS
        ) or all(obs_df[col] == 0):
            other += obs_df[col]
            obs_df.drop(col, axis=1, inplace=True)

    if other.max() > 0:
        obs_df.insert(0, 'other', other)

    if len(obs_df.columns) == 0:
        return None

    ax = axes_list.pop(0)
    obs_df.plot(kind='area', ax=ax, logx=logx, logy=logy)
    ax.set_ylabel('concentration [$\mu$M]')

    return ax


def plot_reaction_fluxes(name, model, sim, axes_list, fig, logx, logy):
    mono_name, site_name = name.split('::')
    site_rules = [
        rule.name for rule in model.rules
        if rule.name in sim and
        (get_rule_modifies_site_state(
            rule, mono_name, site_name
        ) or get_rule_removes_site(
            rule, mono_name, site_name
        ))
    ]
    if not site_rules:
        return None

    site_flux = sim[site_rules].copy()
    for rule in site_flux.columns.values:
        if get_rule_removes_site(model.components[rule], mono_name,
                                 site_name):
            site_flux[rule] = -site_flux[rule]

    ax = axes_list.pop(0)

    if not site_flux.columns.empty:
        site_flux.plot(
            kind='line', logx=logx, logy=logy,
            ax=ax, fig=fig,
        )
    site_flux.sum(axis=1).plot(
        kind='line', logx=logx, logy=logy,
        ax=ax, fig=fig, color='k', linewidth=2, label='net'
    )

    ax.set_ylabel('rule flux [$\mu$M/h]')
    ax.grid('on', which='major', axis='y', linestyle=':')
    return ax


def plot_pattern_fluxes(name, model, sim, axes_list, fig, logx, logy):
    pattern_name = name
    fluxes = [
        col for col in sim.columns
        if col.endswith(pattern_name)
    ]
    if not fluxes:
        return None

    pattern_flux = sim[fluxes].copy()
    pattern_flux.rename(
        lambda x: x.replace(f'::{pattern_name}', ''),
        axis='columns',
        inplace=True
    )

    net = pattern_flux.sum(axis=1)

    # reorder according to max flux
    pattern_flux = pattern_flux.loc[:, pattern_flux
                                       .apply(np.abs).max(axis=0)
                                       .sort_values(ascending=True).index]
    other = net * 0

    # filter according to thresholds
    max_flux = pattern_flux.apply(np.abs).max(axis=1)
    for icol, col in enumerate(pattern_flux.columns):
        if len(fluxes) - icol >= MAX_FLUXES or all(
                pattern_flux[col].apply(np.abs) / max_flux < THRESHOLD_FLUXES
        ) or all(pattern_flux[col] == 0):
            other += pattern_flux[col]
            pattern_flux.drop(col, axis=1, inplace=True)

    if other.max() > 0:
        pattern_flux.insert(0, 'other', other)

    ax = axes_list.pop(0)
    if len(pattern_flux.columns):
        pattern_flux.plot(
            kind='line', logx=logx, logy=logy,
            ax=ax, fig=fig,
        )
    pattern_flux.sum(axis=1).plot(
        kind='line', logx=logx, logy=logy,
        ax=ax, fig=fig, color='k', linewidth=2, label='net'
    )

    ax.set_ylabel('pattern flux [$\mu$M/h]')
    ax.grid('on', which='major', axis='y', linestyle=':')
    return ax


def plot_rule_reactions(name, model, sim, axes_list, fig, logx, logy):
    filter_name = f'{name}::'
    rule_reactions = sim[[col for col in sim.columns
                          if col.startswith(filter_name)]].copy()
    rule_reactions.rename(
        lambda x: x.replace(filter_name, 'r'),
        axis='columns',
        inplace=True
    )
    ncols = len(rule_reactions.columns)

    net = rule_reactions.sum(axis=1)

    # reorder according to max flux
    rule_reactions = rule_reactions.loc[:, rule_reactions
                                        .apply(np.abs).max(axis=0)
                                        .sort_values(ascending=True).index]
    other = net * 0

    # filter according to thresholds
    max_flux = rule_reactions.apply(np.abs).max(axis=1)
    for icol, col in enumerate(rule_reactions.columns):
        if ncols - icol >= MAX_FLUXES or all(
                rule_reactions[col].apply(np.abs) / max_flux < THRESHOLD_FLUXES
        ):
            other += rule_reactions[col]
            rule_reactions.drop(col, axis=1, inplace=True)

    rule_reactions.insert(0, 'other', other)

    ax = axes_list.pop(0)
    rule_reactions.plot(
        kind='line', logx=logx, logy=logy,
        ax=ax, fig=fig,
    )
    rule_reactions.sum(axis=1).plot(
        kind='line', logx=logx, logy=logy,
        ax=ax, fig=fig, color='k', linewidth=2, label='net'
    )

    ax.set_ylabel('rule flux [$\mu$M/h]')
    ax.grid('on', which='major', axis='y', linestyle=':')
    return ax


def plot_time_course_dose_response(amici_model, obs, t, doses, dose_name,
                                   preequilibrate=True, preequidict=None,
                                   preequilibrate_dose_response=True,
                                   logx=True, logy=True,
                                   figdir=None, filename=None):
    amici_model.setTimepoints(t)
    edata = amici.ExpData(amici_model.get())
    process_preequilibration(amici_model, edata, preequidict, preequilibrate)

    edatas = get_dose_response_edatas(amici_model, dose_name, doses, edata,
                                      preequilibrate,
                                      preequilibrate_dose_response)

    solver = MARM.estimation.get_solver(amici_model)
    solver.setSensitivityOrder(amici.SensitivityOrder_none)

    rdatas = amici.runAmiciSimulations(amici_model, solver, edatas,
                                       num_threads=8)

    ncols = 3
    nrows = math.ceil(len(obs) / 3)
    fig, axes_list = get_figure_with_subplots(nrows, ncols)

    for observable in obs:
        iy = amici_model.getObservableNames().index(observable)
        if logx:
            xx = np.log10(np.asarray(doses))
        else:
            xx = np.asarray(doses)

        if logy:
            yy = np.log10(np.asarray(t))
        else:
            yy = np.asarray(t)
        x, y = np.meshgrid(xx, yy)
        z = np.zeros((len(t), len(doses),))
        for ix, it in itertools.product(range(len(doses)), range(len(t))):
            z[it, ix] = rdatas[ix].y[it, iy]

        ax = axes_list.pop(0)
        cf = ax.contourf(x, y, z, levels=20, vmin=0, vmax=1.5)
        ax.set_ylabel('time [h]')
        ax.invert_yaxis()

        xlabel = dose_name
        for alias, name in {
            'RAFi_0': 'Vemurafenib',
            'MEKi_0': 'Cobimetinib',
        }.items():
            xlabel = xlabel.replace(alias, name)

        ax.set_title(f'{observable}')
        ax.set_xlabel(f'{xlabel} [$\mu$M]')

        if logx:
            ax.set_xticklabels([
                f'$10^{{{x}}}$'.replace('.0', '')
                for x in ax.get_xticks()
            ])

        if logy:
            ax.set_yticklabels([
                f'$10^{{{x}}}$'.replace('.0', '')
                for x in ax.get_yticks()
            ])

    fig.colorbar(cf, ax=axes_list.pop(0))

    for ax in axes_list:
        ax.remove()

    plot_and_save_fig(figdir, filename)


def plot_time_course(rdata, amici_model, pysb_model, fun, mode, figdir=None,
                     logx=True, logy=False, filename=None, noreverse=False):

    timecourse = process_rdata_timecourse(rdata, amici_model, pysb_model, fun,
                                          mode)

    if noreverse:
        timecourse = pd.DataFrame({
            col: timecourse[col].values + timecourse[col + '__reverse'].values
            if col + '__reverse' in timecourse.columns
            else timecourse[col].values
            for col in timecourse.columns
            if not col.split('::')[0].endswith('__reverse')
        }, index=timecourse.index)

    fig, axes_list, names = get_subplots(pysb_model, mode)

    for name in names:

        ax = plot_mode_dependent(name, pysb_model, timecourse, axes_list, fig,
                                 logx, logy, mode)
        if ax is None:
            continue

        ax.set_title(f'{name}')
        ax.set_xlabel('time [h]')
        ax.legend(prop=dict(size=LEGEND_FONTSIZE))

    for ax in axes_list:
        ax.remove()

    plot_and_save_fig(figdir, filename.format(plottype=f'timecourse_{mode}'))

    return timecourse


def plot_dose_response(amici_model, pysb_model, rdatas, doses, dose_name,
                       fun, mode, logx=True,
                       logy=False, figdir=None, filename=None,
                       noreverse=False):

    check_mode(mode)

    doseresponse = process_rdata_doseresponse(
        rdatas, amici_model, pysb_model, fun, doses, mode
    )

    fig, axes_list, names = get_subplots(pysb_model, mode)

    if noreverse:
        doseresponse = pd.DataFrame({
            col: doseresponse[col].values + doseresponse[col + '__reverse'].values
            if col + '__reverse' in doseresponse.columns
            else doseresponse[col].values
            for col in doseresponse.columns
            if not col.split('::')[0].endswith('__reverse')
        }, index=doseresponse.index)

    for name in names:
        ax = plot_mode_dependent(name, pysb_model, doseresponse, axes_list,
                                 fig, logx, logy, mode)

        if ax is None:
            continue

        xlabel = dose_name
        for alias, replacement in {
            'RAFi_0': 'Vemurafenib',
            'MEKi_0': 'Cobimetinib',
        }.items():
            xlabel = xlabel.replace(alias, replacement)

        ax.set_title(f'{name}')
        ax.set_xlabel(f'{xlabel} [$\mu$M]')
        ax.legend(prop=dict(size=LEGEND_FONTSIZE))

    for ax in axes_list:
        ax.remove()

    plot_and_save_fig(figdir, filename.format(plottype=f'doseresponse_{mode}'))

    return doseresponse


def get_monomer_fraction_data_frame(full_obs, mono_name):
    filter_str = f'expl_{mono_name}__'
    obs_df = full_obs[[
        col for col in full_obs.columns
        if col.startswith(filter_str)
    ]].copy()
    obs_df.rename(lambda x: x.replace(filter_str, ''), axis='columns',
                  inplace=True)
    obs_df[obs_df < 0] = 0
    obs_df = obs_df.loc[:, obs_df.max().sort_values(ascending=True).index]
    return obs_df


def plot_synergy(df, rafi_0, meki_0,
                 ax, kind='bliss', marker='pERK_IF_obs'):
    rafi_concs = df[rafi_0].unique()
    rafi_concs = sorted(rafi_concs[rafi_concs > 0])

    meki_concs = df[meki_0].unique()
    meki_concs = sorted(meki_concs[meki_concs > 0])

    synergy = np.zeros((len(rafi_concs), len(meki_concs)))
    for (iRAFi, RAFi), (iMEKi, MEKi) in itertools.product(
            enumerate(rafi_concs), enumerate(meki_concs)):
        ref = df.loc[(df[rafi_0] == 0) & (df[meki_0] == 0), marker].median()
        mono_rafi = df.loc[(df[rafi_0] == RAFi) & (df[meki_0] == 0),
                           marker].median()
        mono_meki = df.loc[(df[rafi_0] == 0) & (df[meki_0] == MEKi),
                           marker].median()
        combo = df.loc[(df[rafi_0] == RAFi) & (df[meki_0] == MEKi),
                       marker].median()

        effect_rafi = (ref - mono_rafi) / ref
        effect_meki = (ref - mono_meki) / ref
        effect_combo = (ref - combo) / ref

        if kind == 'bliss':
            synergy[iRAFi, iMEKi] = effect_combo - (
                        effect_rafi + effect_meki - effect_meki * effect_rafi)

        if kind == 'bliss_nn':
            synergy[iRAFi, iMEKi] = combo - (
                        mono_rafi + mono_meki - mono_rafi * mono_meki)

        if kind == 'hsa':
            synergy[iRAFi, iMEKi] = effect_combo - max(
                [effect_rafi, effect_meki])

        if kind == 'lsa':
            synergy[iRAFi, iMEKi] = effect_combo - min(
                [effect_rafi, effect_meki])

        if kind == 'cdi':
            synergy[iRAFi, iMEKi] = combo / (mono_rafi * mono_meki)

        if kind == 'combo':
            synergy[iRAFi, iMEKi] = effect_combo

        if kind == 'mono_bliss':
            synergy[iRAFi, iMEKi] = (
                        effect_rafi + effect_meki - effect_meki * effect_rafi)

        if kind == 'mono_hsa':
            synergy[iRAFi, iMEKi] = max([effect_rafi, effect_meki])

    ax.set_aspect(1)
    return ax.imshow(synergy, cmap='RdBu', origin='lower', vmin=-1, vmax=1)


def plot_synergies(df_edata, df_rdata, rafi_0, meki_0,
                   kind='bliss', marker='pERK_IF_obs', mode='egf'):
    fig, axes = plt.subplots(nrows=3 if mode == 'egf' else 2,
                             ncols=4, figsize=(15, 10))

    if mode == 'egf':
        conditions = [
            {'time': 0.0833, 'egf': 100.0},
            {'time': 8.0, 'egf': 100.0},
            {'time': 0.0833, 'egf': 0.0},
        ]
    else:
        conditions = [
            {'time': 0.0, 'nras': 1.0},
            {'time': 0.0, 'nras': 0.0},
        ]

    for icond, cond in enumerate(conditions):
        if mode == 'egf':
            subset = lambda df: (df.time == cond['time']) & \
                                (df.EGF_0 == cond['egf'])
        else:
            subset = lambda df: (df.time == cond['time']) & \
                                (df.NRAS_Q61mut == cond['nras'])

        plot_synergy(df_edata[subset(df_edata)], rafi_0=rafi_0, meki_0=meki_0,
                     kind=kind, ax=axes[icond, 0], marker=marker)
        plot_synergy(df_rdata[subset(df_rdata)], rafi_0=rafi_0, meki_0=meki_0,
                     kind=kind, ax=axes[icond, 1], marker=marker)
        plot_synergy(df_rdata[subset(df_rdata)], rafi_0=rafi_0, meki_0=meki_0,
                     kind='combo', ax=axes[icond, 2], marker=marker)
        im = plot_synergy(df_rdata[subset(df_rdata)], rafi_0=rafi_0,
                          meki_0=meki_0,
                          kind=f'mono_{kind}', ax=axes[icond, 3],
                          marker=marker)
    fig.colorbar(im, label=f'excess over {kind}')


def plot_isobologram(df, time, egf, rafi_0, meki_0, ax, vmax=2):
    df = df[(df.time == time) & (df.EGF_0 == egf)].copy()

    rafi_concs = df[rafi_0].unique()
    rafi_concs = sorted(rafi_concs[rafi_concs > 0])

    meki_concs = df[meki_0].unique()
    meki_concs = sorted(meki_concs[meki_concs > 0])

    response = np.zeros((len(rafi_concs), len(meki_concs)))
    for (iRAFi, RAFi), (iMEKi, MEKi) in itertools.product(
            enumerate(rafi_concs), enumerate(meki_concs)):
        response[iRAFi, iMEKi] = df.loc[
            (df[rafi_0] == RAFi) & (df[meki_0] == MEKi),
            'pERK_IF_obs'
        ].median()

    smooth_response = scipy.ndimage.filters.gaussian_filter(response, 0.75)
    #ax.set_aspect(1)
    x, y = np.meshgrid(meki_concs, rafi_concs)
    return ax.contour(x, y, smooth_response, cmap='viridis', origin='lower',
                      vmin=0, vmax=vmax, levels=10)


def plot_isobolograms(df_edata, df_rdata, meki_0, rafi_0, time=8, egf=100,
                      vmax=2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    plot_isobologram(df_edata, time=time, egf=egf, ax=axes[0],
                     rafi_0=rafi_0, meki_0=meki_0, vmax=vmax)
    plot_isobologram(df_rdata, time=time, egf=egf, ax=axes[1],
                     rafi_0=rafi_0, meki_0=meki_0, vmax=vmax)
