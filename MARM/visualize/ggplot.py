from plotnine import *
import mizani.formatters
import mizani.transforms
import matplotlib.pyplot as plt

import copy
import amici
import os
import math
import pandas as pd
import numpy as np
import itertools

from .common import uM_to_molecules, molecules_to_uM

from MARM.estimation import RAFI, PANRAFI, MEKI

from scipy.optimize import least_squares

PLOTNINE_FIGWIDTH = 10


PLOTNINE_THEME = {
    'dpi': 300,
    'legend_background': element_blank(),
    'legend_key': element_blank(),
    'panel_background': element_blank(),
    'panel_border': element_blank(),
    'strip_background': element_blank(),
    'panel_grid_major': element_blank(),
}


def minus_keys(dictionary, keys):
    shallow_copy = dict(dictionary)
    for key in keys:
        del shallow_copy[key]
    return aes(**shallow_copy)


def pctile_lower(x):
    return np.quantile(x, 0.10)


def pctile_upper(x):
    return np.quantile(x, 0.90)


def plot_simdatadecomp_grid(
        df_data,
        df_sim,
        obs,
        act_obs,
        xindex,
        xlabel,
        ylabel,
        ylimits=None,
        logx=True,
        logy=True,
        height_scale=2/3,
        filename=None,
        figdir=None
):
    ncols = len(obs)
    nrows = 1

    channels = ['phys', 'onco', 'background', 'data']

    data = copy.deepcopy(df_data[df_data.variable.apply(lambda x: x in obs)])
    data_mappings = {
        'ymin': 'ymin',
        'ymax': 'ymax',
    }

    data['channel'] = 'data'
    data.channel = pd.Categorical(data.channel, ordered=True,
                                  categories=channels)

    sim = copy.deepcopy(df_sim[df_sim.variable.apply(
        lambda x: x in act_obs
    )])

    if len(sim) == 0:
        return

    sim['channel'] = sim.variable.apply(lambda x: x.split('_')[1])
    sim['variable'] = sim.variable.apply(lambda x: x.split('_')[0] +
                                         '_IF_obs')
    sim.channel = pd.Categorical(sim.channel, ordered=True,
                                 categories=channels)

    base_mappings = dict(
        x=xindex,
        y='value',
        color='channel',
        group='channel',
        fill='channel',
    )

    mapping_data = aes(
        **base_mappings,
        **data_mappings,
    )
    mapping_sim = aes(
        **base_mappings,
    )

    data.variable = pd.Categorical(data.variable, ordered=True,
                                   categories=obs)

    plot = (
            ggplot()
            + xlab(xlabel)
            + ylab(ylabel)
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH,
                                 PLOTNINE_FIGWIDTH*nrows/ncols * height_scale),
                    **PLOTNINE_THEME)
            + facet_grid(('.', 'variable',), labeller='label_context')
    )

    # manually stack
    sim_stacked = copy.deepcopy(sim)
    for conc in sim_stacked[xindex].unique():
        for var in sim_stacked.variable.unique():
            for pidx in sim_stacked.par_index.unique():
                for channel, base_channel in zip(reversed(channels[:-2]),
                                                 reversed(channels[1:-1])):
                    subset = (sim_stacked.variable == var) & \
                             (sim_stacked.par_index == pidx) & \
                             (sim_stacked[xindex] == conc)

                    base_val = (sim_stacked.channel == base_channel) & subset
                    val = (sim_stacked.channel == channel) & subset

                    sim_stacked.loc[val, 'value'] += \
                        sim_stacked.loc[base_val, 'value'].values[0]

    sim_stacked.variable = pd.Categorical(sim_stacked.variable, ordered=True,
                                          categories=obs)

    plot = plot + stat_summary(data=sim_stacked,
                               mapping=minus_keys(mapping_sim, ['color']),
                               size=0.5, fun_y=np.median,
                               geom='area', position='identity')

    plot = plot + stat_summary(data=sim_stacked,
                               mapping=minus_keys(mapping_sim,
                                                  ['color', 'fill']),
                               size=0.5, fill='black', alpha=0.2,
                               fun_ymin=pctile_lower, fun_ymax=pctile_upper,
                               geom='ribbon')

    plot = plot + geom_pointrange(data=data, mapping=mapping_data,
                                  size=0.75, fatten=2)

    set_xy_scale(plot, logy, logx, False, '', xindex, ylimits, None, None,
                 adapt_color_scale=False, adapt_fill_scale=False,
                 expand=False)

    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    plot += scale_color_manual(values=colors)
    plot += scale_fill_manual(values=colors)

    plot_and_save_fig(plot, figdir, filename)


def plot_simdata_grid(
        df_data,
        df_sim,
        obs,
        xindex,
        xlabel,
        ylabel,
        ylimits=None,
        logx=True,
        logy=True,
        logz=False,
        group=None,
        rows=None,
        color='datatype',
        height_scale=2/3,
        expand=False,
        ec50=False,
        ecmax=False,
        filename=None,
        figdir=None
):
    data, sim = process_sim_data_dfs(df_data, df_sim, obs)

    if len(data) + len(sim) == 0:
        return

    ncols = len(obs)
    if rows:
        nrows = len(sim[rows].unique())
    else:
        nrows = 1

    data_mappings = {
        'ymin': 'ymin',
        'ymax': 'ymax',
    }
    sim_mappings = {
        'fill': color,
    }
    base_mappings = dict(
        x=xindex,
        y='value',
        color=color,
    )
    if group is not None:
        base_mappings['group'] = group

    mapping_data = aes(
        **base_mappings,
        **data_mappings
    )
    mapping_sim = aes(
        **base_mappings,
        **sim_mappings,
    )

    plot = (
            ggplot()
            + xlab(xlabel)
            + ylab(ylabel)
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH,
                                 PLOTNINE_FIGWIDTH*nrows/ncols * height_scale),
                    **PLOTNINE_THEME)
    )

    if rows:
        plot = plot + facet_grid(
                (rows, 'variable',),
                labeller='label_context'
            )
    else:
        plot = plot + facet_grid(
            ('.', 'variable',),
            labeller='label_context'
        )

    if len(data):
        plot = plot + geom_pointrange(data=data, mapping=mapping_data,
                                      size=0.75, fatten=2)

    if len(sim):
        plot = plot + stat_summary(data=sim,
                                   mapping=minus_keys(mapping_sim, ['fill']),
                                   fun_y=np.median,
                                   geom='line', size=1)
        plot = plot + stat_summary(data=sim,
                                   mapping=minus_keys(mapping_sim, ['color']),
                                   fun_y=np.median,
                                   fun_ymin=pctile_lower,
                                   fun_ymax=pctile_upper,
                                   geom='ribbon', alpha=0.2)

    set_xy_scale(plot, logy, logx, logz, '', xindex, ylimits, None, None,
                 expand=expand,
                 adapt_fill_scale=False, adapt_color_scale=False)

    if color == 'time':
        kwargs = {
            'limits': (0.0833, 24),
            'trans': 'log10',
            'low': '#bd0026',
            'high': '#fec44f',
        }
        plot += scale_color_gradient(**kwargs)
        plot += scale_fill_gradient(**kwargs)
    elif color == 'EGF_0':
        kwargs = {
            'name': 'inferno',
        }
        plot += scale_color_cmap_d(**kwargs)
        plot += scale_fill_cmap_d(**kwargs)
    elif logz:
        trans = mizani.transforms.log10_trans()
        plot += scale_color_continuous(trans=trans)
        plot += scale_fill_continuous(trans=trans)

    # compute EC50/ECMax
    if ec50 or ecmax:
        if group is None:
            sel = sim.variable
        else:
            sel = itertools.product(sim.variable.unique(),
                                    sim[group].unique())

        sim_cf = []
        data_cf = []
        for s in sel:
            for ip in sim.par_index.unique():
                sim_cf.append(analyse_dr(sim, ip, s, xindex, group))
            data_cf.append(analyse_dr(data, None, s, xindex, group))

        sim_ec = pd.DataFrame(sim_cf)
        data_ec = pd.DataFrame(data_cf)

        if color == 'EGF_0':
            for df in [sim_ec, data_ec]:
                df[color] = pd.Categorical(df[color], ordered=True)

        groupers = ['variable', color]
        if group is not None and group != color:
            groupers.append(group)

        averaged_sim = sim_ec.drop(columns='par_index').groupby(
            groupers
        ).aggregate(np.median).reset_index()

        lb_sim = sim_ec.drop(columns='par_index').groupby(
            groupers
        ).aggregate(lambda x: np.percentile(x, 10)).reset_index()

        ub_sim = sim_ec.drop(columns='par_index').groupby(
            groupers
        ).aggregate(lambda x: np.percentile(x, 90)).reset_index()

        for df, label in zip([averaged_sim, lb_sim, ub_sim, data_ec],
                             ['sim_ec_med', 'sim_ec_lb', 'sim_ec_ub',
                              'data_ec']):
            df.to_csv(os.path.join(figdir,
                                   filename.replace('.pdf', f'{label}.csv')))

        for val, ecbool, intercept, ecgeom in zip(['ec_50', 'ec_max'],
                                                  [ec50, ecmax],
                                                  ['xintercept', 'yintercept'],
                                                  [geom_vline, geom_hline]):
            if not ecbool:
                continue

            ecmapping = aes(**{
                'group': group,
                'color': color,
                intercept: val,
            })

            plot += ecgeom(
                data=averaged_sim,
                mapping=ecmapping,
                linetype='solid',
            )
            plot += ecgeom(
                data=data_ec,
                mapping=ecmapping,
                linetype='dashed',
            )
            plot += ecgeom(
                data=lb_sim,
                mapping=ecmapping,
                linetype='solid',
                alpha=0.3
            )
            plot += ecgeom(
                data=ub_sim,
                mapping=ecmapping,
                linetype='solid',
                alpha=0.3
            )

    plot_and_save_fig(plot, figdir, filename)


def analyse_dr(df, ip, s, xindex, group):
    if group is None:
        if ip is None:
            curve = df[df.variable == s]
            cf_map = {
                'variable': s,
            }
        else:
            curve = df[(df.par_index == ip) &
                       (df.variable == s)]
            cf_map = {
                'variable': s,
                'par_index': ip,
            }

    else:
        if ip is None:
            curve = df[(df.variable == s[0]) &
                       (df[group] == s[1])]
            cf_map = {
                'variable': s[0],
                group: s[1]
            }
        else:
            curve = df[(df.par_index == ip) &
                       (df.variable == s[0]) &
                       (df[group] == s[1])]
            cf_map = {
                'variable': s[0],
                group: s[1],
                'par_index': ip,
            }

    return {
            ** compute_ec50(curve[xindex].values,
                            curve.value.values),
            **cf_map
        }


def compute_ec50(x, y):
    if not x.size:
        return {'ec_0': np.nan,
                'ec_50': np.nan,
                'ec_max': np.nan}

    def hill(p):
        return p[2] - (p[2] - p[0]) / (1 + (10 ** p[1] / x)) - y

    def hill_jac(p):
        return np.vstack([
            1 / (1 + (10 ** p[1] / x)),
            + (p[2] - p[0]) * 10 ** p[1] * x * np.log(10) /
            np.power(x + 10 ** p[1], 2),
            1 - 1 / (1 + (10 ** p[1] / x))
        ]).T

    p0 = np.asarray([0.5,
                     np.median(np.log10(x)),
                     min([max([y[0], 0]), 2.5])])

    res = least_squares(hill, p0, hill_jac,
                        bounds=([  0, np.min(np.log10(x)), 0  ],
                                [1.5, np.max(np.log10(x)), 2.5]))

    return {'ec_0': res.x[2],
            'ec_50': 10 ** res.x[1],
            'ec_max': res.x[0],
            'ec_max_rel': (res.x[2]-res.x[0])/res.x[2]}


def plot_simdata_wrap(df_data, df_sim, obs, xindex, xlabel, ylabel,
                      ylimits=None, xlimits=None, logx=True, logy=True,
                      logz=False, height_scale=2/3, filename=None,
                      figdir=None):

    data, sim = process_sim_data_dfs(df_data, df_sim, obs)

    ncols = 2
    nrows = math.ceil(len(obs)/2)

    basemappings = dict(
        x=xindex, y='value', color='datatype', fill='datatype',
    )
    datamappings = dict(
        ymin='ymin', ymax='ymax',
    )
    mapping_data = aes(**basemappings, **datamappings)
    mapping_sim = aes(**basemappings)

    plot = (
            ggplot()
            + facet_wrap('variable', ncol=ncols)
            + geom_pointrange(data=data,
                              mapping=minus_keys(mapping_data, ['fill']),
                              size=0.75, fatten=2)
            + stat_summary(data=sim,
                           mapping=minus_keys(mapping_sim, ['fill']),
                           fun_y=np.median,
                           geom='line', size=1)
            + stat_summary(data=sim,
                           mapping=minus_keys(mapping_sim, ['color']),
                           fun_y=np.median,
                           fun_ymin=pctile_lower, fun_ymax=pctile_upper,
                           geom='ribbon', alpha=0.2)
            + xlab(xlabel)
            + ylab(ylabel)
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH,
                                 PLOTNINE_FIGWIDTH * nrows/ncols*height_scale),
                    **PLOTNINE_THEME)
    )

    set_xy_scale(plot, logy, logx, logz, '', xindex, ylimits, xlimits, None,
                 adapt_color_scale=False, adapt_fill_scale=False)

    plot_and_save_fig(plot, figdir, filename)


def plot_simdata_heatmap(
        df_in,
        obs,
        xindex,
        xlabel,
        yindex,
        ylabel,
        zlabel,
        rows=None,
        logx=True,
        logy=True,
        logz=False,
        zlims=(0, 1.5),
        filename=None,
        cmap='viridis',
        figdir=None):
    df = copy.deepcopy(df_in)

    # make variable categorical
    df.variable = pd.Categorical(df.variable,
                                 ordered=True,
                                 categories=obs)

    # drop anything that could not be matched to obs
    df.dropna(subset=['variable'], inplace=True)

    hasdata = df['datatype'].str.contains('data').any()
    hassim = df['datatype'].str.contains('simulation').any()

    ncols = hasdata + hassim
    if rows is None:
        nrows = 1
    else:
        nrows = len(df[rows].unique())

    mapping = aes(
        x=xindex,
        y=yindex,
        fill='value',
    )

    plot = (
            ggplot(df, mapping)
            + xlab(xlabel)
            + ylab(ylabel)
            + theme(figure_size=(PLOTNINE_FIGWIDTH / 2,
                                 PLOTNINE_FIGWIDTH * nrows / ncols / 4),
                    dpi=300,
                    panel_background=element_rect(fill='white'),
                    strip_background=element_blank(),)
    )

    if rows is None:
        plot = plot + facet_grid(
            ('.', 'datatype',),
            labeller='label_context'
        )
    else:
        plot = plot + facet_grid(
            (rows, 'datatype',),
            labeller='label_context'
        )

    if hasdata or hassim:
        plot += geom_tile(aes(width=.5, height=.5))

    set_xy_scale(plot, logy, logx, logz, '', xindex, None, None, zlims, cmap)

    plot_and_save_fig(plot, figdir, filename)


def plot_relationship2D(model, solver,
                        x_range=np.logspace(-1, 1, 10),
                        y_range=np.logspace(-1, 1, 10),
                        x_var='', output_var='', y_var='',
                        x_label='', output_label='', y_label='',
                        logy=True, logx=True, logz=False, zlimits=None,
                        filename=None, figdir=None):
    x_vals = []
    output_vals = []
    y_vals = []
    original_pars = model.getParameters()
    original_fpars = model.getFixedParameters()

    if zlimits is None:
        zlimits = (0, 1.5)

    for x_val, y_val in itertools.product(x_range, y_range):
        x_vals.append(x_val)
        y_vals.append(y_val)

        uM_pars = ['MEKi_0', 'RAFi_0', 'BRAFBRAF_0', 'BRAFCRAF_0',
                   'CRAFCRAF_0']
        ng_pars = ['EGF_0']

        if x_var.endswith('_0') and x_var not in uM_pars + ng_pars:
            x_val = uM_to_molecules(x_val)

        if y_var.endswith('_0') and y_var not in uM_pars + ng_pars:
            y_val = uM_to_molecules(y_val)

        if x_var in model.getParameterNames():
            model.setParameterByName(x_var, x_val)
        elif x_var in model.getFixedParameterNames():
            model.setFixedParameterByName(x_var, x_val)
        else:
            raise ValueError(f'Unknown input variable {x_var}.')

        if y_var in model.getParameterNames():
            model.setParameterByName(y_var, y_val)
        elif y_var in model.getFixedParameterNames():
            model.setFixedParameterByName(y_var, y_val)
        else:
            raise ValueError(f'Unknown parameter variable {y_var}.')

        if output_var in model.getObservableNames():
            iy = model.getObservableNames().index(output_var)
        else:
            raise ValueError(f'Unknown observable variable {output_var}.')

        model.setTimepoints([np.inf])

        rdata = amici.runAmiciSimulation(model, solver)

        output_val = rdata['y'][0, iy]
        if not output_var.endswith('_IF_obs') and 'active' not in output_var:
            output_val = molecules_to_uM(output_val)

        output_vals.append(output_val)

    model.setParameters(original_pars)
    model.setFixedParameters(original_fpars)

    df_results = pd.DataFrame.from_dict({
        x_label: x_vals,
        output_label: output_vals,
        y_label: y_vals,
    },
        orient='columns'
    )

    mapping = aes(
        x=x_label,
        y=y_label,
        fill=output_label,
    )

    plot = (
            ggplot(df_results, mapping)
            + geom_tile(aes(width=.5, height=.5))
            + xlab(x_label)
            + ylab(y_label)
            + theme(figure_size=(PLOTNINE_FIGWIDTH / 2,
                                 PLOTNINE_FIGWIDTH / 2),
                    dpi=300,
                    panel_background=element_rect(fill='white'),
                    strip_background=element_blank(), )
    )

    set_xy_scale(plot, logy, logx, logz, x_label, y_label, None, None, zlimits)

    plot_and_save_fig(plot, figdir, filename)
    return df_results


def plot_relationship(model, solver,
                      input_range=np.logspace(-1, 1, 10),
                      parameter_range=np.logspace(-1, 1, 10),
                      input_var='', output_var='', parameter_var='',
                      input_label='', output_label='', parameter_label='',
                      logy=False, logx=True, logc=True, ylimits=None,
                      filename=None, figdir=None):
    input_vals = []
    output_vals = []
    parameter_vals = []
    original_pars = model.getParameters()
    original_fpars = model.getFixedParameters()

    for input_val, parameter_val in itertools.product(input_range,
                                                      parameter_range):
        parameter_vals.append(parameter_val)
        input_vals.append(input_val)

        uM_pars = ['MEKi_0', 'RAFi_0', 'BRAFBRAF_0', 'BRAFCRAF_0',
                   'CRAFCRAF_0']
        ng_pars = ['EGF_0']
        if input_var.endswith('_0') and input_var not in uM_pars + ng_pars:
            input_val = uM_to_molecules(input_val)

        if parameter_var.endswith('_0') and \
                parameter_var not in uM_pars + ng_pars:
            parameter_val = uM_to_molecules(parameter_val)

        if input_var in model.getParameterNames():
            model.setParameterByName(input_var, input_val)
        elif input_var in model.getFixedParameterNames():
            model.setFixedParameterByName(input_var, input_val)
        else:
            raise ValueError(f'Unknown input variable {input_var}.')

        if parameter_var in model.getParameterNames():
            model.setParameterByName(parameter_var, parameter_val)
        elif parameter_var in model.getFixedParameterNames():
            model.setFixedParameterByName(parameter_var, parameter_val)
        else:
            raise ValueError(f'Unknown parameter variable {parameter_var}.')

        if output_var in model.getObservableNames():
            iy = model.getObservableNames().index(output_var)
        else:
            raise ValueError(f'Unknown observable variable {output_var}.')

        model.setTimepoints([np.inf])

        rdata = amici.runAmiciSimulation(model, solver)

        output_val = rdata['y'][0, iy]
        if not output_var.endswith('_IF_obs') and 'active' not in output_var:
            output_val = molecules_to_uM(output_val)

        output_vals.append(output_val)

    model.setParameters(original_pars)
    model.setFixedParameters(original_fpars)

    df_results = pd.DataFrame.from_dict({
        input_label: input_vals,
        output_label: output_vals,
        parameter_label: parameter_vals,
    },
        orient='columns'
    )

    mapping = aes(
        x=input_label,
        y=output_label,
        color=parameter_label,
        group=parameter_label
    )
    plot = (
            ggplot(mapping)
            + geom_line(data=df_results)
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH/3, PLOTNINE_FIGWIDTH/3),
                    **PLOTNINE_THEME)
    )

    set_xy_scale(plot, logy, logx, False, output_label, input_label,
                 ylimits, None, None)

    if logc:
        plot += scale_color_cmap(
            trans='log10',
            labels=lambda lbl: log_breaks_labels(lbl, parameter_label)
        )
    else:
        plot += scale_color_cmap(
            labels=mizani.formatters.mpl_format()
        )

    plot_and_save_fig(plot, figdir, filename)


def process_sim_data_dfs(df_data, df_sim, obs):
    dfs = []
    for df in [df_data, df_sim]:
        # make variable categorical to ensure proper ordering
        df_post = copy.deepcopy(df)
        df_post.variable = pd.Categorical(df_post.variable, ordered=True,
                                          categories=obs)
        # drop anything that could not be matched to obs
        df_post.dropna(subset=['variable', 'value'], inplace=True)
        dfs.append(df_post)

    return dfs


def plot_deconvolution(df, iterator, figdir, filename):

    nrows = len(df.step.unique())
    ncols = len(df.channel.unique())
    height_scale = 1

    df.channel = pd.Categorical(df.channel, ordered=True,
                                categories=['phys', 'onco'])

    plot = (
            ggplot(mapping=aes(x='time', y='value',
                               color=iterator,
                               group=iterator),
                   data=df)
            + xlab('time after EGF stimulation [h]')
            + ylab('concentration [uM]')
            + stat_summary(fun_y=np.median, geom='line', size=1)
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH,
                                 PLOTNINE_FIGWIDTH*nrows/ncols * height_scale),
                    **PLOTNINE_THEME)
            + facet_grid(('step', 'channel',),
                         labeller='label_context',
                         scales='free_y')
    )

    set_xy_scale(plot, False, False, True, '', '', None, None, None, \
                 expand=False)

    plot_and_save_fig(plot, figdir, filename)


def plot_gains(df, x_axis, x_label, figdir, filename):

    plot = (
            ggplot(mapping=aes(x=x_axis, y='value', color='channel',
                               group='channel', fill='channel'),
                   data=df[df.variable.apply(lambda x: '_to_' in x)])
            + xlab(x_label)
            + ylab('signaling gain')
            + theme_matplotlib()
            + theme(figure_size=(PLOTNINE_FIGWIDTH * 1/3,
                                 PLOTNINE_FIGWIDTH),
                    **PLOTNINE_THEME)
            + scale_color_cmap_d(name='tab10')
            + scale_fill_cmap_d(name='tab10')
            + facet_grid(('step', '.',),
                         labeller='label_context')
    )

    plot = median_with_uncertainty(plot)

    set_xy_scale(plot, False, True, False, '', '', (0, 3), None, None,
                 adapt_color_scale=False, adapt_fill_scale=False, expand=False)

    plot_and_save_fig(plot, figdir, filename)


def plot_raf_states(df, iterator, xlabel, figdir, filename):
    df_melt = pd.melt(df, id_vars=[iterator, 'par_index'],
                      value_vars=['baseline_R', 'baseline_IR',
                                  'baseline_RR', 'baseline_RRI',
                                  'baseline_IRRI'])
    plot = (
        ggplot(df_melt, aes(x=iterator, y='value', group='variable',
                            color='variable', fill='variable'))
        + stat_summary(fun_y=np.median, geom='line', size=1)
        + xlab(xlabel)
        + scale_x_log10()
        + ylab('state')
        + theme('minimal', figure_size=(
            PLOTNINE_FIGWIDTH / 3, PLOTNINE_FIGWIDTH / 3 * 2 / 3
        ))
    )
    plot = median_with_uncertainty(plot)

    plot_and_save_fig(plot, figdir, filename)


def plot_entropy(df_melt, figdir, figname):

    df_melt.drug = pd.Categorical(df_melt.drug, ordered=True,
                                  categories=['Vemurafenib',
                                              'Cobimetinib'])

    df_melt.time = pd.Categorical(df_melt.time, ordered=True,
                                  categories=sorted(df_melt.time.unique()))

    plot = (
            ggplot(df_melt, aes(x='conc', y='value', group='time',
                                color='time', fill='time'))
            + facet_grid(('variable', 'drug'), scales='free_x')
            + xlab('concentration [$\mu$M]')
            + scale_x_log10(expand=(0.01, 0.0))
            + scale_y_continuous(expand=(0.01, 0.0))
            + scale_color_discrete()
            + scale_fill_discrete()
            + ylab('entropy [bits]')
            + theme_matplotlib()
            + theme('minimal', figure_size=(PLOTNINE_FIGWIDTH / 3,
                                            PLOTNINE_FIGWIDTH),
                    **PLOTNINE_THEME)
    )

    plot = median_with_uncertainty(plot)

    plot_and_save_fig(plot, figdir, figname)


def plot_drug_free_monomers(df, iterator, xlabel, figdir, filename,
                            channel=False):
    df_drug_free = copy.deepcopy(df)

    drug = iterator.replace('_0', '')

    if drug in RAFI + PANRAFI:
        dimers = ['baseline_RR', 'baseline_RRI', 'baseline_IRRI']
        monomers = ['baseline_R', 'baseline_IR']

        df_drug_free[dimers] = df_drug_free[dimers].div(
            df_drug_free[dimers].sum(axis=1), axis=0
        )

        frac = df_drug_free[df_drug_free[iterator] ==
                            df_drug_free[iterator].max()]['baseline_RRI']
        print(f'{drug} RRI: {frac.mean()} +- {frac.std()}%')

        df_drug_free['drug_free_dimeric_raf'] = \
            df_drug_free['baseline_RR'] \
            + 0.5 * df_drug_free['baseline_RRI']

        df_drug_free['drug_free_monomeric_raf'] = \
            df_drug_free['baseline_R'].div(
                df_drug_free[monomers].sum(axis=1),
                axis=0
            )

        values = ['drug_free_monomeric_raf', 'drug_free_dimeric_raf']
    elif drug in MEKI:
        if not channel:
            meks1 = ['drugfree_pMEK', 'inhibited_pMEK']
            meks2 = ['drugfree_uMEK', 'inhibited_uMEK']
            values = ['drugfree_uMEK', 'drugfree_pMEK']
        else:
            meks1 = [f'drugfree_pMEK_phys', f'inhibited_pMEK_phys']
            meks2 = [f'drugfree_pMEK_onco', f'inhibited_pMEK_onco']
            values = ['drugfree_pMEK_onco', 'drugfree_pMEK_phys']

        for meks in meks1, meks2:
            df_drug_free[meks] = \
                df_drug_free[meks].div(
                    df_drug_free[meks].sum(axis=1),
                    axis=0
                )
    else:
        raise ValueError(f'invalid iterator {iterator}')

    df_melt = pd.melt(df_drug_free, id_vars=[iterator, 'par_index'],
                      value_vars=values)
    df_melt.variable = pd.Categorical(df_melt.variable, ordered=True,
                                      categories=values)

    plot = (
        ggplot(df_melt,
               aes(x=iterator, y='value', group='variable',
                   color='variable', fill='variable'))
        + xlab(xlabel)
        + scale_x_log10(expand=(0.01, 0.0))
        + ylab(f'drug free {iterator.replace("i_0","")} molecules')
        + scale_y_continuous(expand=(0.01, 0.0), limits=(0.0, 1.0))
        + scale_color_cmap_d(name='tab10')
        + scale_fill_cmap_d(name='tab10')
        + theme('minimal', figure_size=(
                    PLOTNINE_FIGWIDTH / 3, PLOTNINE_FIGWIDTH / 3 * 2 / 3)
                ))
    plot = median_with_uncertainty(plot)

    plot_and_save_fig(plot, figdir, filename)


def plot_raf_dimerization(df, iterator, xlabel, figdir, filename):
    df_melt = pd.melt(df, id_vars=[iterator, 'par_index'],
                      value_vars=['RAF_marginal_RAS', 'RAF_marginal_RAF'])

    plot = (
        ggplot(df_melt,
               aes(x=iterator, y='value', group='variable',
                   color='variable', fill='variable'))
        + xlab(xlabel)
        + scale_x_log10()
        + ylab(f'fraction of RAF molecules')
        + scale_color_cmap_d(name='tab10')
        + scale_fill_cmap_d(name='tab10')
        + theme('minimal', figure_size=(
                    PLOTNINE_FIGWIDTH / 3, PLOTNINE_FIGWIDTH / 3 * 2 / 3)
                ))
    plot = median_with_uncertainty(plot)
    plot_and_save_fig(plot, figdir, filename)


def median_with_uncertainty(plot):
    return (
        plot
        + stat_summary(fun_y=np.median,
                       fun_ymin=lambda x: np.quantile(x, 0.40),
                       fun_ymax=lambda x: np.quantile(x, 0.60),
                       geom='ribbon', alpha=0.2, colour=None)
        + stat_summary(fun_y=np.median,
                       fun_ymin=lambda x: np.quantile(x, 0.30),
                       fun_ymax=lambda x: np.quantile(x, 0.70),
                       geom='ribbon', alpha=0.2, colour=None)
        + stat_summary(fun_y=np.median,
                       fun_ymin=lambda x: np.quantile(x, 0.20),
                       fun_ymax=lambda x: np.quantile(x, 0.80),
                       geom='ribbon', alpha=0.2, colour=None)
        + stat_summary(fun_y=np.median,
                       fun_ymin=lambda x: np.quantile(x, 0.10),
                       fun_ymax=lambda x: np.quantile(x, 0.90),
                       geom='ribbon', alpha=0.2, colour=None)
        + stat_summary(fun_y=np.median, geom='line', size=1)
    )


def set_xy_scale(plot, logy, logx, logz, ylabel, xlabel, ylimits,
                 xlimits, zlimits, cmap='viridis', adapt_color_scale=True,
                 adapt_fill_scale=True, expand=True):

    # YSCALE
    if logy:
        y_scale = scale_y_log10
        y_labels = lambda lbl: log_breaks_labels(lbl, ylabel)
    else:
        y_scale = scale_y_continuous
        y_labels = mizani.formatters.mpl_format()

    yoptions = dict(
        labels=y_labels
    )

    if ylimits:
        yoptions['limits'] = ylimits
    else:
        ylimits = [0.0]

    if not expand:
        yoptions['expand'] = (0.01, min(0.0, ylimits[0]), 0.01, 0.0)

    plot += y_scale(**yoptions)

    # XSCALE
    if logx:
        xscale = scale_x_log10
        xlabels = lambda lbl: log_breaks_labels(lbl, xlabel)
    else:
        xscale = scale_x_continuous
        xlabels = mizani.formatters.mpl_format()

    xoptions = dict(
        labels=xlabels
    )

    if xlimits:
        xoptions['limits'] = xlimits
    else:
        xlimits = [0.0]

    if not expand:
        xoptions['expand'] = (0.01, xlimits[0])

    plot += xscale(**xoptions)

    # ZSCALE
    zlimargs = {
        'cmap_name': cmap,
    }
    if logz:
        zlimargs['trans'] = mizani.transforms.log10_trans()
    if zlimits is not None:
        zlimargs['limits'] = zlimits

    if adapt_color_scale:
        plot += scale_color_continuous(**zlimargs)
    if adapt_fill_scale:
        plot += scale_fill_continuous(**zlimargs)


def log_breaks_labels(breaks, indexvar):

    breaks = mizani.formatters.scientific_format()(breaks)
    for ib, b in enumerate(breaks):
        if (b == '1.e-04') & (indexvar == 'RAFi_0'):
            breaks[ib] = '0.0'
        elif (b == '1.e-05') & (indexvar == 'MEKi_0'):
            breaks[ib] = '0.0'
        else:
            exponent = int(b[b.index('e')+1:])
            factor = float(b[:b.index('e')])
            if factor == 1.0:
                breaks[ib] = f'$10^{{{exponent}}}$'
            else:
                breaks[ib] = f'${factor} \cdot 10^{{{exponent}}}$'

    return breaks


def plot_and_save_fig(plot, figdir, filename):
    plt.tight_layout()
    if figdir is None:
        figdir = os.path.join(os.getcwd(), 'figures')

    if not os.path.exists(figdir):
        os.makedirs(figdir, exist_ok=True)

    if filename is not None:
        plot.save(os.path.join(figdir, filename))
