import networkx as nx
import itertools
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.animation
import os
from matplotlib.patches import ArrowStyle

from .common import plot_and_save_fig, process_rules, \
    get_rule_patterns

from pysb.pattern import match_complex_pattern

NETWORKX_FIGWIDTH = 15


def aggregate_pattern(pattern, level, pathways):
    if level == 'protein':
        nodes = [mp.monomer.name for mp in pattern.monomer_patterns]
    elif level == 'complex':
        nodes = ['#'.join(sorted([
            mp.monomer.name for mp in pattern.monomer_patterns
        ]))]
    elif level == 'pathway':
        nodes = list(set(
            pathway
            for mp in pattern.monomer_patterns
            for pathway, pathway_members in pathways.items()
            if mp.monomer.name in pathway_members
        ))
    else:
        raise ValueError(f'Unsupported level {level}!')

    return nodes


def aggregate_rules(rule, pathways):
    nodes = list(set(
        pathway
        for pathway, pathway_members in pathways.items()
        if rule in pathway_members
    ))

    return nodes


def node_aggregate_graph(graph, pathways=None):
    G = nx.MultiDiGraph(model=graph.graph['model'])

    for u, data in graph.nodes(data=True):
        nodes = aggregate_rules(u, pathways=pathways)

        for node in nodes:
            if node not in G.nodes():
                G.add_node(node, rules=data['rules'],
                           patterns=data['patterns'])
            else:
                G.nodes[node]['rules'] |= data['rules']
                G.nodes[node]['patterns'].update(data['patterns'])

    added_rule_labels = []
    for u, v, data in graph.edges(data=True):
        source_nodes = aggregate_rules(u, pathways=pathways)
        target_nodes = aggregate_rules(v, pathways=pathways)

        for source_node, target_node in itertools.product(
            source_nodes, target_nodes,
        ):
            if source_node == target_node:
                continue
            rule_label = f'{data["flux_label"].split("::")[0]}' \
                f'_{target_node}_{data["inhibition"]}_{data["activation"]}'
            if rule_label not in added_rule_labels:
                G.add_edge(
                    source_node,
                    target_node,
                    **data
                )
                added_rule_labels.append(rule_label)

    for u, udata in G.nodes(data=True):
        for (((a, b, k, abdata), (c, d, l, cddata)),) in zip(itertools.product(
                G.in_edges(u, data=True, keys=True),
                G.out_edges(u, data=True, keys=True)
        )):
            source = abdata['flux_label'].split('::')[0]
            target = cddata['flux_label'].split('::')[1]
            edge_label = f'{source}::{target}'
            target_rule = target.split(' ')[0]
            paths = nx.all_simple_paths(graph, source, target_rule)
            for path in paths:
                flux_label = f'{path[-2]}::{target}'
                G.add_edge(
                    u, u,
                    edge_label=edge_label,
                    flux_label=flux_label
                )

    return G


def rule2pattern_graph(rule_graph):
    rule_graph = rule_graph.copy()
    patterns = get_rule_patterns(rule_graph.graph['model'])

    G = nx.MultiDiGraph(model=rule_graph.graph['model'])

    for u, v, data in rule_graph.edges(data=True):
        if data['edge_label'] != data['flux_label']:
            continue
        rule, pattern_label = data['edge_label'].split('::')
        pattern = patterns[pattern_label]['pattern']
        if pattern_label not in G.nodes():
            G.add_node(
                str(pattern),
                edges=[(u, v)]
            )
        else:
            G.nodes[pattern_label]['edges'].append((u, v))

    edge_labels = []
    for u, v, edata in rule_graph.edges(data=True):
        rule, pattern = edata['edge_label'].split('::')
        for node, ndata in rule_graph.nodes(data=True):
            if rule in ndata['rules']:
                for rule_pattern in ndata['patterns'][rule].keys():
                    x = str(patterns[rule_pattern]['pattern'])
                    if x not in G.nodes():
                        continue
                    y = str(patterns[pattern]['pattern'])
                    if y not in G.nodes():
                        continue
                    edge_label = f'{u}_{v}_{node}'
                    if edge_label in edge_labels:
                        continue
                    G.add_edge(
                        x,
                        y,
                        rules=[rule],
                    )
                    edge_labels.append(edge_label)

    return G


def pattern2rule_graph(pattern_graph):
    G = nx.MultiDiGraph(model=pattern_graph.graph['model'])

    for u, v, data in pattern_graph.edges(data=True):
        node = data['rule']
        if node.startswith('init'):
            continue
        if node not in G.nodes():
            G.add_node(node, edges=[(u, v)])
        else:
            G.nodes[node]['edges'].append((u, v))

    for u, data in pattern_graph.nodes(data=True):
        if 'rule' not in data:
            continue

        in_edges = pattern_graph.in_edges(u, keys=True)
        out_edges = pattern_graph.out_edges(u, keys=True)
        for w in itertools.product(in_edges, out_edges):
            source_node = pattern_graph.edges[v]['rule']
            target_node = pattern_graph.edges[w]['rule']
            if source_node.startswith('init') or \
                    target_node.startswith('init'):
                continue

            G.add_edge(
                source_node, target_node,
                **data
            )

        for w in out_edges:
            if 'rule' in data and \
                    data['rule'] != pattern_graph.edges[w]['rule']:
                G.add_edge(
                    pattern_graph.edges[w]['rule'], data['rule'],
                    **data
                )

    return G


def create_rule_graph(model, ignore_rules=None):
    processed_rules = process_rules(model, ignore_rules=ignore_rules)
    patterns = get_rule_patterns(model)

    G = nx.MultiDiGraph(model=model)

    for rule, rule_data in processed_rules.items():
        rule_patterns = {
            f'{rule} E{ip}': patterns[f'{rule} E{ip}']
            for ip, pattern in enumerate(rule_data['raw_input'])
            if f'{rule} E{ip}' in patterns
        }
        if len(patterns) and len(rule_data['reaction_idx']):
            G.add_node(
                f'{rule}',
                rules={rule},
                patterns={rule: rule_patterns},
            )

    for node, node_data in G.nodes(data=True):
        for rule, rule_data in processed_rules.items():
            for pattern_label, pattern in node_data['patterns'][node].items():
                pos_rxns = [
                    r_idx
                    for r_idx in rule_data['reaction_idx']
                    if sum(
                        pattern['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['products']
                    ) > sum(
                        pattern['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['reactants']
                    )
                ]
                neg_rxns = [
                    r_idx
                    for r_idx in rule_data['reaction_idx']
                    if sum(
                        pattern['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['products']
                    ) < sum(
                        pattern['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['reactants']
                    )
                ]
                rxns = pos_rxns + neg_rxns
                label = f'{rule}::{pattern_label}'
                if len(rxns) > 0:
                    G.add_edge(
                        rule, node,
                        flux_label=label,
                        edge_label=label,
                        reaction_idx=rxns,
                        inhibition=len(neg_rxns) > 0,
                        activation=len(pos_rxns) > 0,
                    )
    return G


def create_pattern_graph(model, ignore_rules=None):
    processed_rules = process_rules(model, ignore_rules=ignore_rules)
    patterns = get_rule_patterns(model)

    G = nx.MultiDiGraph(model=model)

    # graph should only have inputs, outputs are only there if sink states,
    # every node is associated with a list of species
    # every edge is associated with a list of reactions and a rule
    for pattern, data in patterns.items():
        rule, educt_id = pattern.split(' ')
        if len(data['species']):
            G.add_node(
                f'{rule}#{data["pattern"]}',
                label=str(data["pattern"]),
                species=data['species'],
                pattern=data['pattern'],
                rule=rule,
            )

    for inode in G.nodes(data=True):
        for jnode in G.nodes(data=True):
            rule_name = jnode[1]['rule']
            rule = processed_rules[rule_name]
            # does the jnode rule produces species for inode pattern?
            connecting_species_idx = set(
                inode[1]['species'].keys()
            ).intersection(
                set(rule['product_idx'])
            )
            if len(connecting_species_idx):
                # compute list of reactions that increase pattern match in
                # inode, this corresponds to an activation of the respective
                # rule
                rxns = [
                    r_idx
                    for r_idx in rule['reaction_idx']
                    if sum(
                        inode[1]['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['products']
                    ) > sum(
                        inode[1]['species'].get(
                            ix, 0.0
                        )
                        for ix in model.reactions[r_idx]['reactants']
                    )
                ]
                if len(rxns) > 0:
                    G.add_edge(
                        jnode[0], inode[0],
                        reaction_idx=rxns,
                        rule=rule['name'],
                    )

    for inode in list(G.nodes(data=True)):
        for init in model.initial_conditions:
            if match_complex_pattern(inode[1]['pattern'], init[0]):
                G.add_node(
                    str(init[1].name),
                    species=next(
                        specie
                        for specie in model.species
                        if match_complex_pattern(init[0], specie, exact=True)
                    ),
                    pattern=init[0],
                    label=str(init[0]),
                    input=True,
                )

                G.add_edge(
                    str(init[1].name), inode[0],
                    reaction_idx=[],
                    rule=init[1].name,
                )

    return G


def get_positions_as_numpy(pos):
    return np.asarray([np.asarray(x) for x in pos.values()])


def compute_angle(v):
    if np.linalg.norm(v) == 0:
        return 0

    u = np.array([0, 1])

    return np.arccos(np.clip(np.dot(
        v/np.linalg.norm(v), u/np.linalg.norm(u)
    ), -1.0, 1.0))


def center_and_rotate_positions(pos):
    if not len(pos):
        return pos

    # center
    pos_array = get_positions_as_numpy(pos)
    mean_pos = pos_array.mean(axis=0)
    pos_array = pos_array - mean_pos
    if len(pos) > 1:
        # rotate
        rot_angle = compute_angle(
            pos_array[compute_radius_array(pos_array).argmax(), :]
        )
        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ])
        pos_array = pos_array.dot(rot_matrix)

    return {
        key: tuple(
            np.around(x, 0) for x in pos_array[idx, :]
        )
        for idx, key in enumerate(pos.keys())
    }


def recenter_positions(pos, center):
    if not len(pos):
        return pos

    # center
    pos_array = get_positions_as_numpy(pos)
    pos_array = pos_array + center

    return {
        key: tuple(
            np.around(x, -1) for x in pos_array[idx, :]
        )
        for idx, key in enumerate(pos.keys())
    }


def compute_radius_pos(pos):
    if len(pos) <= 1:
        return np.zeros((1,))
    pos_array = get_positions_as_numpy(pos)
    return compute_radius_array(pos_array)


def compute_radius_array(pos_array):
    return np.sqrt(np.power(pos_array, 2.0).sum(axis=1))


def plot_network_graph(graph, figdir, figname, algorithm='neato',
                       ignore_nodes=None):

    G = graph.copy()
    if ignore_nodes is None:
        ignore_nodes = []

    for n in ignore_nodes:
        G.remove_node(n)

    H = G.__class__()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges(keys=True))

    pos = nx.nx_pydot.graphviz_layout(
        H,
        prog=algorithm
    )

    node_style = {
        'node_size': 4800/np.sqrt(len(pos)),
    }
    label_style = {
        'font_size': 48/np.sqrt(len(pos))
    }

    input_nodes = [
        u
        for u in G.nodes
        if G.in_degree(u) == 0
    ]

    output_nodes = [
        u
        for u in G.nodes
        if G.out_degree(u) == 0
    ]

    regular_nodes = [
        u
        for u in G.nodes
        if u not in input_nodes
        and u not in output_nodes
    ]

    fig, ax = plt.subplots(figsize=(NETWORKX_FIGWIDTH, NETWORKX_FIGWIDTH))

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=input_nodes,
                           node_color='c',
                           **node_style)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=output_nodes,
                           node_color='r',
                           **node_style)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=regular_nodes,
                           node_color='0.9',
                           **node_style)

    draw_flux_edges(graph, pos, ax)

    nx.draw_networkx_labels(graph, pos,
                            **label_style)

    plot_and_save_fig(figdir, figname)

    return G, pos


def update_fluxgraph(index, G, pos, ax, flux_data, normalize_single,
                        index_name):
    ax.clear()

    node_style = {
        'node_size': 480/np.sqrt(len(pos)),
    }
    edge_style = {
        'width': 4 / np.sqrt(len(pos)),
        'arrowsize': 12/np.sqrt(len(pos)),
    }
    label_style ={
        'font_size': 15/np.sqrt(len(pos))
    }

    for u, v, k in G.edges(keys=True):
        G.edges[(u, v, k)]['activation_flux'] = 0.0
        G.edges[(u, v, k)]['inhibition_flux'] = 0.0

    for node, data in G.nodes(data=True):
        for rule in data['rules']:
            for pattern in data['patterns'][rule]:
                fluxes = flux_data[[
                    col for col in flux_data.columns
                    if col.endswith(pattern)
                ]].copy()

                for u, v, k, edata in G.in_edges(node, keys=True,
                                                 data=True):
                    flux_label = edata['flux_label']

                    # check if
                    if flux_label not in fluxes.columns:
                        continue

                    # normalize
                    if normalize_single:
                        norm_factor = \
                            fluxes[flux_label].apply(abs).max()
                    else:
                        norm_factor = \
                            fluxes.apply(abs).max(axis=0).max()

                    flux = fluxes.iloc[index][flux_label]/norm_factor
                    if flux > 0:
                        G.edges[(u, v, k)]['activation_flux'] = \
                            abs(flux)
                    else:
                        G.edges[(u, v, k)]['inhibition_flux'] = \
                            abs(flux)

    draw_flux_edges(G, pos, ax=ax)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color='0.9',
                           **node_style)

    nx.draw_networkx_labels(G, pos, ax=ax,
                            **label_style)

    if index_name == 'RAFi_0':
        dose_label = ' $\mu$M Vemurafenib'
    elif index_name == 'MEKi_0':
        dose_label = ' $\mu$M Cobimetinib'
    else:
        dose_label = f'[$\mu$M] {index_name.replace("_0","")}'

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{flux_data.index[index]:.4f} {dose_label}')


INHIBITION_ARROW = ArrowStyle.BracketA(
    widthA=0.25,
    lengthA=0.2,
    angleA=None,
)

ACTIVATION_ARROW = ArrowStyle.CurveFilledA(
    head_length=0.3,
    head_width=0.15,
)


def draw_flux_edges(G, pos, ax):
    for u, v, k, data in G.edges(keys=True, data=True):
        pos1 = pos[u]
        pos2 = pos[v]

        if 'inhibition_flux' in data and 'activation_flux' in data:
            color = (
                data['inhibition_flux'],
                0.0,
                data['activation_flux'],
                max(data['activation_flux'], data['inhibition_flux'])
            )

            if data['inhibition_flux'] > 0:
                arrowstyle = INHIBITION_ARROW
            else:
                arrowstyle = ACTIVATION_ARROW

        elif 'inhibition' in data and 'activation' in data:
            color = (
                float(data['inhibition']),
                0.0,
                float(data['activation']),
                1
            )
            if data['inhibition']:
                arrowstyle = INHIBITION_ARROW
            else:
                arrowstyle = ACTIVATION_ARROW
        else:
            color = '0.5'
            arrowstyle = "->"

        ax.annotate(
            "",
            xy=pos1, xycoords='data',
            xytext=pos2, textcoords='data',
            arrowprops=dict(
                arrowstyle=arrowstyle,
                color=color,
                shrinkA=15, shrinkB=15,
                patchA=None,
                patchB=None,
                connectionstyle=f"arc3,rad=-{(k + 1) * 0.05}",
            ),
        )


def plot_network_doseresponse(graph, doseresponse, dose_name,
                               algorithm='neato', figdir=None, filename=None,
                               normalize_single=True):

    G = graph.copy()
    G.remove_edges_from(list(G.selfloop_edges()))

    H = G.__class__()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges(keys=True))

    pos = nx.nx_pydot.graphviz_layout(
        H,
        prog=algorithm
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ani = matplotlib.animation.FuncAnimation(
        fig, update_fluxgraph, frames=len(doseresponse), interval=100,
        repeat=True, fargs=(G, pos, ax, doseresponse, normalize_single,
                              dose_name)
    )
    plt.show()

    ani.save(os.path.join(
            figdir,
            filename.format(plottype=f'graph_flux')
        ),
        dpi=300
    )


def plot_network_timecourse(graph, timecourse,
                            algorithm='neato', figdir=None, filename=None,
                            normalize_single=True):

    G = graph.copy()
    G.remove_edges_from(list(G.selfloop_edges()))

    H = G.__class__()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges(keys=True))

    pos = nx.nx_pydot.graphviz_layout(
        H,
        prog=algorithm
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ani = matplotlib.animation.FuncAnimation(
        fig, update_fluxgraph, frames=len(timecourse), interval=50,
        repeat=True, fargs=(G, pos, ax, timecourse, normalize_single,
                            't [h]')
    )
    plt.show()

    ani.save(os.path.join(
            figdir,
            filename.format(plottype=f'graph_flux')
        ),
        dpi=300
    )


def get_modules(model, level):
    if not hasattr(model, 'source_models') or level == 0:
        return [model]

    modules = []
    for submodel in model.source_models:
        modules.extend(get_modules(submodel, level-1))

    return set(modules)


def get_module_rules(model, level):
    modules = get_modules(model, level)
    return {
        module.name: [rule.name for rule in module.rules]
        for module in modules
        if module is not None
    }


def process_rule_annotation(model, mode):
    annotated_rules = dict()

    if mode == 'protein':
        def object_filter(x):
            return '#' not in x
    elif mode == 'site':
        def object_filter(x):
            return '#' in x

    for annotation in model.annotations:
        if annotation.predicate not in ['rule_has_object', 'rule_has_subject']\
                or not object_filter(annotation.object):
            continue

        if annotation.subject not in annotated_rules:
            annotated_rules[annotation.subject] = {
                'objects': set(),
                'subjects': set(),
            }

        annotated_rules[annotation.subject][
            predicate_mapping[annotation.predicate]
        ] |= {annotation.object}

    return annotated_rules


def subdict(d, keys):
    return dict((key, d[key]) for key in keys)


def create_signaling_graph(df, colors, iterator):
    G = nx.DiGraph()

    G.add_nodes_from([
        (node, dict(fillstyle=nodeset['fillstyle'], color=nodeset['color']))
        for nodeset in get_graph_nodes(colors)
        for node in nodeset['nodes']
    ])

    edge_data = df[df.variable.apply(lambda x: '_to_' in x)].drop(
        ['value', iterator], axis=1
    ).drop_duplicates()
    G.add_edges_from([
        tuple(row.variable.split('_to_')[::-1] + [{
            'type': row.channel,
            'color': colors[row.channel],
        }])
        for ir, row in edge_data.iterrows()
    ])
    return G


def get_graph_nodes(colors):
    return [
        {
            'nodes': ['active EGFR', 'RASgtp'],
            'fillstyle': 'full',
            'color': colors['phys']
        },
        {
            'nodes': ['phys pMEK', 'phys pERK'],
            'fillstyle': 'left',
            'color': colors['phys']
        },
        {
            'nodes': ['BRAFV600E'],
            'fillstyle': 'full',
            'color': colors['onco']
        },
        {
            'nodes': ['onco pMEK', 'onco pERK'],
            'fillstyle': 'right',
            'color': colors['onco']
        }
    ]


def get_graph_pos(deltax_outer, deltay_aligned, deltay_unaligned):
    return {
        'active EGFR': (-deltax_outer, deltay_unaligned + deltay_aligned),
        'RASgtp': (-deltax_outer, deltay_unaligned),
        'phys pMEK': (0, 0),
        'phys pERK': (0, -deltay_aligned),
        'BRAFV600E': (deltax_outer, deltay_unaligned),
        'onco pMEK': (0, 0),
        'onco pERK': (0, -deltay_aligned),
    }


def plot_contextualized_graph(df, iterator, n_plots, figdir, figname,
                              lw=1.5):
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 4))

    groupvars = [iterator, 'variable', 'step', 'channel']
    df_transduction_melt = pd.DataFrame([
        dict(value=values.value.median(), **dict(zip(groupvars, cond)))
        for cond, values in df[df.step<4].groupby(groupvars)
    ])
    df_transduction = df_transduction_melt.pivot(index=iterator,
                                                 columns='variable')
    df_transduction = df_transduction.reset_index().sort_values(iterator)

    colors = {
        'phys': '#F57F20',
        'onco': '#2278B5'
    }

    G = create_signaling_graph(df_transduction_melt, colors, iterator)

    deltax_outer = 0.3
    deltay_aligned = 1.0
    deltay_unaligned = 1.0
    pos = get_graph_pos(deltax_outer, deltay_aligned, deltay_unaligned)

    for iplot in range(n_plots):
        transduction = df_transduction.iloc[
            int(np.round(iplot / (n_plots - 1) * (len(df_transduction) - 1)))]
        ax = axes[iplot]
        for node, data in G.nodes(data=True):
            nx.draw_networkx_nodes(
                G, pos=pos, node_size=1000,
                nodelist=[node],
                node_shape=matplotlib.markers.MarkerStyle(
                                marker='o', fillstyle=data['fillstyle']
                           ),
                node_color=[data['color']],
                alpha=transduction[('value', node)],
                edgecolors='k',
                linewidths=lw,
                ax=ax,
            )

        rad = {
            'phys': 0.2,
            'onco': -0.2
        }

        for u, v, data in G.edges(data=True):
            pos1 = pos[u]
            pos2 = pos[v]

            ax.annotate(
                "",
                xy=pos1, xycoords='data',
                xytext=pos2, textcoords='data',
                arrowprops=dict(
                    arrowstyle='simple',
                    color=data['color'],
                    shrinkA=20, shrinkB=20,
                    patchA=None,
                    patchB=None,
                    alpha=min([
                        transduction[('value', '_to_'.join((v, u)))] / 2,
                        1.0
                    ]),
                    mutation_scale=20,
                    connectionstyle=f'arc3,rad={rad[data["type"]]}',
                ),
            )
        ax.set_ylim(
            (-(deltay_aligned + 0.5), deltay_unaligned + deltay_aligned + 0.5))
        ax.set_xlim((-(deltax_outer + 0.5), deltax_outer + 0.5))
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(figdir, figname))


