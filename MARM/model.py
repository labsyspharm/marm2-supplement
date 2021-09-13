import pysb
import pysb.bng
import pysb.export
import os
import importlib
import re
import itertools
import copy
import amici.pysb_import
import logging

import sympy as sp

from shutil import copyfile
from pysb.bng import BngFileInterface

from .paths import get_model_instance_name, get_model_name_variant

CONSTANTS = [
    'RAFi_0', 'MEKi_0', 'EGF_0', 'EGFR_crispr', 'NRAS_Q61mut',
]


def cleanup_unused(model):

    model.reset_equations()
    pysb.bng.generate_equations(model)

    observables = [
        obs.name for obs in model.expressions
        if obs.name.endswith('_obs')
    ]

    dynamic_eq = sp.Matrix(model.odes)

    expression_dynamic_symbols = set()
    for sym in dynamic_eq.free_symbols:
        if str(sym) in model.expressions.keys():
            expression_dynamic_symbols |= model.expressions[
                str(sym)
            ].expand_expr().free_symbols

    initial_eq = sp.Matrix([
        initial.value.expand_expr()
        for initial in model.initials
    ])

    observable_eq = sp.Matrix([
        expression.expand_expr()
        for expression in model.expressions
        if expression.name in observables
    ])

    free_symbols = list(
        dynamic_eq.free_symbols | initial_eq.free_symbols |
        observable_eq.free_symbols | expression_dynamic_symbols
    )

    unused_pars = set(
        par
        for par in model.parameters
        if par not in free_symbols and sp.Symbol(par.name) not in free_symbols
    )

    rule_reaction_count = {
        rule.name: 0
        for rule in model.rules
    }

    for reaction in model.reactions:
        for rule in reaction['rule']:
            rule_reaction_count[rule] += 1

    model.parameters = pysb.ComponentSet([
        par for par in model.parameters
        if par not in unused_pars
    ])

    model.expressions = pysb.ComponentSet([
        expr for expr in model.expressions
        if len(expr.expand_expr().free_symbols.intersection(unused_pars)) == 0
        and not expr.name.startswith('_')
    ])

    model.rules = pysb.ComponentSet([
        rule for rule in model.rules
        if rule_reaction_count[rule.name] > 0
    ])

    model.energypatterns = pysb.ComponentSet([
        ep for ep in model.energypatterns
        if len(ep.energy.expand_expr().free_symbols.intersection(
            unused_pars)) == 0
    ])

    model.reset_equations()


def get_model_instance(name, variant, instance, instances):

    full_name = get_model_name_variant(name, variant)

    model_variant = importlib.import_module(f'.pysb_flat.{full_name}',
                                            'MARM').model

    # don't touch the variant as we don't want to propagate changes to
    # future loading of the model
    model_instance = pysb.Model(base=model_variant)

    instance_initials = [
        instances[instance_var]
        for instance_var in instance.split('_')
        if instance_var not in ['', 'EGFR']
    ]

    model_instance.initials = [
        initial
        for initial in model_instance.initials
        if initial.value.name not in instances.values()
        or initial.value.name in instance_initials
    ]

    model_instance.name = get_model_instance_name(name, variant, instance,
                                                  None)

    return model_instance


def generate_equations(model, verbose=False):
    if model.reactions:
        model.reset_equations()
    pysb.bng.generate_equations(model, verbose=verbose)


def export_model(model, formats):
    simplify_energy_rates(model)
    base_dir = os.path.dirname(__file__)
    export_formats(model, base_dir, formats)


def simplify_energy_rates(model):
    for expr in model.expressions:
        if re.search(r'^(_bind|__reverse)', expr.name) \
                and re.search(r'_local[0-9]+$', expr.name):
            model.components[expr.name].expr = simplify_rate(expr.expand_expr())


def simplify_rate(mul):
    return sp.powsimp((sp.expand_power_base(sp.powdenest(sp.logcombine(
        sp.expand_log(mul, force=True),
        force=True), force=True), force=True)), force=True)


def export_formats(model, base_dir, formats):
    for language in formats:
        file_dir = os.path.join(base_dir, language)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if language == 'pysb_flat':
            suffix = 'py'
        elif language == 'latex':
            suffix = 'tex'
        else:
            suffix = language
        file = os.path.join(file_dir, '{}.{}'.format(model.name, suffix))

        if language == 'latex':
            with BngFileInterface(model, verbose=False, cleanup=True) as \
                    bngfile:
                bngfile.action('generate_network', overwrite=True)
                bngfile.action('writeLatex', overwrite=True, verbose=False)
                bngfile.execute()
                copyfile(f'{bngfile.base_filename}.tex', file)
        else:
            with open(file, 'w') as f:
                f.write(pysb.export.export(model, language))


def compile_model(model):
    base_dir = os.path.dirname(__file__)

    observables = [
        obs.name for obs in model.expressions
        if obs.name.endswith('_obs')
    ]

    outdir = os.path.join(base_dir, 'build', model.name)

    amici.pysb_import.pysb2amici(model,
                                 outdir,
                                 verbose=logging.INFO,
                                 observables=observables,
                                 constant_parameters=CONSTANTS,
                                 compute_conservation_laws=True)


def add_monomer_configuration_observables(model):
    for monomer in model.monomers:
        site_states = copy.copy(model.components[monomer.name].site_states)
        for site in model.components[monomer.name].sites:
            if site not in site_states:
                site_states[site] = [None, pysb.ANY]

        combos = list(itertools.product(*site_states.values()))
        explicit_states = [
            {key: combo[ikey] for ikey, key in enumerate(site_states.keys())}
            for combo in combos
        ]

        for state in explicit_states:
            pysb.Observable(
                expl_name(model.components[monomer.name](**state)),
                model.components[monomer.name](**state)
            )


def expl_name(name):
    name = str(name)
    name = re.sub(r', [\w]*=None', '', name)
    name = re.sub(r'\([\w]*=None, ', '(', name)
    name = re.sub(r'\([\w]*=None\)', '__mono', name)
    name = re.sub(r'\(\)', '__mono', name)
    name = re.sub(r'AA([0-9]+)=', '\g<1>', name)
    name = name.replace('=ANY', '')
    name = name.replace(' ', '')
    name = name.replace("'", "")
    name = name.replace(',', '_')
    name = name.replace('=', '')
    name = name.replace('(', '__')
    name = name.replace(')', '')

    return 'expl_' + name


def add_monomer_label(model, monomer, name, label_site,
                      act_rules, deact_rule):

    channels = list(act_rules.keys())
    mono = model.monomers[monomer]
    mono.sites += ['channel']
    mono.site_states['channel'] = channels + ['NA']

    for initial in model.initials:
        for mp in initial.pattern.monomer_patterns:
            if mp.monomer.name == monomer:
                mp.site_conditions['channel'] = 'NA'
                if hasattr(initial.pattern, 'canonical_repr'):
                    initial.pattern.canonical_repr = None

    for channel in channels:
        pysb.Expression(
            f'{name}_{channel}_obs',
            model.parameters[f'{name}_IF_scale'] *
            pysb.Observable(
                f'{name}_{channel}',
                mono(channel=channel)
            )/model.observables[f't{monomer}']
        )

    for ia, (channel, rules) in enumerate(act_rules.items()):
        for rule in rules:
            if rule not in model.rules.keys():
                continue

            for cp in model.rules[rule].reactant_pattern.complex_patterns + \
                    model.rules[rule].product_pattern.complex_patterns:
                for mp in cp.monomer_patterns:
                    if mp.monomer.name == monomer \
                            and label_site[0] in mp.site_conditions:
                        if mp.site_conditions[label_site[0]] == label_site[1]:
                            mp.site_conditions['channel'] = channel
                        else:
                            mp.site_conditions['channel'] = 'NA'

        if deact_rule in model.rules.keys():
            for cp in model.rules[deact_rule].reactant_pattern.complex_patterns\
                    + model.rules[deact_rule].product_pattern.complex_patterns:
                for mp in cp.monomer_patterns:
                    if mp.monomer.name == monomer \
                            and label_site[0] in mp.site_conditions:
                        if mp.site_conditions[label_site[0]] == label_site[1]:
                            mp.site_conditions['channel'] = channel
                        else:
                            mp.site_conditions['channel'] = 'NA'

        if ia < len(channels) - 1:
            rule_copy = copy.deepcopy(model.rules[deact_rule])
            rule_copy.name = f'{deact_rule}_{channel}'
            model.add_component(rule_copy)
        else:
            model.rules[deact_rule].name = f'{deact_rule}_{channel}'

    return model


def propagate_monomer_label(model, source_monomer, target_monomer, name,
                            channels, trans_rule, rem_rule,
                            label_site):

    mono = model.monomers[target_monomer]
    mono.sites += ['channel']
    mono.site_states['channel'] = channels + ['NA']

    for initial in model.initials:
        for mp in initial.pattern.monomer_patterns:
            if mp.monomer.name == target_monomer:
                mp.site_conditions['channel'] = 'NA'
                if hasattr(initial.pattern, 'canonical_repr'):
                    initial.pattern.canonical_repr = None

    for ia, channel in enumerate(channels):
        pysb.Expression(
            f'{name}_{channel}_obs',
            model.parameters[f'{name}_IF_scale'] *
            pysb.Observable(
                f'{name}_{channel}',
                mono(channel=channel)
            )/model.observables[f't{target_monomer}']
        )

        if trans_rule not in model.rules.keys():
            continue

        for cp in model.rules[trans_rule].reactant_pattern.complex_patterns + \
                  model.rules[trans_rule].product_pattern.complex_patterns:
            for mp in cp.monomer_patterns:
                if mp.monomer.name == target_monomer \
                        and label_site[0] in mp.site_conditions:
                    if mp.site_conditions[label_site[0]] == label_site[1]:
                        mp.site_conditions['channel'] = channel
                    else:
                        mp.site_conditions['channel'] = 'NA'
                if mp.monomer.name == source_monomer \
                        and mp.site_conditions.get(label_site[0], '') == label_site[1]:
                    mp.site_conditions['channel'] = channel

        if ia < len(channels) - 1:
            rule_copy = copy.deepcopy(model.rules[trans_rule])
            rule_copy.name = f'{trans_rule}_{channel}'
            model.add_component(rule_copy)
        else:
            model.rules[trans_rule].name = f'{trans_rule}_{channel}'

        if rem_rule in model.rules.keys():
            for cp in model.rules[rem_rule].reactant_pattern.complex_patterns + \
                      model.rules[rem_rule].product_pattern.complex_patterns:
                for mp in cp.monomer_patterns:
                    if mp.monomer.name == target_monomer \
                            and label_site[0] in mp.site_conditions:
                        if mp.site_conditions[label_site[0]] == label_site[1]:
                            mp.site_conditions['channel'] = channel
                        else:
                            mp.site_conditions['channel'] = 'NA'

        if ia < len(channels) - 1:
            rule_copy = copy.deepcopy(model.rules[rem_rule])
            rule_copy.name = f'{rem_rule}_{channel}'
            model.add_component(rule_copy)
        else:
            model.rules[rem_rule].name = f'{rem_rule}_{channel}'