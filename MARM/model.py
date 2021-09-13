import pysb
import pysb.bng
import os
import importlib

import sympy as sp

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


def write_aux_model_functions(model):
    simplify_energy_rates(model)


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
