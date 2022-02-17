import amici
import os
import pysb
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..estimation import get_solver

from collections import Counter
from scipy.constants import N_A
from astropy import units
from pysb.pattern import SpeciesPatternMatcher

ATOL_MONO_FRACTIONS = 1e-12
ATOL_FLUXES = 1e-10

MPL_FIGWIDTH = 15


def check_mode(mode):
    modes = ['monomer', 'flux', 'reaction', 'pattern', 'pattern_ab']

    if mode not in modes:
        raise ValueError(f'Invalid mode selected, must be in {modes}')


def run_doseresponse_simulation(amici_model, dose_name, doses, preequidict,
                                preequilibrate_dose_response=True,
                                preequilibrate=True):

    solver = get_solver(amici_model)
    solver.setSensitivityOrder(amici.SensitivityOrder_none)

    edata = amici.ExpData(amici_model.get())
    process_preequilibration(amici_model, edata, preequidict, preequilibrate)

    edatas = get_dose_response_edatas(amici_model, dose_name, doses, edata,
                                      preequilibrate, preequilibrate_dose_response)

    rdatas = amici.runAmiciSimulations(amici_model, solver, edatas,
                                       num_threads=8)

    return rdatas


def run_timecourse_simulation(amici_model, t, preequidict,
                              preequilibrate=True):
    solver = get_solver(amici_model)
    solver.setSensitivityOrder(amici.SensitivityOrder_none)

    amici_model.setTimepoints(t)
    edata = amici.ExpData(amici_model.get())
    process_preequilibration(amici_model, edata, preequidict, preequilibrate)

    rdata = amici.runAmiciSimulation(amici_model, solver, edata)
    if rdata['status'] < 0:
        print(rdata)
        raise Exception('failed simulation')
    return rdata


def process_preequilibration(amici_model, edata, preequidict, preequilibrate):
    fixed_parameters_preequilibration = np.zeros((amici_model.nk(),))
    if isinstance(preequidict, dict):
        for fixpar in preequidict:
            fixed_parameters_preequilibration[
                amici_model.getFixedParameterNames().index(fixpar)
            ] = preequidict[fixpar]

    if preequilibrate:
        edata.fixedParametersPreequilibration = \
            fixed_parameters_preequilibration
        edata.reinitializeFixedParameterInitialStates = True


def get_dose_response_edatas(amici_model, dose_name, doses, edata,
                             preequilibrate, preequilibrate_dose_response):
    edatas = []

    dose_idx = amici_model.getFixedParameterNames().index(dose_name)
    for dose in doses:
        data = amici.ExpData(edata)
        fp = list(data.fixedParameters)
        fp[dose_idx] = dose
        data.fixedParameters = tuple(fp)
        if preequilibrate and preequilibrate_dose_response:
            fp = list(data.fixedParametersPreequilibration)
            fp[dose_idx] = dose
            data.fixedParametersPreequilibration = tuple(fp)
        edatas.append(data)

    return edatas


def process_rdata_doseresponse(rdatas, amici_model, pysb_model, fun, doses,
                               mode):
    check_mode(mode)

    p = get_pysb_parameters(amici_model, pysb_model)

    funvals = []
    for id, rdata in enumerate(rdatas):

        x = rdata['x'][0, :].transpose()
        val = fun(x, p)

        value_dict = get_value_dict(val, mode)

        funvals.append(value_dict)

    doseresponse = pd.DataFrame.from_dict(
        {key: [funval[key] for funval in funvals] for key in funvals[0]},
        orient='columns',
    )
    doseresponse.index = [doses[idose] for idose in
                          doseresponse.index]
    return doseresponse


def process_rdata_timecourse(rdata, amici_model, pysb_model, fun, mode):
    p = get_pysb_parameters(amici_model, pysb_model)

    funvals = []
    for it, x in enumerate(rdata['x']):
        val = fun(x.transpose(), p)

        value_dict = get_value_dict(val, mode)

        funvals.append(value_dict)

    timecourse = pd.DataFrame.from_dict(
        {key: [funval[key] for funval in funvals] for key in funvals[0]},
        orient='columns',
    )

    timecourse.index = [rdata['t'][int(it)] for it in timecourse.index]
    return timecourse


def get_pysb_parameters(amici_model, pysb_model):
    for value, name in zip(amici_model.getUnscaledParameters(),
                           amici_model.getParameterNames()):
        pysb_model.parameters[name].value = value
    for value, name in zip(amici_model.getFixedParameters(),
                           amici_model.getFixedParameterNames()):
        pysb_model.parameters[name].value = value

    return [par.value for par in pysb_model.parameters]


def get_value_dict(val, mode):
    check_mode(mode)

    if mode == 'monomer':
        return {
            name: value if value > ATOL_MONO_FRACTIONS else 0
            for name, value in val.items()
        }
    else:
        return {
            name: rate if np.abs(rate) > ATOL_FLUXES else 0
            for name, rate in val.items()
        }


def get_figure_with_subplots(nrows, ncols, resizey=2/3, surface=False):
    subplot_kw = {}
    if surface:
        subplot_kw['projection'] = '3d'

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols, sharex=False, sharey=False,
        figsize=(MPL_FIGWIDTH,
                 MPL_FIGWIDTH * nrows / ncols * resizey),
        dpi=300,
        subplot_kw=subplot_kw
    )
    if nrows > 1:
        axes_list = [item for sublist in axes for item in sublist]
    else:
        axes_list = [ax for ax in axes]

    return fig, axes_list


def uM_to_molecules(c, V=1e-12, N_A=N_A):
    # divide C by 1e6 to get M
    N = c / 1e6 * N_A * V

    return N


def molecules_to_uM(N, V=1e-12, N_A=N_A):
    # multiply by 1e6 to get uM
    c = N / (N_A * V) * 1e6

    return c


def ng_per_mL_to_uM_EGF(c, m_Da, N_A=N_A):
    c_g_per_L = c * 1e9 * 1e-3
    moles_per_g = 1 / (m_Da * units.u.cgs.scale * N_A)

    # multiplication with 1e-6 to get from per L to per uL
    c_uM = c_g_per_L * moles_per_g * 1e-6

    return c_uM


def plot_and_save_fig(figdir, filename):
    plt.tight_layout()
    if figdir is None:
        figdir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'figures'
        )
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    if filename is not None:
        plt.savefig(os.path.join(figdir, filename))


def get_rule_modifies_site_state(rule, monomer_name, site_name):
    reactant_states = Counter([
        mp.site_conditions[site_name]
        for cp in rule.reactant_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
        if mp.monomer.name == monomer_name and site_name in mp.site_conditions
    ])
    product_states = Counter([
        mp.site_conditions[site_name]
        for cp in rule.product_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
        if mp.monomer.name == monomer_name and site_name in mp.site_conditions
    ])
    return reactant_states != product_states


def get_rule_removes_site(rule, monomer_name, site_name):
    reactant_states = Counter([
        mp.site_conditions[site_name]
        for cp in rule.reactant_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
        if mp.monomer.name == monomer_name and site_name in mp.site_conditions
    ])
    product_states = Counter([
        mp.site_conditions[site_name]
        for cp in rule.product_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
        if mp.monomer.name == monomer_name and site_name in mp.site_conditions
    ])
    for base_state in ['u', 'gdp', None]:
        if base_state in product_states and base_state not in reactant_states:
            return True
        elif base_state in product_states and base_state in reactant_states:
            return reactant_states[base_state] > product_states[base_state]
    return False


def process_rules(model, ignore_rules=None):
    if not model.reactions:
        pysb.bng.generate_equations(model)
    rules = dict()
    if ignore_rules is None:
        ignore_rules = []

    for rule in model.rules:
        if rule.name in ignore_rules:
            continue
        rdict = construct_rule_dict(rule, model, reverse=False)
        rules[rdict['name']] = rdict
        if rule.is_reversible:
            rdict = construct_rule_dict(rule, model, reverse=True)
            rules[rdict['name']] = rdict

    return rules


def get_rule_patterns(model):
    fwd_patterns = [
        (pattern, rule.name, f'E{ip}')
        for rule in model.rules
        for ip, pattern in enumerate(rule.reactant_pattern.complex_patterns)
        if pattern is not None
    ]
    bwd_patterns = [
        (pattern, f'{rule.name}__reverse', f'E{ip}')
        for rule in model.rules
        if rule.is_reversible
        for ip, pattern in enumerate(rule.product_pattern.complex_patterns)
        if pattern is not None
    ]

    spm = SpeciesPatternMatcher(model)

    return {
        f'{rule_name} {ep_id}': {
            'pattern': pattern,
            'species': spm.match(pattern, index=True, counts=True)
        }
        for pattern, rule_name, ep_id in fwd_patterns + bwd_patterns
    }


def construct_rule_dict(rule, model, reverse=False):
    rule_dict = dict()

    reactions_w_ids = [
        (ir, reaction)
        for ir, reaction in enumerate(model.reactions)
        if rule.name in reaction['rule']
        and reaction['reverse'][reaction['rule'].index(rule.name)] == reverse
    ]

    reactions = [r[1] for r in reactions_w_ids]

    rule_dict['reaction_idx'] = [r[0] for r in reactions_w_ids]

    if reverse:
        educts = [
            product
            for reaction in reactions
            for product in reaction['products']
        ]
        products = [
            product
            for reaction in reactions
            for product in reaction['reactants']
        ]
    else:
        educts = [
            product
            for reaction in reactions
            for product in reaction['reactants']
        ]
        products = [
            product
            for reaction in reactions
            for product in reaction['products']
        ]

    rule_dict['educt_idx'] = educts
    rule_dict['educt_species'] = [
        model.species[idx]
        for idx in educts
    ]
    rule_dict['product_idx'] = products
    rule_dict['product_species'] = [
        model.species[idx]
        for idx in products
    ]

    if reverse:
        output_pattern = rule.reactant_pattern.complex_patterns
        input_pattern = rule.product_pattern.complex_patterns
    else:
        input_pattern = rule.reactant_pattern.complex_patterns
        output_pattern = rule.product_pattern.complex_patterns

    rule_dict['raw_input'] = deepcopy_rp(
        input_pattern
    )

    rule_dict['raw_output'] = deepcopy_rp(
        output_pattern
    )

    if reverse:
        name_suffix = '__reverse'
    else:
        name_suffix = ''
    rule_dict['name'] = rule.name + name_suffix

    return rule_dict


def deepcopy_rp(rp):
    return [deepcopy_cp(cp) for cp in rp]


def deepcopy_cp(cp):
    if not hasattr(cp, 'monomer_patterns'):
        return copy.deepcopy(cp)
    return pysb.ComplexPattern(
        [deepcopy_mp(mp) for mp in cp.monomer_patterns],
        cp.compartment,
        cp.match_once,
    )


def deepcopy_mp(mp):
    return pysb.MonomerPattern(
        mp.monomer,
        copy.deepcopy(mp.site_conditions),
        mp.compartment
    )
