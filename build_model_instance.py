import os
import sys
import pysb
import copy

import MARM
from MARM.paths import get_model_instance_name

model_name = sys.argv[1]
variant = sys.argv[2]

modifications = None
if len(sys.argv) > 3:
    instance = sys.argv[3]
    if any(mod in instance.split('_') for mod in ['channel', 'channelcf',
                                                  'monoobs']):
        modifications = instance
        instance = ''
    if len(sys.argv) > 4:
        modifications = sys.argv[4]
else:
    instance = ''

INSTANCES = {
    'EGF': 'initEGF',
    'RAFi': 'RAFi_0',
    'MEKi': 'MEKi_0',
    'PRAFi': 'PRAFi_0',
    'NRAS': 'NRAS_Q61mut',
}

model = MARM.model.get_model_instance(model_name, variant, instance, INSTANCES)
if 'NRAS' in instance.split('_'):
    RAS = model.monomers['RAS']
    NRAS_Q61mut = pysb.Parameter('NRAS_Q61mut', 0.0)
    q61_RAS_gtp_kcat = pysb.Parameter('q61_RAS_gtp_kcat', 0.01)
    NRAS_mut_activation = pysb.Expression('NRAS_mut_activation',
                                          NRAS_Q61mut * q61_RAS_gtp_kcat)
    pysb.Rule('mutated_RAS_guanosine_exchange',
              RAS(sos1=None, state='gdp') >> RAS(sos1=None, state='gtp'),
              NRAS_mut_activation)

model.name = get_model_instance_name(model_name, variant, instance,
                                     modifications)

if modifications is None:
    modifications = []
else:
    modifications = modifications.split('_')

if 'channel' in modifications or 'channelcf' in modifications:
    MARM.model.add_monomer_label(
        model,
        'MEK',
        'pMEK',
        ('phospho', 'p'),
        {
            'onco': [
                'BRAFV600E_phosphorylates_MEK_bound1',
                'BRAFV600E_phosphorylates_MEK_bound2',
                'BRAFV600E_phosphorylates_MEK_bound3',
                'BRAFV600E_phosphorylates_MEK_bound4',
                'BRAFV600E_phosphorylates_MEK_unbound1',
                'BRAFV600E_phosphorylates_MEK_unbound2',
                'BRAFV600E_phosphorylates_MEK_unbound3',
                'BRAFV600E_phosphorylates_MEK_unbound4',
            ],
            'phys': [
                'BRAF_BRAF_phosphorylates_MEK',
                'CRAF_BRAF_phosphorylates_MEK',
                'BRAF_CRAF_phosphorylates_MEK',
                'CRAF_CRAF_phosphorylates_MEK'
            ],
        },
        'MEK_is_dephosphorylated'
    )
    MARM.model.propagate_monomer_label(
        model,
        'MEK',
        'ERK',
        'pERK',
        ['onco', 'phys'],
        'pMEK_phosphorylates_ERK',
        'DUSP_dephosphorylates_ERK',
        ('phospho', 'p')
    )
    # additional observables
    pysb.Expression(
        'activeEGFR_obs',
        pysb.Observable(
            'activeEGFR',
            model.monomers['EGFR'](Tyr='p')
        )
    )
    pysb.Expression(
        'gtpRAS_obs',
        pysb.Observable(
            'gtpRAS',
            RAS(state='gtp')
        )
    )
    BRAF = model.monomers['BRAF']
    CRAF = model.monomers['CRAF']
    MEK = model.monomers['MEK']
    pysb.Observable(
        'baseline_R',
        BRAF(raf=None, rafi=None) +
        CRAF(raf=None, rafi=None)
    )
    pysb.Observable(
        'baseline_IR',
        BRAF(raf=None, rafi=pysb.ANY) +
        CRAF(raf=None, rafi=pysb.ANY)
    )
    pysb.Observable(
        'baseline_RR',
        BRAF(rafi=None) % BRAF(rafi=None) +
        CRAF(rafi=None) % BRAF(rafi=None) +
        CRAF(rafi=None) % CRAF(rafi=None)
    )
    pysb.Observable(
        'baseline_RRI',
        BRAF(rafi=pysb.ANY) % BRAF(rafi=None) +
        BRAF(rafi=pysb.ANY) % CRAF(rafi=None) +
        BRAF(rafi=None) % CRAF(rafi=pysb.pysb.ANY) +
        CRAF(rafi=pysb.ANY) % CRAF(rafi=None)
    )
    pysb.Observable(
        'baseline_IRRI',
        BRAF(rafi=pysb.ANY) % BRAF(rafi=pysb.ANY)
        + BRAF(rafi=pysb.ANY) % CRAF(rafi=pysb.ANY)
        + CRAF(rafi=pysb.ANY) % CRAF(rafi=pysb.ANY)
    )
    pysb.Observable('RAF_marginal_RAFi', BRAF(rafi=pysb.ANY) + CRAF(rafi=pysb.ANY))
    pysb.Observable('RAF_marginal_RAS', BRAF(RBD=pysb.ANY) + CRAF(RBD=pysb.ANY))
    pysb.Observable('RAF_marginal_RAF', BRAF(raf=pysb.ANY) + CRAF(raf=pysb.ANY))
    pysb.Observable('RAF_marginal_MEK', BRAF(mek=pysb.ANY) + CRAF(mek=pysb.ANY))
    pysb.Observable('drugfree_uMEK', MEK(meki=None, phospho='u'))
    pysb.Observable('inhibited_uMEK', MEK(meki=pysb.ANY, phospho='u'))
    pysb.Observable('drugfree_pMEK', MEK(meki=None, phospho='p'))
    pysb.Observable('inhibited_pMEK', MEK(meki=pysb.ANY, phospho='p'))


    if 'channelcf' in modifications:
        model.monomers['DUSP'].sites = ['erk_onco', 'erk_phys']
        MARM.model.update_monomer_patterns(model, model.monomers['DUSP'])


        def make_binding_channel_specific(rule, monomer, binding_site,
                                          channel):
            for pattern in [rule.rule_expression.reactant_pattern,
                            rule.rule_expression.product_pattern]:
                for cp in pattern.complex_patterns:
                    for mp in cp.monomer_patterns:
                        if mp.monomer.name == monomer \
                                and binding_site in mp.site_conditions:
                            mp.site_conditions[f'{binding_site}_{channel}'] = \
                                mp.site_conditions.pop(binding_site)
                        if mp.monomer.name == 'ERK' \
                                and 'phospho' in mp.site_conditions \
                                and mp.site_conditions['phospho'] == 'p':
                            mp.site_conditions['channel'] = channel

        rule_name = 'bind_DUSP_pERK'
        onco_rule = model.rules[rule_name]
        onco_rule.rename(onco_rule.name.replace('ERK', 'ERK_onco'))

        phys_rule = copy.deepcopy(onco_rule)
        phys_rule.name = phys_rule.name.replace('_onco', '_phys')
        model.add_component(phys_rule)

        for rule, channel in zip([onco_rule, phys_rule],
                                 ['onco', 'phys']):
            make_binding_channel_specific(rule, 'DUSP', 'erk', channel)

        for rule_name, channel in zip(
                ['DUSP_dephosphorylates_ERK_onco',
                 'DUSP_dephosphorylates_ERK_phys'],
                ['onco', 'phys']
        ):
            make_binding_channel_specific(
                model.rules[rule_name], 'DUSP', 'erk', channel
            )

        rule = model.rules['synthesis_pDUSP']
        for cp in rule.product_pattern.complex_patterns:
            for mp in cp.monomer_patterns:
                if mp.monomer.name == 'DUSP':
                    condition = mp.site_conditions.pop('erk')
                    mp.site_conditions['erk_onco'] = condition
                    mp.site_conditions['erk_phys'] = condition

if 'monoobs' in modifications:
    MARM.model.add_monomer_configuration_observables(model)

MARM.model.cleanup_unused(model)
MARM.model.export_model(model, ['pysb_flat', 'bngl'])
MARM.model.generate_equations(model)
MARM.model.write_observable_function(model)
#os.environ["AMICI_IMPORT_NPROCS"] = '4'
#os.environ["AMICI_PARALLEL_COMPILE"] = '4'
os.environ["AMICI_CXXFLAGS"] = '-O1'
MARM.model.compile_model(model)
