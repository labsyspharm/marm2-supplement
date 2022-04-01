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
    'RAFi': 'initRAFi',
    'MEKi': 'initMEKi',
    'PRAFi': 'initPRAFi',
}

model = MARM.model.get_model_instance(model_name, variant, instance, INSTANCES)
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
            'mbraf': [
                'BRAFV600E_phosphorylates_MEK_bound1',
                'BRAFV600E_phosphorylates_MEK_bound2',
                'BRAFV600E_phosphorylates_MEK_bound3',
                'BRAFV600E_phosphorylates_MEK_bound4',
                'BRAFV600E_phosphorylates_MEK_unbound1',
                'BRAFV600E_phosphorylates_MEK_unbound2',
                'BRAFV600E_phosphorylates_MEK_unbound3',
                'BRAFV600E_phosphorylates_MEK_unbound4',
            ],
            'dbraf': [
                'BRAF_BRAF_phosphorylates_MEK',
                'BRAF_CRAF_phosphorylates_MEK',
            ],
            'craf': [
                'CRAF_BRAF_phosphorylates_MEK',
                'CRAF_CRAF_phosphorylates_MEK'
            ]
        },
        'MEK_is_dephosphorylated'
    )
    MARM.model.propagate_monomer_label(
        model,
        'MEK',
        'ERK',
        'pERK',
        ['mbraf', 'dbraf', 'craf'],
        'pMEK_phosphorylates_ERK',
        'DUSP_dephosphorylates_ERK',
        ('phospho', 'p')
    )

    pysb.Expression(
        'activeEGFR_obs',
        pysb.Observable(
            'activeEGFR',
            model.monomers['EGFR'](Tyr='p')
        )
    )
    pysb.Expression(
        'BRAF_mono_obs',
        pysb.Observable(
            'monoBRAF',
            model.monomers['BRAF'](raf=None)
        )
    )

    pysb.Expression(
        'BRAF_dimer_obs',
        pysb.Observable(
            'dimerBRAF',
            model.monomers['BRAF'](raf=pysb.ANY)
        )
    )

    if 'channelcf' in modifications:
        model.monomers['DUSP'].sites = ['erk_mbraf', 'erk_dbraf', 'erk_craf']
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

        for rule_name in ['DUSP_binds_pERK', 'DUSP_dissociates_from_ERK']:
            mbraf_rule = model.rules[rule_name]
            mbraf_rule.rename(mbraf_rule.name.replace('ERK', 'ERK_mbraf'))

            dbraf_rule = copy.deepcopy(mbraf_rule)
            dbraf_rule.name = dbraf_rule.name.replace('_mbraf', '_dbraf')
            model.add_component(dbraf_rule)

            craf_rule = copy.deepcopy(mbraf_rule)
            craf_rule.name = craf_rule.name.replace('_mbraf', '_craf')
            model.add_component(craf_rule)

            for rule, channel in zip([mbraf_rule, dbraf_rule, craf_rule],
                                     ['mbraf', 'dbraf', 'craf']):
                make_binding_channel_specific(rule, 'DUSP', 'erk', channel)

        for rule_name, channel in zip(
                ['DUSP_dephosphorylates_ERK_mbraf',
                 'DUSP_dephosphorylates_ERK_dbraf',
                 'DUSP_dephosphorylates_ERK_craf'],
                ['mbraf', 'dbraf', 'craf']
        ):
            make_binding_channel_specific(
                model.rules[rule_name], 'DUSP', 'erk', channel
            )

        rule = model.rules['synthesis_pDUSP']
        for cp in rule.product_pattern.complex_patterns:
            for mp in cp.monomer_patterns:
                if mp.monomer.name == 'DUSP':
                    condition = mp.site_conditions.pop('erk')
                    for channel in ['mbraf', 'dbraf', 'craf']:
                        mp.site_conditions[f'erk_{channel}'] = condition

if 'monoobs' in modifications:
    MARM.model.add_monomer_configuration_observables(model)

MARM.model.cleanup_unused(model)
MARM.model.export_model(model, ['pysb_flat', 'bngl'])
MARM.model.generate_equations(model)
MARM.model.write_observable_function(model)
MARM.model.compile_model(model)
