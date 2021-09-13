import sys
import pysb
import MARM.models
from MARM.paths import get_model_instance_name
from MRBM.modules import (
    add_monomer_configuration_observables, add_monomer_label,
    propagate_monomer_label
)

model_name = sys.argv[1]
variant = sys.argv[2]

modifications = None
if len(sys.argv) > 3:
    instance = sys.argv[3]
    if 'channel' in instance.split('_') or 'monoobs' in instance.split('_'):
        modifications = instance
        instance = ''
    if len(sys.argv) > 4:
        modifications = sys.argv[4]
else:
    instance = ''

INSTANCES = MARM.models.get_model_module(model_name).INSTANCES

model = MARM.model.get_model_instance(model_name, variant, instance, INSTANCES)
model.name = get_model_instance_name(model_name, variant, instance,
                                     modifications)

if modifications is None:
    modifications = []
else:
    modifications = modifications.split('_')

if 'channel' in modifications:
    add_monomer_label(
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
    propagate_monomer_label(
        model,
        'MEK',
        'ERK',
        'pERK',
        ['onco', 'phys'],
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

if 'monoobs' in modifications:
    add_monomer_configuration_observables(model)

MARM.model.cleanup_unused(model)
MARM.model.export_model(model, ['pysb_flat', 'bngl'])
MARM.model.generate_equations(model)
MARM.model.write_aux_model_functions(model)
MARM.model.compile_model(model)
