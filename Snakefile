import os
from MARM.paths import (
	get_model_module_file_instance, get_results_path,
	get_model_variant_file, get_multimodel_speedup_result_file,
	get_analysis_results_file, get_parameters_file, get_figure_dir
)
from MARM.estimation import RAFI, PANRAFI, MEKI
from MARM.read_data import (
	DATAFILES_ESTIMATION, DATAFILES_PREDICTION_COMBO, DATAFILES_PANRAFCOMBO,
	DATAFILES_PREDICTION_SINGLE, DATAFILES_MUTRAS_ENGINEERED,
	DATAFILES_MUTRAS_ENGINEERED_COMBO
)
DATAFILES = list(set(
	DATAFILES_ESTIMATION + DATAFILES_PREDICTION_COMBO + DATAFILES_PANRAFCOMBO
	+ DATAFILES_PREDICTION_SINGLE + DATAFILES_MUTRAS_ENGINEERED +
	DATAFILES_MUTRAS_ENGINEERED_COMBO
))
DATAFILES_CSV = [d for d in DATAFILES if not d.endswith('.xls')]
DATAFILES_XLS = [d for d in DATAFILES if d.endswith('.xls')]
import itertools

MODEL = 'RTKERK'

VARIANTS = ['base']
DATASETS = ['EGF_EGFR_MEKi_RAFi']
INSTANCES = 'EGF_MEKi_RAFi'

CONDITIONS = list(itertools.product(VARIANTS, DATASETS))

CONDITION_VARIANTS = [condition[0] for condition in CONDITIONS]
CONDITION_DATASETS = [condition[1] for condition in CONDITIONS]

COMMON_PERTURBATIONS = ['SPRY', 'DUSP', 'EGFR', 'SOS']
RAFI_PERTURBATIONS = ['RAF1', 'RAF2']
MEKI_PERTURBATIONS = ['MEKe', 'MEKr']

def get_instances(wildcards, modifications=None):
	options = f'{wildcards.dataset}'.split('_')
	if 'EGFR' in options:
		options.remove('EGFR')
	instances = []
	for r in range(len(options)+1):
		instances.extend(list(itertools.combinations(options, r)))
	instances = [
		"_".join(sorted(instance)) for instance in instances
	]
	return [
		get_model_module_file_instance(wildcards.model,
									   wildcards.variant,
									   instance, modifications)
		for instance in instances
	]

def get_instances_mutras(wildcards, modifications=None):
	options = f'{wildcards.dataset}'.split('_')
	if 'EGFR' in options:
		options.remove('EGFR')
	instances = []
	for r in range(len(options)+1):
		instances.extend(list(itertools.combinations(options, r)))
	instances = [
		"_".join(sorted(instance)) for instance in instances
	]
	return [
		get_model_module_file_instance(wildcards.model,
									   'nrasq61mut',
									   instance, modifications)
		for instance in instances
	]

JOBS = [str(i) for i in range(int(os.environ['N_JOBS']))]
AJOBS = [f'{i:03}' for i in range(int(os.environ['N_JOBS']))]
MS_PER_JOB = os.environ.get('MS_PER_JOB', 0)
N_THREADS = int(os.environ.get('N_THREADS', 0))

module_files = glob_wildcards(os.path.join('MARM', 'models', 'modules',
										   '{file}.py'))[0]

localrules: multistart_estimation, multimodel_benchmark,
		  process_data, collect_results_benchmark,
		  collect_results_estimation, clean, generate_figures

rule multistart_estimation:
	input:
		expand(
			os.path.join(
				get_results_path(MODEL, '{variant}'),
				'{dataset}.pickle'
			),
			zip, variant=CONDITION_VARIANTS, dataset=CONDITION_DATASETS
		)

rule multimodel_benchmark:
	input:
		os.path.join(
			get_results_path('RTKERK', 'base'),
			'EGF_EGFR_MEKi_RAFi_multimodel_benchmark.csv'
		)


rule clean:
	shell:
		'''
		rm -rf MARM/build
		rm -f  MARM/data/processed_*.csv ||:
		rm -rf tmp
		rm -rf logs/cluster/*.err
		rm -rf logs/cluster/*.out
		'''


rule build_instance:
	input:
		script='build_model_instance.py',
		model_file=get_model_variant_file('{model}','{variant}'),
	output:
		get_model_module_file_instance('{model}', '{variant}', '{instance}',
									   '{modifications}'),
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		instance='[\w]*',
		modifications='[\w]*'
	shell:
		'python3 {input.script} {wildcards.model} {wildcards.variant}' \
		' {wildcards.instance} {wildcards.modifications}'

rule process_data:
	input:
		csv=expand(os.path.join('MARM', 'data', '{d}.csv'),
				   d=DATAFILES_CSV),
		xls=expand(os.path.join('MARM', 'data', '{d}'),
				   d=DATAFILES_XLS),
		script='MARM/read_data.py',
	output:
		os.path.join('MARM', 'data', 'processed_{dataset}.csv'),
	wildcard_constraints:
		dataset='[\w]+',
	shell:
		'python3 {input.script} {wildcards.dataset}'


rule compute_multimodel_speedup:
	input:
		lambda wildcards: get_instances(wildcards),
		data=rules.process_data.output,
		script='multimodel_speedup.py',
	output:
		get_multimodel_speedup_result_file('{model}', '{variant}',
										   '{dataset}', '{repeat}')
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		repeat='\d+'
	shell:
		'python3 {input.script} {wildcards.model} {wildcards.variant}' \
		' {wildcards.dataset} {wildcards.repeat}'


rule parameter_estimation:
	input:
		lambda wildcards: get_instances(wildcards),
		data=rules.process_data.output,
		script='parameter_estimation.py',
	output:
		os.path.join(
			get_results_path('{model}','{variant}'),
			'{dataset}-{repeat}.pickle'
		),
	threads:
		N_THREADS
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		repeat='\d+',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} ' \
		f'{{wildcards.dataset}} {{wildcards.repeat}} {MS_PER_JOB} {{threads}}'


rule compute_simulation_prediction:
	input:
		model=lambda wildcards: get_instances(wildcards, 'channel_monoobs'),
		data_training=rules.process_data.output,
		data_combo=os.path.join('MARM', 'data',
								'processed_EGF_EGFR_MEKi_RAFi_comboprediction.csv'),
		data_single=os.path.join('MARM', 'data',
								'processed_EGF_EGFR_MEKi_RAFi_singleprediction.csv'),
		data_panRAF=os.path.join('MARM', 'data',
								 'processed_EGF_EGFR_MEKi_RAFi_panfrafcomboprediction.csv'),
		script='compute_{rfile}.py',
		parameters=get_parameters_file('{model}', '{variant}', '{dataset}'),
	output:
		get_analysis_results_file('{model}', '{variant}', '{dataset}',
								  '{rfile}', '{repeat}')
	threads:
		N_THREADS
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		rfile='[\w]+',
		repeat='\d+',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} ' \
		f'{{wildcards.dataset}} {{wildcards.repeat}} {{threads}} ' \
		f'{{wildcards.rfile}}'


rule compute_feedbacks:
	input:
		model=lambda wildcards: get_instances(wildcards, 'channel_monoobs'),
		data=rules.process_data.output,
		script='compute_feedback.py',
		parameters=get_parameters_file('{model}', '{variant}', '{dataset}'),
	output:
		observed=get_analysis_results_file('{model}', '{variant}',
										   '{dataset}',
								           'feedback_analysis_observed',
										   '{repeat}'),
		log=get_analysis_results_file('{model}', '{variant}', '{dataset}',
								      'feedback_analysis_log', '{repeat}'),
		preeq=get_analysis_results_file('{model}', '{variant}', '{dataset}',
								        'feedback_analysis_preequilibration',
										'{repeat}'),
	threads:
		N_THREADS
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		repeat='\d+',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} ' \
		f'{{wildcards.dataset}} {{wildcards.repeat}} {{threads}}'


rule compute_mutRASprediction:
	input:
		model=lambda wildcards: get_instances(wildcards, 'channel_monoobs'),
		model_mutras=lambda wildcards: get_instances_mutras(wildcards),
		model_mutras_cm=lambda wildcards: get_instances_mutras(
			wildcards, 'channel_monoobs'
		),
		data_training=expand(rules.process_data.output,
						     dataset='MEKi_RAFi_{cell_line}_mutrastraining'),
		data_prediction=expand(rules.process_data.output,
						       dataset='MEKi_RAFi_{cell_line}_mutrasprediction'),
		data_comboprediction=expand(rules.process_data.output,
						       		dataset='MEKi_RAFi_engineered_mutrascomboprediction'),
		script='compute_mutRASprediction.py',
		parameters=get_parameters_file('{model}', '{variant}', '{dataset}'),
	output:
		sim=get_analysis_results_file('{model}', '{variant}', '{dataset}',
								      'mutRASprediction_{cell_line}',
									  '{repeat}'),
		par=get_analysis_results_file('{model}', '{variant}', '{dataset}',
								      'mutRASpars_{cell_line}', '{repeat}'),
	threads:
		N_THREADS
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		repeat='\d+',
		cell_line='[\w]+',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} ' \
		f'{{wildcards.dataset}} {{wildcards.repeat}} {{threads}} ' \
		f'{{wildcards.cell_line}}'


rule compute_transduction:
	input:
		model=lambda wildcards: get_instances(wildcards, 'channel_monoobs'),
		data=rules.process_data.output,
		script='compute_transduction.py',
		parameters=get_parameters_file('{model}', '{variant}', '{dataset}'),
	output:
		expand(get_analysis_results_file(
			'{{model}}', '{{variant}}', '{{dataset}}',
			'transduction__{{drug}}__{{perturbations}}',
			'{{repeat}}')
		)
	threads:
		N_THREADS
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		repeat='\d+',
		drug='[\d\w_]+',
		perturbations='[\w\_]*',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} ' \
		f'{{wildcards.dataset}} {{wildcards.repeat}} {{threads}} '
		f'{{wildcards.drug}} {{wildcards.perturbations}}'


ruleorder: compute_feedbacks > compute_mutRASprediction > compute_transduction > compute_simulation_prediction


rule generate_figure:
	input:
		script='plot_{rfile}.py'
	output:
		os.path.join(get_figure_dir('{model}','{variant}', '{dataset}'),
					 '{rfile}')
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
	shell:
		f'python3 {{input.script}} ' \
		f'{{wildcards.model}}  {{wildcards.variant}} {{wildcards.dataset}}'


rule generate_figure_transduction:
	input:
		script='plot_transduction.py'
	output:
		os.path.join(get_figure_dir('{model}','{variant}', '{dataset}'),
					 'transduction__{drug}__{perturbations}')
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
		drug='[\d\w_]+',
		perturbations='[\w_]*',
	shell:
		'python3 {input.script} ' \
		'{wildcards.model}  {wildcards.variant} {wildcards.dataset} ' \
	    '{wildcards.drug} {wildcards.perturbations}'

rule generate_figures:
	input:
		expand(rules.generate_figure.output,
			   model=[MODEL], variant=VARIANTS, dataset=DATASETS,
			   rfile=['comboprediction', 'finepulse', 'feedback',
					  'panrafcomboprediction', 'singleprediction',
					  'trainingdata', 'mutRASprediction', 'ht29']),
		expand(rules.generate_figure_transduction.output,
			   model=[MODEL], variant=VARIANTS, dataset=DATASETS,
			   drug=RAFI + MEKI + PANRAFI, perturbations=['']),


rule run_analysis:
	input:
		expand(rules.compute_feedbacks.output.log,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS),
		expand(rules.compute_feedbacks.output.observed,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS),
		expand(rules.compute_feedbacks.output.preeq,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS),
		expand(rules.compute_transduction.output,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS, drug=RAFI+PANRAFI+MEKI,
		       perturbations=['']),
		expand(rules.compute_mutRASprediction.output.par,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS, cell_line=['engineered']),
		expand(rules.compute_mutRASprediction.output.sim,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS, cell_line=['engineered']),
		expand(rules.compute_simulation_prediction.output,
			   repeat=AJOBS, model=[MODEL], variant=VARIANTS,
			   dataset=DATASETS,
			   rfile=['trainingdata', 'comboprediction', 'finepulse',
					  'panrafcomboprediction', 'singleprediction' , 'ht29']),


rule collect_results_estimation:
	input:
		expand(
			os.path.join(
				get_results_path('{{model}}','{{variant}}'),
				'{{dataset}}-{repeat}.pickle'
			),
			repeat=JOBS
		),
		script='collectResultsEstimation.py'
	output:
		os.path.join(
			get_results_path('{model}','{variant}'),
			'{dataset}.pickle'
		)
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
	shell:
		'python3 {input.script} {wildcards.model} {wildcards.variant} ' \
		'{wildcards.dataset}'

rule collect_results_benchmark:
	input:
		expand(
			get_multimodel_speedup_result_file(
				'{{model}}', '{{variant}}','{{dataset}}', '{repeat}'
			), repeat=JOBS
		),
		script='collectResultsBenchmark.py'
	output:
		os.path.join(
			get_results_path('{model}', '{variant}'),
			'{dataset}_multimodel_benchmark.csv'
		)
	wildcard_constraints:
		model='[A-Z]+',
		variant='[\w]+',
		dataset='[\w]+',
	shell:
		'python3 {input.script} {wildcards.model} {wildcards.variant} ' \
		'{wildcards.dataset}'

class Wildcards(dict):
    __getattr__, __setattr__ = dict.get, dict.__setitem__

rule build_models:
	input:
		expand(
			rules.build_instance.output,
			model=MODEL, variant=VARIANTS + ['nrasq61mut'],
			modifications=['', 'channel_monoobs'],
		 	instance=[
				'_'.join(sorted(parts))
				for r in range(len(INSTANCES.split('_'))+1)
		 		for parts in itertools.combinations(INSTANCES.split('_'), r)
			]
		)

rule process_all_data:
	input:
		expand(
			rules.process_data.output,
			dataset=[
				'_'.join([dataset, split])
				for dataset in DATASETS
				for split in ['singleprediction',
							  'engineered_mutrasprediction',
							  'engineered_mutrastraining',
							  'comboprediction',
							  'panfrafcomboprediction',
							  'ht29']
			] + DATASETS
		)
