import os


def get_directory():
    return os.path.dirname(__file__)


def get_results_path(model_name, variant):
    return os.path.join(
        get_directory(),
        'results',
        'multistart',
        model_name,
        variant,
    )


def get_traces_path(model_name, variant):
    return os.path.join(
        'tmp',
        'traces',
        model_name,
        variant,
    )


def get_profile_dir(model_name, variant):
    return os.path.join(
        get_directory(),
        'analysis',
        'profiling',
        model_name,
        variant,
    )


def get_figure_dir(model_name, variant, dataset):
    figdir = os.path.join(
        get_directory(),
        'figures',
        model_name,
        variant,
        dataset,
    )
    if not model_name.startswith('{'):
        os.makedirs(figdir, exist_ok=True)
    return figdir


def get_multimodel_speedup_result_file(model, variant, dataset, index):
    return os.path.join(get_profile_dir(model, variant),
                        f'{dataset}_multimodel_objective_{index}.csv')


def get_model_variant_file(name, variant):
    full_name = get_model_name_variant(name, variant)
    return os.path.join(
        get_directory(),
        'pysb_flat',
        f'{full_name}.py'
    )


def get_analysis_results_file(model, variant, dataset, basename, index):
    analysis_dir = os.path.join(get_directory(), 'analysis')
    file_dir = os.path.join(analysis_dir, model, variant, dataset)
    if isinstance(index, int):
        indexstr = f'{index:03}'
    elif isinstance(index, str):
        if index.startswith('{'):
            indexstr = index
        else:
            indexstr = f'{int(index):03}'
    else:
        raise ValueError('incompatible type for index')

    file = os.path.join(file_dir, f'{basename}_{indexstr}.csv')
    return file


def get_parameters_file(model, variant, dataset):
    par_dir = os.path.join(get_directory(), 'parameters')
    os.makedirs(par_dir, exist_ok=True)
    return os.path.join(par_dir, f'{model}_{variant}_{dataset}.csv')


def get_model_module_file_instance(name, variant, instance,
                                   modifications=None):
    full_name = get_model_instance_name(name, variant, instance, modifications)
    return os.path.join(
        get_model_module_dir_instance(name, variant, instance, modifications),
        f'{full_name}.py'
    )


def get_model_module_dir_instance(name, variant, instance, modifications=None):
    full_name = get_model_instance_name(name, variant, instance, modifications)
    return os.path.join(
        get_directory(),
        'build',
        full_name,
        full_name,
    )


def get_model_module_dir_dataset(name, variant, dataset, modifications=None):

    return get_model_module_dir_instance(
        name, variant, dataset_to_instance(dataset), modifications
    )


def get_model_name_variant(name, variant):
    return f'{name}__{variant}'


def get_model_instance_name(name, variant, instance, modifications):
    if modifications is None:
        modifications = ''
    else:
        modifications = '_'.join(sorted(modifications.split('_')))
    full_name = f'{name}__{variant}__{instance}__{modifications}'
    return full_name


def get_model_name_dataset(name, variant, dataset, modifications=None):
    return get_model_instance_name(name, variant, dataset_to_instance(dataset),
                                   modifications)


def dataset_to_instance(dataset):
    return '_'.join(
        pert
        for pert in dataset.split('_')
        if pert in ['EGF', 'NRAS', 'RAFi', 'PRAFi', 'MEKi']
    )
