from typing import List

from autogluon_zeroshot.simulation.tabular_predictions import TabularModelPredictions


def is_dense_folds(tabular_predictions: TabularModelPredictions) -> bool:
    """
    Return True if all datasets have all folds
    """
    return set(tabular_predictions.folds) == set(get_folds_dense(tabular_predictions))


def is_dense(tabular_predictions: TabularModelPredictions) -> bool:
    """
    Return True if all datasets have all folds, and all tasks have all models
    """
    return is_dense_folds(tabular_predictions) and is_dense_models(tabular_predictions)


def is_dense_models(tabular_predictions: TabularModelPredictions) -> bool:
    """
    Return True if all tasks have all models
    """
    models_dense = get_models(tabular_predictions, present_in_all=True)
    models_sparse = get_models(tabular_predictions, present_in_all=False)
    return set(models_dense) == set(models_sparse)


def get_models_dense(tabular_predictions: TabularModelPredictions) -> List[str]:
    """
    Returns models that appears in all lists, eg that are available for all tasks and splits
    """
    return sorted(tabular_predictions.list_models_available(present_in_all=True))

def get_models(tabular_predictions: TabularModelPredictions, present_in_all=False) -> List[str]:
    """
    Gets all valid models
    :param present_in_all:
        If True, only returns models present in every dataset (dense)
        If False, returns every model that appears in at least 1 dataset (sparse)
    """
    if not present_in_all:
        return tabular_predictions.models
    else:
        return get_models_dense(tabular_predictions)


def force_to_dense(tabular_predictions: TabularModelPredictions,
                   first_prune_method: str = 'task',
                   second_prune_method: str = 'dataset',
                   assert_not_empty: bool = True,
                   verbose: bool = True):
    """
    Force to be dense in all dimensions.
    This means all models will be present in all tasks, and all folds will be present in all datasets.
    # TODO: Not guaranteed to be dense if first_prune_method = 'dataset'
    """
    if first_prune_method in ['dataset', 'fold']:
        first_method = force_to_dense_folds
        second_method = force_to_dense_models
    else:
        first_method = force_to_dense_models
        second_method = force_to_dense_folds
    if verbose:
        print(
            f'Forcing {tabular_predictions.__class__.__name__} to dense representation via two-stage filtering using '
            f'`first_prune_method="{first_prune_method}"`, `second_prune_method="{second_prune_method}"`...')
    first_method(tabular_predictions, prune_method=first_prune_method, assert_not_empty=assert_not_empty, verbose=verbose)
    second_method(tabular_predictions, prune_method=second_prune_method, assert_not_empty=assert_not_empty, verbose=verbose)

    if verbose:
        print(f'The {tabular_predictions.__class__.__name__} object is now guaranteed to be dense.')
    assert is_dense(tabular_predictions)


def get_datasets_with_folds(tabular_predictions: TabularModelPredictions, folds: List[int]) -> List[str]:
    """
    Get list of datasets that have results for all input folds
    """
    datasets = tabular_predictions.datasets
    valid_datasets = []
    for dataset in datasets:
        folds_in_dataset = tabular_predictions.list_folds_available(datasets=[dataset])
        if all(f in folds_in_dataset for f in folds):
            valid_datasets.append(dataset)
    return valid_datasets

def force_to_dense_folds(tabular_predictions: TabularModelPredictions,
                         prune_method: str = 'dataset',
                         assert_not_empty: bool = True,
                         verbose: bool = True):
    """
    Force the pred dict to contain only dense fold results (no missing folds for any dataset)
    :param prune_method:
        If 'dataset', prunes any dataset that doesn't contain results for all folds
        If 'fold', prunes any fold that doesn't exist for all datasets
    """
    if verbose:
        print(f'Forcing {tabular_predictions.__class__.__name__} to dense fold representation using `prune_method="{prune_method}"`...')
    valid_prune_methods = ['dataset', 'fold']
    if prune_method not in valid_prune_methods:
        raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
    pre_num_models = len(tabular_predictions.models)
    pre_num_datasets = len(tabular_predictions.datasets)
    pre_num_folds = len(tabular_predictions.folds)
    if prune_method == 'dataset':
        datasets_dense = get_datasets_with_folds(tabular_predictions, folds=tabular_predictions.folds)
        tabular_predictions.restrict_datasets(datasets=datasets_dense)
    elif prune_method == 'fold':
        folds_dense = get_folds_dense(tabular_predictions)
        tabular_predictions.restrict_folds(folds=folds_dense)
    else:
        raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
    post_num_models = len(tabular_predictions.models)
    post_num_datasets = len(tabular_predictions.datasets)
    post_num_folds = len(tabular_predictions.folds)

    if verbose:
        print(f'\tPre : datasets={pre_num_datasets} | models={pre_num_models} | folds={pre_num_folds}')
        print(f'\tPost: datasets={post_num_datasets} | models={post_num_models} | folds={post_num_folds}')
    assert is_dense_folds(tabular_predictions)
    if assert_not_empty:
        assert not tabular_predictions.is_empty()


def force_to_dense_models(tabular_predictions: TabularModelPredictions,
                          prune_method: str = 'task',
                          assert_not_empty: bool = True,
                          verbose: bool = True):
    """
    Force the pred dict to contain only dense results (no missing result for any task/model)
    :param prune_method:
        If 'task', prunes any task that doesn't contain results for all models
        If 'model', prunes any model that doesn't have results for all tasks
    """
    if verbose:
        print(f'Forcing {tabular_predictions.__class__.__name__} to dense model representation using `prune_method="{prune_method}"`...')
    valid_prune_methods = ['task', 'model']
    if prune_method not in valid_prune_methods:
        raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
    datasets = tabular_predictions.datasets
    valid_models = get_models(tabular_predictions, present_in_all=False)
    pre_num_models = len(valid_models)
    pre_num_datasets = len(datasets)
    pre_num_folds = len(tabular_predictions.folds)
    if prune_method == 'task':
        valid_tasks = []
        for task in tabular_predictions.tasks:
            dataset = task[0]
            fold = task[1]
            models_in_task = tabular_predictions.list_models_available(datasets=[dataset], folds=[fold], present_in_all=True)
            models_in_task_set = set(models_in_task)
            if all(m in models_in_task_set for m in valid_models):
                valid_tasks.append(task)
        tabular_predictions.restrict_tasks(tasks=valid_tasks)
    elif prune_method == 'model':
        valid_models = get_models(tabular_predictions, present_in_all=True)
        tabular_predictions.restrict_models(models=valid_models)
    else:
        raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
    post_num_models = len(tabular_predictions.models)
    post_num_datasets = len(tabular_predictions.datasets)
    post_num_folds = len(tabular_predictions.folds)

    if verbose:
        print(f'\tPre : datasets={pre_num_datasets} | models={pre_num_models} | folds={pre_num_folds}')
        print(f'\tPost: datasets={post_num_datasets} | models={post_num_models} | folds={post_num_folds}')
    assert is_dense_models(tabular_predictions)
    if assert_not_empty:
        assert not tabular_predictions.is_empty()


def get_folds_dense(tabular_predictions: TabularModelPredictions) -> List[int]:
    """
    Returns folds that appear in all datasets
    """
    return tabular_predictions.list_folds_available(present_in_all=True)

def print_summary(tabular_predictions: TabularModelPredictions):
    folds = tabular_predictions.folds
    datasets = tabular_predictions.datasets
    tasks = tabular_predictions.tasks
    models = tabular_predictions.models

    num_folds = len(folds)
    num_datasets = len(datasets)
    num_tasks = len(tasks)
    num_models = len(models)

    folds_dense = get_folds_dense(tabular_predictions)
    models_dense = get_models_dense(tabular_predictions)

    num_folds_dense = len(folds_dense)
    num_models_dense = len(models_dense)

    print(f'Summary of {tabular_predictions.__class__.__name__}:\n'
          f'\tdatasets={num_datasets}\t| folds={num_folds} (dense={num_folds_dense})\t| tasks={num_tasks}\t'
          f'| models={num_models} (dense={num_models_dense})\n'
          f'\tis_dense={is_dense(tabular_predictions)} | '
          f'is_dense_folds={is_dense_folds(tabular_predictions)} | '
          f'is_dense_models={is_dense_models(tabular_predictions)}')
