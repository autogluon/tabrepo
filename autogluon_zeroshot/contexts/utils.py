from . import intersect_folds_and_datasets
from ..simulation.tabular_predictions import TabularPicklePredictions, TabularPicklePredictionsOpt


def load_zeroshot_input(path_pred_proba, path_gt, zsc, lazy_format: bool = False):
    print(f'Loading ZS inputs:\n'
          f'\tpred_proba:  {path_pred_proba}\n'
          f'\tgt:          {path_gt}\n'
          f'\tlazy_format: {lazy_format}')
    zeroshot_gt = zsc.load_groundtruth(path_gt=path_gt)
    zeroshot_pred_proba = zsc.load_pred(
        pred_pkl_path=path_pred_proba,
        lazy_format=lazy_format,
    )

    # keep only dataset whose folds are all present
    intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt)
    zeroshot_pred_proba.force_to_dense(first_prune_method='task', second_prune_method='dataset')

    zsc.subset_models(zeroshot_pred_proba.models)
    zsc.subset_datasets(zeroshot_pred_proba.datasets)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    # Convert to optimized format
    if isinstance(zeroshot_pred_proba, TabularPicklePredictions) and \
            not isinstance(zeroshot_pred_proba, TabularPicklePredictionsOpt):
        zeroshot_pred_proba = TabularPicklePredictionsOpt.from_dict(pred_dict=zeroshot_pred_proba.pred_dict)

    return zeroshot_pred_proba, zeroshot_gt, zsc


def prune_zeroshot_gt(zeroshot_pred_proba, zeroshot_gt):
    num_datasets_start = len(zeroshot_gt)
    datasets = set(zeroshot_pred_proba.datasets)
    datasets_gt = list(zeroshot_gt.keys())
    for d in datasets_gt:
        if d not in datasets:
            zeroshot_gt.pop(d)
    num_datasets_end = len(zeroshot_gt)
    print(f'Aligning GT with pred_proba... (Dataset count {num_datasets_start} -> {num_datasets_end})')
    assert len(datasets) == num_datasets_end
    return zeroshot_gt
