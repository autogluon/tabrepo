from typing import Tuple

from ..simulation.dense_utils import force_to_dense
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.tabular_predictions import TabularModelPredictions, TabularPicklePredictions, TabularPicklePredictionsOpt


def intersect_folds_and_datasets(zsc: ZeroshotSimulatorContext,
                                 zeroshot_pred_proba: TabularModelPredictions,
                                 zeroshot_gt: dict):
    zpp_datasets = zeroshot_pred_proba.datasets
    zsc_datasets = zsc.unique_datasets
    zsc_datasets_set = set(zsc_datasets)
    valid_datasets = [d for d in zpp_datasets if d in zsc_datasets_set]
    if set(zpp_datasets) != set(valid_datasets):
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        zpp_datasets = zeroshot_pred_proba.datasets
        zs_gt_keys = list(zeroshot_gt.keys())
        for d in zs_gt_keys:
            if d not in zpp_datasets:
                zeroshot_gt.pop(d)

    zpp_folds = set(zeroshot_pred_proba.folds)
    if zpp_folds != set(zsc.folds):
        zeroshot_pred_proba.restrict_folds(folds=zsc.folds)
        zs_gt_keys = list(zeroshot_gt.keys())
        for d in zs_gt_keys:
            for f in zpp_folds:
                if f not in zsc.folds:
                    zeroshot_gt[d].pop(f)
    datasets_in_zs = list(zeroshot_pred_proba.datasets)
    zsc.subset_datasets(datasets_in_zs)


def load_zeroshot_input(path_pred_proba: str,
                        path_gt: str,
                        zsc: ZeroshotSimulatorContext,
                        lazy_format: bool = False) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
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
    force_to_dense(zeroshot_pred_proba, first_prune_method='task', second_prune_method='dataset')

    zsc.subset_models(zeroshot_pred_proba.models)
    zsc.subset_datasets(zeroshot_pred_proba.datasets)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    # Convert to optimized format
    if isinstance(zeroshot_pred_proba, TabularPicklePredictions) and \
            not isinstance(zeroshot_pred_proba, TabularPicklePredictionsOpt):
        zeroshot_pred_proba = TabularPicklePredictionsOpt.from_dict(pred_dict=zeroshot_pred_proba.pred_dict)

    return zeroshot_pred_proba, zeroshot_gt, zsc


def prune_zeroshot_gt(zeroshot_pred_proba, zeroshot_gt, verbose: bool = True):
    num_datasets_start = len(zeroshot_gt)
    datasets = set(zeroshot_pred_proba.datasets)
    datasets_gt = list(zeroshot_gt.keys())
    for d in datasets_gt:
        if d not in datasets:
            zeroshot_gt.pop(d)
    num_datasets_end = len(zeroshot_gt)
    if verbose:
        print(f'Aligning GT with pred_proba... (Dataset count {num_datasets_start} -> {num_datasets_end})')
    assert len(datasets) == num_datasets_end
    return zeroshot_gt
