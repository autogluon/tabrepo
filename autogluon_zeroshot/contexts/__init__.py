from ..simulation.tabular_predictions import TabularModelPredictions
from ..simulation.simulation_context import ZeroshotSimulatorContext


def intersect_folds_and_datasets(zsc: ZeroshotSimulatorContext,
                                 zeroshot_pred_proba: TabularModelPredictions,
                                 zeroshot_gt):
    zpp_datasets = zeroshot_pred_proba.datasets
    zsc_datasets = zsc.unique_datasets
    zsc_datasets_set = set(zsc_datasets)
    valid_datasets = [d for d in zpp_datasets if d in zsc_datasets_set]
    if set(zpp_datasets) != set(valid_datasets):
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        zpp_datasets = zeroshot_pred_proba.datasets
        zs_gt_keys = zeroshot_gt.keys()
        for d in zs_gt_keys:
            if d not in zpp_datasets:
                zeroshot_gt.pop(d)

    zpp_folds = set(zeroshot_pred_proba.folds)
    if zpp_folds != set(zsc.folds):
        zeroshot_pred_proba.restrict_folds(folds=zsc.folds)
        zs_gt_keys = zeroshot_gt.keys()
        for d in zs_gt_keys:
            for f in zpp_folds:
                if f not in zsc.folds:
                    zeroshot_gt[d].pop(f)
    datasets_in_zs = list(zeroshot_pred_proba.datasets)
    zsc.subset_datasets(datasets_in_zs)