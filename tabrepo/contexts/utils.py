from typing import List, Tuple

from ..simulation.dense_utils import force_to_dense
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.tabular_predictions import TabularModelPredictions


def intersect_folds_and_datasets(zsc: ZeroshotSimulatorContext,
                                 zeroshot_pred_proba: TabularModelPredictions,
                                 zeroshot_gt: GroundTruth):
    zpp_tids = [zsc.dataset_to_tid_dict[dataset] for dataset in zeroshot_pred_proba.datasets if dataset in zsc.dataset_to_tid_dict]
    zsc_tids = zsc.unique_datasets
    zsc_tids_set = set(zsc_tids)
    valid_tids = [tid for tid in zpp_tids if tid in zsc_tids_set]
    if set(zpp_tids) != set(valid_tids):
        zeroshot_pred_proba.restrict_datasets(datasets=[zsc.tid_to_dataset_dict[tid] for tid in valid_tids])
        zpp_tids = [zsc.dataset_to_tid_dict[dataset] for dataset in zeroshot_pred_proba.datasets]
        for d in zeroshot_gt.tids:
            if d not in zpp_tids:
                zeroshot_gt.remove_tid(d)
    # TODO The dense conversion logic happens at several places, ideally it should happen only once
    # zpp_folds = set(zeroshot_pred_proba.folds)
    # if zpp_folds != set(zsc.folds):
    #     zeroshot_pred_proba.restrict_folds(folds=zsc.folds)
    #     for d in zeroshot_gt.tids:
    #         for f in zpp_folds:
    #             if f not in zsc.folds:
    #                 zeroshot_gt[d].pop(f, None)
    zpp_tids = [zsc.dataset_to_tid_dict[dataset] for dataset in zeroshot_pred_proba.datasets if dataset in zsc.dataset_to_tid_dict]
    zsc.subset_datasets(zpp_tids)


def load_zeroshot_input(path_pred_proba: str,
                        paths_gt: List[str],
                        datasets: List[str],
                        zsc: ZeroshotSimulatorContext,
                        ) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
    print(f'Loading ZS inputs:\n'
          f'\tpred_proba:  {path_pred_proba}\n'
    )
    zeroshot_gt = zsc.load_groundtruth(paths_gt=paths_gt)
    zeroshot_pred_proba = zsc.load_pred(
        path_pred_proba=path_pred_proba,
        datasets=datasets,
    )

    # keep only dataset whose folds are all present
    intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt)
    force_to_dense(zeroshot_pred_proba, first_prune_method='task', second_prune_method='dataset')

    zsc.subset_models(zeroshot_pred_proba.models)
    zsc.subset_datasets([zsc.dataset_to_tid_dict[dataset] for dataset in zeroshot_pred_proba.datasets])
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(dataset_to_tid_dict=zsc.dataset_to_tid_dict, zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    return zeroshot_pred_proba, zeroshot_gt, zsc


def prune_zeroshot_gt(dataset_to_tid_dict, zeroshot_pred_proba, zeroshot_gt: GroundTruth, verbose: bool = True) -> GroundTruth:

    num_datasets_start = len(zeroshot_gt.tids)
    tid_pred = set(dataset_to_tid_dict[dataset] for dataset in zeroshot_pred_proba.datasets if dataset in dataset_to_tid_dict)
    for tid in list(zeroshot_gt.tids):
        if tid not in tid_pred:
            zeroshot_gt.remove_tid(tid)
    num_datasets_end = len(zeroshot_gt.tids)
    if verbose:
        print(f'Aligning GT with pred_proba... (Dataset count {num_datasets_start} -> {num_datasets_end})')
    assert len(tid_pred) == num_datasets_end
    return zeroshot_gt
