from typing import Tuple
from autogluon.common.loaders import load_pd

from ..loaders import load_configs, load_results, combine_results_with_score_val, Paths
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.tabular_predictions import TabularModelPredictions


def load_context_2022_12_11_bag(folds=None, load_zeroshot_pred_proba=False, lazy_format=False) -> Tuple[ZeroshotSimulatorContext, dict, TabularModelPredictions, dict]:
    """
    :param folds:
    :param load_zeroshot_pred_proba:
    :param lazy_format: whether to load with a format where all data is in memory (`TabularPicklePredictions`) or a
    format where data is loaded on the fly (`TabularPicklePerTaskPredictions`). Both formats have the same interface.
    :return:
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    subset = 'small_30'

    df_results, df_results_by_dataset, df_raw, df_metadata = load_results(
        results=str(Paths.bagged_results_root / subset / "results_ranked_valid.csv"),
        results_by_dataset=str(Paths.bagged_results_root / subset / "results_ranked_by_dataset_valid.parquet"),
        raw=str(Paths.bagged_results_root / "openml_ag_2022_12_11_zs_models.parquet"),
        metadata=str(Paths.data_root / "metadata" / "task_metadata.csv"),
    )
    df_results_by_dataset = combine_results_with_score_val(df_raw, df_results_by_dataset)

    # Load in real framework results to score against
    path_prefix_automl = Paths.results_root / 'automl'
    df_results_by_dataset_automl = load_pd.load(f'{path_prefix_automl}/results_ranked_by_dataset_valid.csv')

    zsc = ZeroshotSimulatorContext(
        df_results_by_dataset=df_results_by_dataset,
        df_results_by_dataset_automl=df_results_by_dataset_automl,
        df_raw=df_raw,
        folds=folds,
    )

    configs_prefix_1 = str(Paths.data_root / 'configs/configs_20221004')
    configs_prefix_2 = str(Paths.data_root / 'configs')
    config_files_to_load = [
        f'{configs_prefix_1}/configs_catboost.json',
        f'{configs_prefix_1}/configs_fastai.json',
        f'{configs_prefix_1}/configs_lightgbm.json',
        f'{configs_prefix_1}/configs_nn_torch.json',
        f'{configs_prefix_1}/configs_xgboost.json',
        f'{configs_prefix_2}/configs_rf.json',
        f'{configs_prefix_2}/configs_xt.json',
        f'{configs_prefix_2}/configs_knn.json',
    ]
    configs_full = load_configs(config_files_to_load)

    zeroshot_pred_proba = None
    zeroshot_gt = None
    if load_zeroshot_pred_proba:
        path_zs_gt = str(Paths.bagged_results_root / 'zeroshot_gt_2022_12_11_zs.pkl')
        zeroshot_gt = zsc.load_groundtruth(path_gt=path_zs_gt)
        if lazy_format:
            path_zs_pred_proba = str(Paths.bagged_results_root / 'zeroshot_pred_per_task')
        else:
            path_zs_pred_proba = str(Paths.bagged_results_root / 'zeroshot_pred_proba_2022_12_11_zs.pkl')
        zeroshot_pred_proba = zsc.load_pred(path_pred_proba=path_zs_pred_proba, lazy_format=lazy_format)

        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt)

    return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt


def intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt):
    dataset_names = zeroshot_pred_proba.datasets
    dataset_names_set = set(dataset_names)
    # for d in zsc.unique_datasets:
    #     if d not in dataset_names_set:
    #         raise AssertionError(f'Missing expected dataset {d} in zeroshot_pred_proba!')
    #     folds_in_zs = list(zeroshot_pred_proba[d].keys())
    #     for f in zsc.folds:
    #         if f not in folds_in_zs:
    #             raise AssertionError(f'Missing expected fold {f} in dataset {d} in zeroshot_pred_proba! '
    #                                  f'Expected: {zsc.folds}, Actual: {folds_in_zs}')

    for d in dataset_names:
        if d not in zsc.unique_datasets:
            zeroshot_pred_proba.remove_dataset(d)
            if d in zeroshot_gt:
                zeroshot_gt.pop(d)
        else:
            # folds_in_zs = list(zeroshot_pred_proba[d].keys())
            for f in zeroshot_pred_proba.folds:
                if f not in zsc.folds:
                    zeroshot_pred_proba[d].pop(f)
                    zeroshot_gt[d].pop(f)


def get_configs_default():
    autogluon_configs = [
        'CatBoost_c1',
        'LightGBM_c1',
        'LightGBM_c2',
        'LightGBM_c3',
        'NeuralNetFastAI_c1',
        'RandomForest_c1',
        'ExtraTrees_c1',
    ]
    autogluon_configs = [a + '_BAG_L1' for a in autogluon_configs]
    return autogluon_configs


def get_configs_small(num_per_type=12):
    small_extra_configs = []
    for m in [
        'LightGBM',
        'CatBoost',
        'RandomForest',
        'ExtraTrees',
        'NeuralNetFastAI',
    ]:
        for i in range(1, num_per_type):
            small_extra_configs.append(m + f'_r{i}_BAG_L1')
    small_configs = get_configs_default() + small_extra_configs
    return small_configs
