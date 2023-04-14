import time
from typing import Tuple
from autogluon.common.loaders import load_pd
from pathlib import Path

from .utils import load_zeroshot_input
from ..loaders import load_configs, load_results, combine_results_with_score_val, Paths
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.tabular_predictions import TabularModelPredictions


# FIXME: Generalize this logic to avoid code dupe
def load_context_2023_03_19_bag_289(
        folds=None,
        load_zeroshot_pred_proba=False,
        lazy_format=False,
        max_size_mb: int = 10,
) -> Tuple[ZeroshotSimulatorContext, dict, TabularModelPredictions, dict]:
    """
    :param folds:
    :param load_zeroshot_pred_proba:
    :param lazy_format: whether to load with a format where all data is in memory (`TabularPicklePredictions`) or a
    format where data is loaded on the fly (`TabularPicklePerTaskPredictions`). Both formats have the same interface.
    :return:
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    time_start = time.time()
    path_bagged_root = Paths.bagged_289_results_root
    path_bagged_root_s3 = 's3://automl-benchmark-ag/aggregated/ec2/2023_03_19_zs/'

    path_bagged_root_s3_zs_input = f'{path_bagged_root_s3}zs_input/bagged_289/'

    results_path = str(path_bagged_root / "608/results_ranked_valid.parquet")
    results_by_dataset_path = str(path_bagged_root / "608/results_ranked_by_dataset_valid.parquet")
    raw_path = str(path_bagged_root / "openml_ag_2023_03_19_zs_models.parquet")
    path_automl = str(Paths.automl_289_results_root / "results_ranked_by_dataset_valid.csv")
    local_files_exist = all(Path(f).exists() for f in [results_path, results_by_dataset_path, raw_path, path_automl])
    if not local_files_exist:
        # TODO option to automatically download files in this case
        print(f"Could not find local files, using s3 files from {path_bagged_root_s3_zs_input}.")
        results_path = f"{path_bagged_root_s3_zs_input}608/results_ranked_valid.parquet"
        results_by_dataset_path = f"{path_bagged_root_s3_zs_input}608/results_ranked_by_dataset_valid.parquet"
        raw_path = f"{path_bagged_root_s3_zs_input}openml_ag_2023_03_19_zs_models.parquet"
        path_automl = f'{path_bagged_root_s3}zs_input/automl_289/results_ranked_by_dataset_valid.csv'

    df_results, df_results_by_dataset, df_raw, df_metadata = load_results(
        results=results_path,
        results_by_dataset=results_by_dataset_path,
        raw=raw_path,
        metadata=str(Paths.data_root / "metadata" / "task_metadata_289.csv"),
        require_tid_in_metadata=True,
    )
    df_results_by_dataset = combine_results_with_score_val(df_raw, df_results_by_dataset)

    # Load in real framework results to score against
    print(f'Loading comparison_frameworks: {path_automl}')
    df_results_by_dataset_automl = load_pd.load(path_automl)

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
        max_size_mb_str = f'_{int(max_size_mb)}_mb' if max_size_mb is not None else ''
        path_zs_gt = str(path_bagged_root / f'zeroshot_gt{max_size_mb_str}.pkl')
        pred_pkl_path = path_bagged_root / f'zeroshot_pred_proba{max_size_mb_str}.pkl'
        zeroshot_pred_proba, zeroshot_gt, zsc = load_zeroshot_input(
            path_pred_proba=pred_pkl_path,
            path_gt=path_zs_gt,
            zsc=zsc,
            lazy_format=lazy_format,
        )

    time_end = time.time()
    print(f'Loaded ZS Context in {time_end-time_start:.2f}s')

    return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt


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
