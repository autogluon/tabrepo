from autogluon.common.loaders import load_pd

from ..loaders import load_configs, load_results, combine_results_with_score_val
from ..simulation.simulation_context import ZeroshotSimulatorContext


def load_context_2022_10_13(folds=None, load_zeroshot_pred_proba=False) -> (ZeroshotSimulatorContext, dict, dict, dict):
    if folds is None:
        folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    path_prefix = '../data/results/all_v3'
    df_results, df_results_by_dataset, df_raw, df_metadata = load_results(
        results=f"{path_prefix}/results_ranked_valid.csv",
        results_by_dataset=f"{path_prefix}/results_ranked_by_dataset_valid.parquet",
        raw=f"{path_prefix}/openml_ag_2022_10_13_zs_models.parquet",
        metadata=f"../data/metadata/task_metadata.csv",
    )
    df_results_by_dataset = combine_results_with_score_val(df_raw, df_results_by_dataset)

    # Load in real framework results to score against
    path_prefix_automl = '../data/results/automl'
    df_results_by_dataset_automl = load_pd.load(f'{path_prefix_automl}/results_ranked_by_dataset_valid.csv')

    zsc = ZeroshotSimulatorContext(
        df_results_by_dataset=df_results_by_dataset,
        df_results_by_dataset_automl=df_results_by_dataset_automl,
        df_raw=df_raw,
        folds=folds,
    )

    configs_prefix_1 = '../data/configs/configs_20221004'
    configs_prefix_2 = '../data/configs'
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
        path_zs_pred_proba = f'{path_prefix}/zeroshot_pred_proba_2022_10_13_zs.pkl'
        path_zs_gt = f'{path_prefix}/zeroshot_gt_2022_10_13_zs.pkl'
        zeroshot_pred_proba, zeroshot_gt = zsc.load_zeroshot_pred_proba(path_pred_proba=path_zs_pred_proba,
                                                                        path_gt=path_zs_gt)

    return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt
