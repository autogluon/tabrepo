import numpy as np
import pandas as pd
from autogluon.common.loaders import load_pd

from autogluon_zeroshot.simulation.config_generator import ZeroshotConfigGeneratorCV
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.loaders import load_configs, load_results, combine_results_with_score_val
from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext


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

folds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

zsc = ZeroshotSimulatorContext(
    df_results_by_dataset=df_results_by_dataset,
    df_results_by_dataset_automl=df_results_by_dataset_automl,
    df_raw=df_raw,
    folds_to_use=folds_to_use,
)

df_results_by_dataset_vs_automl = zsc.df_results_by_dataset_vs_automl
dataset_name_to_tid_dict = zsc.dataset_name_to_tid_dict
dataset_to_tid_dict = zsc.dataset_to_tid_dict
dataset_name_to_fold_dict = zsc.dataset_name_to_fold_dict
unique_dataset_folds = zsc.unique_dataset_folds
unique_datasets = zsc.unique_datasets
rank_scorer_vs_automl = zsc.rank_scorer_vs_automl

datasets = unique_dataset_folds
print(datasets)
print(len(datasets))

autogluon_configs = [
    'CatBoost_c1',
    'LightGBM_c1',
    'LightGBM_c2',
    'LightGBM_c3',
    'NeuralNetFastAI_c1',
    'RandomForest_c1',
    'ExtraTrees_c1',
]

small_extra_configs = []
for m in [
    'LightGBM',
    'CatBoost',
    'RandomForest',
    'ExtraTrees',
    'NeuralNetFastAI',
]:
    for i in range(1, 12):
        small_extra_configs.append(m + f'_r{i}')
small_configs = autogluon_configs + small_extra_configs

configs_all = zsc.get_configs()
print(configs_all)

path_zs_pred_proba = f'{path_prefix}/zeroshot_pred_proba_2022_10_13_zs.pkl'
path_zs_gt = f'{path_prefix}/zeroshot_gt_2022_10_13_zs.pkl'
# zeroshot_pred_proba, zeroshot_gt = zsc.load_zeroshot_pred_proba(path_pred_proba=path_zs_pred_proba, path_gt=path_zs_gt)
# zeroshot_pred_proba = zsc.minimize_memory_zeroshot_pred_proba(zeroshot_pred_proba=zeroshot_pred_proba, configs=small_configs)

config_scorer_single_best = SingleBestConfigScorer(
    df_results_by_dataset=df_results_by_dataset_vs_automl,
    datasets=datasets,
)

# config_scorer_ensemble = EnsembleSelectionConfigScorer(
#     datasets=datasets,
#     zeroshot_gt=zeroshot_gt,
#     zeroshot_pred_proba=zeroshot_pred_proba,
#     ranker=rank_scorer_vs_automl,
#     ensemble_size=100,
#     dataset_name_to_tid_dict=dataset_name_to_tid_dict,
#     dataset_name_to_fold_dict=dataset_name_to_fold_dict,
# )

b = config_scorer_single_best.score(configs=autogluon_configs)
print(b)

b = config_scorer_single_best.score(configs=small_configs)
print(b)

# TODO: Add simulation results for CV
zs_single_best_config_cv = ZeroshotConfigGeneratorCV(
    n_splits=2,
    df_results_by_dataset=df_results_by_dataset_vs_automl,
    zeroshot_sim_name='ZS_SingleBest_CV2',
    config_scorer=config_scorer_single_best,
    unique_datasets_map=dataset_name_to_tid_dict,
    configs=small_configs,
)


# zs_ensemble_config_cv = ZeroshotConfigGeneratorCV(
#     n_splits=2,
#     df_results_by_dataset=df_results_by_dataset_vs_automl,
#     zeroshot_sim_name='ZS_Ensemble_CV2',
#     config_scorer=config_scorer_ensemble,
#     unique_datasets_map=dataset_name_to_tid_dict,
#     configs=small_configs,
# )

df_raw_zeroshots_single_best = zs_single_best_config_cv.run()
print(f'Final Score: {np.mean(df_raw_zeroshots_single_best)}')
# df_raw_zeroshots_ensemble = zs_ensemble_config_cv.run()
# print(f'Final Score: {np.mean(df_raw_zeroshots_ensemble)}')


if __name__ == '__main__':
    pass
