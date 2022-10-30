import numpy as np
import pandas as pd
from autogluon.common.loaders import load_pd, load_pkl

from autogluon_zeroshot.utils.rank_utils import RankScorer
from autogluon_zeroshot.simulation.config_generator import ZeroshotEnsembleConfigCV
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.simulation.sim_utils import get_dataset_to_tid_dict, get_dataset_name_to_tid_dict, filter_datasets
from autogluon_zeroshot.loaders import load_configs, load_results, combine_results_with_score_val


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

unique_dataset_folds = set(list(df_results_by_dataset['dataset'].unique()))
df_results_by_dataset_automl = df_results_by_dataset_automl[df_results_by_dataset_automl['dataset'].isin(unique_dataset_folds)]

unique_dataset_folds = set(list(df_results_by_dataset_automl['dataset'].unique()))
df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                df_raw=df_raw,
                                                datasets=unique_dataset_folds)

folds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a = df_results_by_dataset[['tid', 'fold']].drop_duplicates()
a = a[a['fold'].isin(folds_to_use)]
b = a['tid'].value_counts()
b = b[b == len(folds_to_use)]
unique_datasets = list(b.index)

dataset_name_to_fold_dict = df_results_by_dataset[['dataset', 'fold']].drop_duplicates().set_index('dataset')['fold'].to_dict()

dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
unique_dataset_folds_list = []
unique_datasets_set = set(unique_datasets)
for dataset in unique_dataset_folds:
    if dataset_name_to_tid_dict[dataset] in unique_datasets_set:
        unique_dataset_folds_list.append(dataset)
unique_dataset_folds = set(unique_dataset_folds_list)

df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                df_raw=df_raw,
                                                datasets=unique_dataset_folds)

dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
dataset_to_tid_dict = get_dataset_to_tid_dict(df_raw=df_raw)

automl_error_dict = {}
for i, dataset in enumerate(unique_dataset_folds_list):
    automl_error_list = sorted(list(df_results_by_dataset_automl[df_results_by_dataset_automl['dataset'] == dataset]['metric_error']))
    automl_error_dict[dataset] = automl_error_list

rank_scorer_vs_automl = RankScorer(df_results_by_dataset=df_results_by_dataset_automl, datasets=unique_dataset_folds_list)
df_results_by_dataset_vs_automl = df_results_by_dataset.copy()
df_results_by_dataset_vs_automl['rank'] = [rank_scorer_vs_automl.rank(r[1], r[0]) for r in zip(df_results_by_dataset_vs_automl['metric_error'], df_results_by_dataset_vs_automl['dataset'])]


##########
# PART TWO
df_metadata_tmp = df_metadata[df_metadata['tid'].isin(unique_datasets)][['tid', 'name']]
dataset_list = list(df_metadata_tmp['name'])
# numerai is evil and has a period in its name, which breaks everything
# Buzz idk why it fails
# FIXME: Use TID instead of names when creating zeroshot_gt and zeroshot_pred_proba as keys, life will be much better.
banned_datasets = [
    'numerai28.6',
    'Buzzinsocialmedia_Twitter',
]

datasets_clean = [d for d in dataset_list if d not in banned_datasets]
datasets_clean = [dataset_to_tid_dict[d] for d in datasets_clean]
datasets_clean = unique_dataset_folds_list
# datasets_clean = [d for d in datasets_clean if '_0' in d]
print(datasets_clean)
print(len(datasets_clean))

print('Loading zeroshot...')
zeroshot_gt = load_pkl.load(f'{path_prefix}/zeroshot_gt_2022_10_13_zs.pkl')
# NOTE: This file is BIG (17 GB)
zeroshot_pred_proba = load_pkl.load(f'{path_prefix}/zeroshot_pred_proba_2022_10_13_zs.pkl')
print('Loading successful!')

zeroshot_gt = {k: v for k, v in zeroshot_gt.items() if k in dataset_to_tid_dict}
zeroshot_gt = {dataset_name_to_tid_dict[dataset_to_tid_dict[k]]: v for k, v in zeroshot_gt.items()}

zeroshot_pred_proba = {k: v for k, v in zeroshot_pred_proba.items() if k in dataset_to_tid_dict}
zeroshot_pred_proba = {dataset_name_to_tid_dict[dataset_to_tid_dict[k]]: v for k, v in zeroshot_pred_proba.items()}

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

# small_configs = autogluon_configs

small_configs_set = set(small_configs)


# import sys
# import pickle
# size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
# print(f'OLD Size: {round(size_bytes / 1e6, 3)} MB')
# task_names = list(zeroshot_pred_proba.keys())
# for t in task_names:
#     model_keys = list(zeroshot_pred_proba[t][0]['pred_proba_dict_val'].keys())
#     for k in model_keys:
#         if k not in small_configs_set:
#             zeroshot_pred_proba[t][0]['pred_proba_dict_val'].pop(k)
#             zeroshot_pred_proba[t][0]['pred_proba_dict_test'].pop(k)
# print('shrunk zeroshot_pred_proba')
#
# size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
# print(f'NEW Size: {round(size_bytes / 1e6, 3)} MB')

config_scorer_single_best = SingleBestConfigScorer(
    df_results_by_dataset_with_score_val=df_results_by_dataset_vs_automl,
    datasets=datasets_clean,
)

# FIXME: Remove folds logic from this class, handled in CV class
config_scorer_ensemble = EnsembleSelectionConfigScorer(
    datasets=datasets_clean,
    folds=[0],
    zeroshot_gt=zeroshot_gt,
    zeroshot_pred_proba=zeroshot_pred_proba,
    ranker=rank_scorer_vs_automl,
    ensemble_size=100,
    dataset_name_to_tid_dict=dataset_name_to_tid_dict,
    dataset_name_to_fold_dict=dataset_name_to_fold_dict,
)

b = config_scorer_single_best.score(configs=autogluon_configs)
print(b)
# a = config_scorer_ensemble.score(configs=autogluon_configs)
# print(a)

b = config_scorer_single_best.score(configs=small_configs)
print(b)
# a = config_scorer_ensemble.score(configs=small_configs)
# print(a)

# TODO: Add simulation results for CV
zs_single_best_config_cv = ZeroshotEnsembleConfigCV(
    n_splits=2,
    df_results_by_dataset=df_results_by_dataset_vs_automl,
    zeroshot_sim_name='ZS_SingleBest_CV2',
    config_scorer=config_scorer_single_best,
    unique_datasets_map=dataset_name_to_tid_dict,
    configs=small_configs,
)


zs_ensemble_config_cv = ZeroshotEnsembleConfigCV(
    n_splits=2,
    df_results_by_dataset=df_results_by_dataset_vs_automl,
    zeroshot_sim_name='ZS_Ensemble_CV2',
    config_scorer=config_scorer_ensemble,
    unique_datasets_map=dataset_name_to_tid_dict,
    configs=small_configs,
)

df_raw_zeroshots_single_best = zs_single_best_config_cv.run()
print(f'Final Score: {np.mean(df_raw_zeroshots_single_best)}')
df_raw_zeroshots_ensemble = zs_ensemble_config_cv.run()
print(f'Final Score: {np.mean(df_raw_zeroshots_ensemble)}')


if __name__ == '__main__':
    pass
