import numpy as np

from autogluon_zeroshot.simulation.config_generator import ZeroshotConfigGeneratorCV
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13

zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=True)
zsc.print_info()

datasets = zsc.unique_dataset_folds

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

config_scorer_single_best = SingleBestConfigScorer.from_zsc(
    zeroshot_simulator_context=zsc,
    datasets=datasets,
)
# TODO: Add simulation results for CV
zs_single_best_config_cv = ZeroshotConfigGeneratorCV(
    n_splits=2,
    zeroshot_simulator_context=zsc,
    zeroshot_sim_name='ZS_SingleBest_CV2',
    config_scorer=config_scorer_single_best,
    configs=small_configs,
)

if zeroshot_pred_proba is not None:
    zeroshot_pred_proba = zsc.minimize_memory_zeroshot_pred_proba(zeroshot_pred_proba=zeroshot_pred_proba,
                                                                  configs=small_configs)
config_scorer_ensemble = EnsembleSelectionConfigScorer.from_zsc(
    datasets=datasets,
    zeroshot_simulator_context=zsc,
    zeroshot_gt=zeroshot_gt,
    zeroshot_pred_proba=zeroshot_pred_proba,
    ensemble_size=100,
)
zs_ensemble_config_cv = ZeroshotConfigGeneratorCV(
    n_splits=2,
    zeroshot_simulator_context=zsc,
    zeroshot_sim_name='ZS_Ensemble_CV2',
    config_scorer=config_scorer_ensemble,
    configs=small_configs,
)

score = config_scorer_single_best.score(configs=autogluon_configs)
print(score)
score = config_scorer_single_best.score(configs=small_configs)
print(score)

df_raw_zeroshots_single_best = zs_single_best_config_cv.run()
print(f'Final Test Score (Single Best): {np.mean(df_raw_zeroshots_single_best)}')
df_raw_zeroshots_ensemble = zs_ensemble_config_cv.run()
print(f'Final Test Score (Ensemble): {np.mean(df_raw_zeroshots_ensemble)}')


if __name__ == '__main__':
    pass
