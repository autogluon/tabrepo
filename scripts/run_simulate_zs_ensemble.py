import numpy as np

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13, get_configs_default, get_configs_small
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation


if __name__ == '__main__':
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=True)
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    #  If 'ray' isn't available, then specify 'native'.
    backend = 'ray'

    configs = get_configs_small()
    if configs is not None:
        zeroshot_pred_proba = zsc.minimize_memory_zeroshot_pred_proba(zeroshot_pred_proba=zeroshot_pred_proba,
                                                                      configs=configs)
    score_total = 0
    len_datasets_total = 0
    results_total = []
    # for problem_type in ['binary', 'multiclass', 'regression']:
    for problem_type in [None]:
        datasets = zsc.get_dataset_folds(problem_type=problem_type)

        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            datasets=datasets,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=10,  # 100 is better, but 10 allows to simulate 10x faster
        )

        len_datasets = len(datasets)
        len_datasets_total += len_datasets
        results = run_zs_simulation(
            zsc=zsc,
            config_scorer=config_scorer,
            n_splits=5,
            configs=configs,
            backend=backend,
        )
        score = np.mean([r['score'] for r in results])
        print(f'{problem_type}: {score} | {len_datasets}')
        score_total += score*len_datasets
        results_total += results
    score_total = score_total / len_datasets_total
    print(f'Final Score: {score_total}')

    save_pkl.save(path='./sim_results/ensemble_result.pkl', object=results_total)
