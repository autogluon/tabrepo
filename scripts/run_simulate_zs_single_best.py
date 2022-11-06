import numpy as np

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13, get_configs_default, get_configs_small
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation


if __name__ == '__main__':
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13()
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    #  If 'ray' isn't available, then specify 'native'.
    #  Speed comparison on m6i.16xlarge:
    #   native: 19.04s per forward selection step
    #   ray:    1.49s  per forward selection step (>10x faster)
    # Overall time requirement for all configs & datasets:
    #  fs_time * fs_rounds * n_splits
    #  For 10-fold with 20 rounds on ray: 1.49s * 20 * 10 = 298s
    #  For LOO with 20 rounds on ray: 1.49s * 20 * 60 = 1788s
    backend = 'ray'

    configs = get_configs_small()

    score_total = 0
    len_datasets_total = 0
    results_total = []
    # for problem_type in ['binary', 'multiclass', 'regression']:
    for problem_type in [None]:
        datasets = zsc.get_dataset_folds(problem_type=problem_type)

        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=datasets,
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

    save_pkl.save(path='./sim_results/single_best_result.pkl', object=results_total)
