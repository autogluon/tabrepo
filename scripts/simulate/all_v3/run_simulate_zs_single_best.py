from pathlib import Path

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.portfolio import PortfolioCV
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2022_10_13 import get_configs_small
from autogluon_zeroshot.contexts import get_context, list_contexts
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation


if __name__ == '__main__':
    context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)
    benchmark_context.download()
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load()
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

    results_cv_list = []
    # for problem_type in ['binary', 'multiclass', 'regression']:
    for problem_type in [None]:
        datasets = zsc.get_dataset_folds(problem_type=problem_type)

        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=datasets,
        )

        len_datasets = len(datasets)
        results_cv = run_zs_simulation(
            zsc=zsc,
            config_scorer=config_scorer,
            n_splits=5,
            configs=configs,
            backend=backend,
        )
        print(f'{problem_type}: {results_cv.get_test_score_overall()} | {len_datasets}')
        results_cv_list.append(results_cv)
    results_cv = PortfolioCV.combine(results_cv_list)
    # Final Score: 4.8197 | 61 datasets, n_splits=5, 608 configs
    print(f'Final Score: {results_cv.get_test_score_overall()}')

    save_pkl.save(path=str(Path(__file__).parent / 'sim_results' / 'single_best_result.pkl'), object=results_cv)
