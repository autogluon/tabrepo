from pathlib import Path

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.portfolio import PortfolioCV
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2023_03_19_bag_289 import load_context_2023_03_19_bag_289
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation


if __name__ == '__main__':
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2023_03_19_bag_289()
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    backend = 'ray'

    # configs = get_configs_small()
    configs = None

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

    print(f'Final Score: {results_cv.get_test_score_overall()}')

    save_pkl.save(path=str(Path(__file__).parent / 'sim_results' / 'single_best_result_bagged.pkl'), object=results_cv)
