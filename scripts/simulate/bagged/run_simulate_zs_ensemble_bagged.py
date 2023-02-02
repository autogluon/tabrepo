import pickle
import sys
from pathlib import Path

import numpy as np

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.contexts.context_2022_12_11_bag import load_context_2022_12_11_bag, get_configs_default, get_configs_small
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation
from autogluon_zeroshot.portfolio import PortfolioCV
from autogluon_zeroshot.utils import catchtime

if __name__ == '__main__':
    with catchtime("load"):
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_12_11_bag(
            load_zeroshot_pred_proba=True,
            lazy_format=False,
            subset='small_30'
        )
    # zsc.subset_models(zeroshot_pred_proba.models)
    zsc.print_info()

    # NOTE: For speed of simulation, it is recommended backend='ray'
    #  If 'ray' isn't available, then specify 'native'.
    backend = 'ray'

    configs = zsc.get_configs()
    if configs is not None:
        zeroshot_pred_proba.restrict_models(configs)

    results_cv_list = []
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
        results_cv: PortfolioCV = run_zs_simulation(
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

    save_pkl.save(path=str(Path(__file__).parent / 'sim_results' / 'ensemble_result_bagged.pkl'), object=results_cv)
