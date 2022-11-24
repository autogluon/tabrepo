"""
Compare different methods that searches for ensemble configurations given offline evaluations.
Several strategies are available:
* all: evaluate the ensemble performance when using all model available
* zeroshot: evaluate the ensemble performance of zeroshot configurations
* zeroshot-ensemble: evaluate the ensemble performance of zeroshot configurations and when scoring list of models with
their ensemble performance
* randomsearch: performs a randomsearch after initializing the initial configuration with zeroshot
* localsearch: performs a localsearch after initializing the initial configuration with zeroshot. For each new
candidate, the best current configuration is mutated.

For random/local search, the search is done asynchronously with multiple workers.

Example:
PYTHONPATH=. python scripts/run_method_comparison.py --setting slow --n_workers 64
"""
import string
from argparse import ArgumentParser
from dataclasses import dataclass
import random
from typing import List

import numpy as np
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment

from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13, get_configs_small
from autogluon_zeroshot.loaders import Paths
from autogluon_zeroshot.simulation.config_generator import ZeroshotConfigGenerator
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.simulation.synetune_wrapper.synetune_search import RandomSearch, LocalSearch
from autogluon_zeroshot.utils import catchtime
from scripts.evaluate_ensemble import evaluate_ensemble

logging.getLogger().setLevel(logging.INFO)


def compute_zeroshot(models: List[str], datasets: List[str], ensemble_score: bool = False) -> List[str]:
    """evaluate the performance of a list of configurations with Caruana ensembles on the provided datasets"""
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=ensemble_score)
    dataset_ids = [zsc.dataset_to_tid_dict[k] for k in datasets]
    if ensemble_score:
        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=dataset_ids,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=10,
        )
    else:
        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=dataset_ids,
        )
    zs_config_generator = ZeroshotConfigGenerator(
        config_scorer=config_scorer,
        configs=models,
        backend="ray",
    )
    zeroshot_configs = zs_config_generator.select_zeroshot_configs(10, removal_stage=False)
    return zeroshot_configs


def learn_ensemble_configuration(
        train_datasets,
        test_datasets,
        models,
        num_folds,
        ensemble_size,
        num_base_models,
        max_wallclock_time,
        n_workers,
        max_num_trials_completed,
        name,
        searcher,
):
    assert searcher in ["randomsearch", "localsearch", "all", "zeroshot", "zeroshot-ensemble"]

    print(searcher)
    if searcher in ["randomsearch", "localsearch"]:
        synetune_logger = logging.getLogger("syne_tune")
        synetune_logger.setLevel(logging.WARNING)

        with catchtime("Compute zeroshot config to initialize local search"):
            zs_config = compute_zeroshot(models=models, datasets=train_datasets, ensemble_score=False)
        searcher_cls = LocalSearch if searcher == "localsearch" else RandomSearch

        scheduler = searcher_cls(
            models=models,
            metric='train_error',
            num_base_models=num_base_models,
            train_datasets=train_datasets,
            # Important note, we pass the test datasets for benchmarking purposes though but they are not used by the
            # searcher alternatively, an empty list can be passed but then test errors cannot be analyzed over time
            test_datasets=test_datasets,
            num_folds=num_folds,
            ensemble_size=ensemble_size,
            initial_suggestions=[zs_config[:num_base_models]],
        )

        tuner = Tuner(
            trial_backend=LocalBackend(entry_point=Path(__file__).parent / 'evaluate_ensemble.py'),
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(
                max_wallclock_time=max_wallclock_time,
                max_num_trials_completed=max_num_trials_completed
            ),
            n_workers=n_workers,
            tuner_name=name,
            metadata={"searcher": searcher}
        )
        tuner.run()

        tuning_experiment = load_experiment(tuner.name)
        # tuning_experiment.plot()

        print(tuning_experiment)

        print(f"best result found: {tuning_experiment.best_config()}")
        best_config_dict = eval(tuning_experiment.best_config()['config_configs'])

    elif searcher == "zeroshot":
        with catchtime("Compute zeroshot config to initialize local search"):
            best_config_dict = compute_zeroshot(models=models, datasets=train_datasets, ensemble_score=False)
    elif searcher == "zeroshot-ensemble":
        with catchtime("Compute zeroshot config to initialize local search"):
            best_config_dict = compute_zeroshot(models=models, datasets=train_datasets, ensemble_score=True)
    elif searcher == "all":
        best_config_dict = models

    train_error, test_error = evaluate_ensemble(
        configs=best_config_dict,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        num_folds=10,
        ensemble_size=ensemble_size,
    )

    return best_config_dict, train_error, test_error


@dataclass
class Arguments:
    n_workers: int
    num_folds: int
    ensemble_size: int
    max_wallclock_time: float
    num_base_models: int
    searchers: List[str]
    max_num_trials_completed: int = 10000


def random_string(length: int) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))


def get_setting(setting):
    if setting == "fast":
        return Arguments(
            num_folds=1,
            ensemble_size=20,
            max_wallclock_time=600,
            max_num_trials_completed=4,
            num_base_models=4,
            n_workers=input_args.n_workers,
            searchers=['localsearch'],
        )
    elif setting == "medium":
        return Arguments(
            num_folds=5,
            ensemble_size=20,
            max_wallclock_time=600,
            # max_num_trials_completed=100,
            num_base_models=10,
            n_workers=input_args.n_workers,
            searchers=['randomsearch', 'localsearch'],
        )
    elif setting == "slow":
        return Arguments(
            num_folds=10,
            ensemble_size=20,
            max_wallclock_time=1200,
            max_num_trials_completed=100000,
            num_base_models=10,
            n_workers=input_args.n_workers,
            searchers=[
                "zeroshot",
                "zeroshot-ensemble",
                "all",
                "randomsearch",
                "localsearch",
            ],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--setting", type=str, default="fast")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--expname", type=str)
    input_args, _ = parser.parse_known_args()

    if input_args.expname is None:
        expname = random_string(5)
    else:
        expname = input_args.expname

    args = get_setting(setting=input_args.setting)
    # TODO, for now we are using datanames rather than id, we may want to use dataset-id to have the same splits
    #  as the other zeroshot simulation script
    all_datasets = [
        'abalone', 'ada', 'adult', 'Amazon_employee_access', 'arcene', 'Australian', 'Bioresponse', 'black_friday',
        'blood-transfusion-service-center', 'boston', 'Brazilian_houses', 'car', 'christine', 'churn',
        'Click_prediction_small', 'cmc', 'cnae-9', 'colleges', 'credit-g', 'diamonds', 'dna', 'elevators', 'eucalyptus',
        'first-order-theorem-proving', 'GesturePhaseSegmentationProcessed', 'gina', 'house_prices_nominal',
        'house_sales',
        'Internet-Advertisements', 'jannis', 'jasmine', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kick',
        'madeline',
        'Mercedes_Benz_Greener_Manufacturing', 'MIP-2016-regression', 'Moneyball', 'numerai28_6', 'ozone-level-8hr',
        'PhishingWebsites', 'phoneme', 'pol', 'QSAR-TID-10980', 'QSAR-TID-11', 'quake', 'SAT11-HAND-runtime-regression',
        'Satellite', 'segment', 'sensory', 'shuttle', 'socmob', 'space_ga', 'steel-plates-fault', 'sylvine', 'tecator',
        'us_crime', 'vehicle', 'wilt', 'wine_quality', 'yprop_4_1']
    all_datasets = np.array(all_datasets)
    np.random.shuffle(all_datasets)
    models = get_configs_small()
    n_splits = input_args.n_splits
    if n_splits == 1:
        indices = np.arange(len(all_datasets))
        splits = [(indices[:len(indices) // 2], indices[len(indices) // 2:])]
    else:
        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        splits = kf.split(all_datasets)
        fold_results = []

    print(f"Running experiment {expname} with {input_args.setting} settings: {args}")

    results = []
    for i, (train_index, test_index) in enumerate(splits):
        for searcher in args.searchers:
            with catchtime(f'****Fitting method {searcher} on fold {i + 1}****'):
                train_datasets = list(all_datasets[train_index])
                test_datasets = list(all_datasets[test_index])

                best_config, train_error, test_error = learn_ensemble_configuration(
                    train_datasets=train_datasets,
                    test_datasets=test_datasets,
                    models=models,
                    num_folds=args.num_folds,
                    ensemble_size=args.ensemble_size,
                    num_base_models=args.num_base_models,
                    max_wallclock_time=args.max_wallclock_time,
                    n_workers=args.n_workers,
                    max_num_trials_completed=args.max_num_trials_completed,
                    searcher=searcher,
                    name=f"{expname}-fold-{i}-{searcher}"
                )
                print(f"best config found: {best_config}")
                print(f"train/test error of best config found: {train_error}/{test_error}")
                results.append({
                    'fold': i + 1,
                    'train-score': train_error,
                    'test-score': test_error,
                    'selected_configs': best_config,
                    'searcher': searcher,
                })
                csv_filename = Paths.results_root / f"{expname}.csv"
                print(f"update results in {csv_filename}")
                pd.DataFrame(results).to_csv(csv_filename, index=False)
    print(results)
