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
import logging
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{Path(__file__).parent}/log.txt"),
        logging.StreamHandler()
    ]
)

print(f"log can be found at {Path(__file__).parent}/log.txt")


import string
from argparse import ArgumentParser
from dataclasses import dataclass
import random
from typing import List

import numpy as np

import pandas as pd
from sklearn.model_selection import KFold
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment

from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13, get_configs_small
from autogluon_zeroshot.contexts.context_2022_12_11_bag import load_context_2022_12_11_bag
from autogluon_zeroshot.loaders import Paths
from autogluon_zeroshot.simulation.config_generator import ZeroshotConfigGenerator
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.simulation.synetune_wrapper.synetune_search import RandomSearch, LocalSearch
from autogluon_zeroshot.utils import catchtime
from scripts.method_comparison.evaluate_ensemble import evaluate_ensemble



def compute_zeroshot(
        models: List[str],
        datasets_folds: List[str],
        ensemble_size: int,
        num_models: int,
        num_folds: int,
        bag: bool,
        ensemble_score: bool = False,
) -> List[str]:
    """evaluate the performance of a list of configurations with Caruana ensembles on the provided datasets"""
    load_ctx = load_context_2022_12_11_bag if bag else load_context_2022_10_13
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_ctx(load_zeroshot_pred_proba=True, lazy_format=True)
    if ensemble_score:
        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=datasets_folds,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=ensemble_size,
            max_fold=num_folds,
        )
    else:
        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=zsc,
            datasets=datasets_folds,
        )
    zs_config_generator = ZeroshotConfigGenerator(
        config_scorer=config_scorer,
        configs=models,
        backend="ray",
    )
    metadata_list = zs_config_generator.select_zeroshot_configs(num_models, removal_stage=False)
    zeroshot_configs = metadata_list[-1]['configs']
    return zeroshot_configs


def learn_ensemble_configuration(
        train_datasets_folds,
        test_datasets_folds,
        configs,
        num_folds,
        ensemble_size,
        num_base_models,
        max_wallclock_time,
        n_workers,
        max_num_trials_completed,
        name,
        searcher,
        bag,
):
    assert searcher in ["randomsearch", "localsearch", "all", "zeroshot", "zeroshot-ensemble"]

    if searcher in ["randomsearch", "localsearch"]:
        synetune_logger = logging.getLogger("syne_tune")
        synetune_logger.setLevel(logging.WARNING)

        with catchtime("Compute zeroshot config to initialize local search"):
            zs_config = compute_zeroshot(
                models=configs,
                datasets_folds=train_datasets_folds,
                ensemble_score=False,
                ensemble_size=ensemble_size,
                num_models=num_base_models,
                num_folds=num_folds,
                bag=bag,
            )
        searcher_cls = LocalSearch if searcher == "localsearch" else RandomSearch

        scheduler = searcher_cls(
            models=configs,
            metric='train_error',
            num_base_models=num_base_models,
            train_datasets=train_datasets_folds,
            # Important note, we pass the test datasets for benchmarking purposes though but they are not used by the
            # searcher alternatively, an empty list can be passed but then test errors cannot be analyzed over time
            test_datasets=test_datasets_folds,
            num_folds=num_folds,
            ensemble_size=ensemble_size,
            initial_suggestions=[zs_config[:num_base_models]],
            backend="ray",
            bag=bag,
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

        logger.info(f"best result found: {tuning_experiment.best_config()}")
        best_config_dict = eval(tuning_experiment.best_config()['config_configs'])

    elif searcher == "zeroshot":
        with catchtime("Compute zeroshot without ensemble"):
            best_config_dict = compute_zeroshot(
                models=configs,
                datasets_folds=train_datasets_folds,
                ensemble_score=False,
                ensemble_size=ensemble_size,
                num_models=num_base_models,
                num_folds=num_folds,
                bag=bag,
            )
    elif searcher == "zeroshot-ensemble":
        with catchtime("Compute zeroshot with ensemble"):
            best_config_dict = compute_zeroshot(
                models=configs,
                datasets_folds=train_datasets_folds,
                ensemble_score=True,
                ensemble_size=ensemble_size,
                num_models=num_base_models,
                num_folds=num_folds,
                bag=bag,
            )
    elif searcher == "all":
        best_config_dict = configs

    with catchtime("eval performance of best config found"):
        train_error, test_error = evaluate_ensemble(
            configs=best_config_dict,
            train_datasets=train_datasets_folds,
            test_datasets=test_datasets_folds,
            num_folds=10,
            ensemble_size=ensemble_size,
            backend="ray",
            bag=bag,
        )

    return best_config_dict, train_error, test_error


@dataclass
class Arguments:
    n_workers: int  # number of workers used when tuning with syne tune
    num_folds: int  # number of folds to consider when fitting, can be lowered to reduce runtime
    ensemble_size: int  # number of ensemble to use for caruana ensemble computation
    num_base_models: int  # number of models that should be returned by search strategies
    searchers: List[str]  # list of searcher to run
    max_wallclock_time: float  # maximum wallclock time allowed for syne tune searchers
    max_num_trials_completed: int = 10000  # maximum number of choices that can be evaluated by syne tune searchers


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
            n_workers=1,
            searchers=[
                "zeroshot",
                "all",
                "randomsearch",
            ],
        )
    elif setting == "slow":
        return Arguments(
            num_folds=5,
            ensemble_size=10,
            max_wallclock_time=3600 * 2,
            max_num_trials_completed=100000,
            num_base_models=10,
            n_workers=input_args.n_workers,
            searchers=[
                "randomsearch",
                "localsearch",
                "zeroshot",
                # "zeroshot-ensemble",
                "all",
            ],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--setting", type=str, default="fast")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--expname", type=str)
    input_args, _ = parser.parse_known_args()
    if input_args.expname is None:
        expname = random_string(5)
    else:
        expname = input_args.expname

    args = get_setting(setting=input_args.setting)
    logger.info(f"Running experiment {expname} with {input_args.setting} settings: {args}/{input_args}")

    bag = True
    with catchtime("load"):
        load_ctx = load_context_2022_12_11_bag if bag else load_context_2022_10_13
        zsc, _, _, _ = load_ctx(load_zeroshot_pred_proba=True, lazy_format=True)
    configs = zsc.get_configs()

    # configs = get_configs_small()
    datasets = zsc.get_datasets()
    all_datasets = np.array(datasets)
    np.random.shuffle(all_datasets)
    n_splits = input_args.n_splits
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    splits = list(kf.split(all_datasets))

    # Evaluate all search strategies on `n_splits` of the datasets. Results are logged in a csv and can be
    # analysed with plot_results_comparison.py.
    results = []
    logger.info(f"Fitting {len(splits)} folds.")
    for i, (train_index, test_index) in enumerate(splits):
        for searcher in args.searchers:
            logger.info(f'****Fitting method {searcher} on fold {i + 1}****')
            train_datasets = list(all_datasets[train_index])
            test_datasets = list(all_datasets[test_index])
            best_config, train_error, test_error = learn_ensemble_configuration(
                train_datasets_folds=zsc.get_dataset_folds(train_datasets),
                test_datasets_folds=zsc.get_dataset_folds(test_datasets),
                configs=configs,
                num_folds=args.num_folds,
                ensemble_size=args.ensemble_size,
                num_base_models=args.num_base_models,
                max_wallclock_time=args.max_wallclock_time,
                n_workers=args.n_workers,
                max_num_trials_completed=args.max_num_trials_completed,
                searcher=searcher,
                name=f"{expname}-fold-{i}-{searcher}",
                bag=bag,
            )
            logger.info(f"best config found: {best_config}")
            logger.info(f"train/test error of best config found: {train_error}/{test_error}")
            results.append({
                'fold': i + 1,
                'train-score': train_error,
                'test-score': test_error,
                'selected_configs': best_config,
                'searcher': searcher,
            })
            csv_filename = Paths.results_root / f"{expname}.csv"
            logger.info(f"update results in {csv_filename}")
            pd.DataFrame(results).to_csv(csv_filename, index=False)
    logger.info(results)
