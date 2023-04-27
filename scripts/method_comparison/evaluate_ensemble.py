"""
Main script to evaluate an ensemble configuration, used to run tuning with syne tune and as util to get
train/test scores for ensemble configurations.
"""
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import KFold
from syne_tune import Reporter

from autogluon_zeroshot.contexts import get_context
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.utils import catchtime


def evaluate_ensemble(
        configs: List[str],
        train_datasets: List[str],
        test_datasets: List[str],
        ensemble_size: int,
        num_folds: int = 10,
        backend: str = "native",
        bag: bool = False,
        n_splits: int = None,
):
    """
    :param configs: model configurations to evaluate, for instance ['CatBoost_r2_BAG_L1', 'LightGBM_r12_BAG_L1']
    :param train_datasets: dataset with fold to evaluate for instance ['359979_0', '359979_1', '359979_2', ...]
    :param test_datasets:
    :param ensemble_size: number of ensembles to compute in greedy caruana.
    :param num_folds: number of folds to consider in the datasets, used to quickly check things work and only works with n_splits=None.
    :param backend: native or ray
    :param bag: whether to load configurations evaluated with bagging or without
    :param n_splits: The train error is estimated by taking the average score on `n_splits` subsets of `train_datasets`.
    If the value is None, then all datasets are used.
    :return:
    """
    assert backend in ['native', 'ray']
    print(f"Evaluating backend/bag/ensemble_size/num_folds:{backend}/{bag}/{ensemble_size}/{num_folds}/{n_splits}")
    if bag:
        context_name = 'BAG_D104_F10_C608_FULL'
    else:
        context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(
        load_predictions=True, lazy_format=True,
    )
    if n_splits is not None:
        datasets = np.array(train_datasets)
        np.random.shuffle(datasets)
        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        splits = kf.split(datasets)
        train_datasets_splits = [
            [datasets[i] for i in train_indices]
            for (train_indices, _) in splits
        ]
    else:
        train_datasets_splits = [train_datasets]
    # compute errors on all splits and then take average to estimate the final performance
    train_errors = []
    for datasets in train_datasets_splits:
        train_score = EnsembleSelectionConfigScorer.from_zsc(
            datasets=datasets,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=ensemble_size,
            max_fold=num_folds,
            backend=backend,
        ).score(configs)
        train_errors.append(train_score)
    train_error = np.mean(train_errors)
    if len(test_datasets) > 0:
        test_scorer = EnsembleSelectionConfigScorer.from_zsc(
            datasets=test_datasets,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=ensemble_size,
            max_fold=num_folds,
            backend=backend,
        )
        test_error = test_scorer.score(configs)
    else:
        test_error = None
    return train_error, test_error


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(f"--st_checkpoint_dir", type=str, default="./")
    args, _ = parser.parse_known_args()

    # gets hyperparameters that are written into {trial_path}/config.json
    # note: only works with LocalBackend for now.
    trial_path = Path(args.st_checkpoint_dir).parent
    with open(trial_path / "config.json", "r") as f:
        config = json.load(f)
    print(args.__dict__)
    print(config)
    configs = config['configs']
    train_datasets = config['train_datasets']
    test_datasets = config['test_datasets']
    num_folds = config['num_folds']
    ensemble_size = config['ensemble_size']
    backend = config["backend"]
    bag = bool(config["bag"])

    reporter = Reporter()
    with catchtime("evaluate ensemble"):
        train_error, test_error = evaluate_ensemble(
            configs=configs,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            num_folds=num_folds,
            ensemble_size=ensemble_size,
            backend=backend,
            bag=bag,
        )
        metrics = {
            "train_error": train_error
        }
        if len(test_datasets) > 0:
            metrics["test_error"] = test_error
        reporter(**metrics)
