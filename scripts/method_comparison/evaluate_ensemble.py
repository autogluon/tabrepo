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
from syne_tune import Reporter

from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13
from autogluon_zeroshot.contexts.context_2022_12_11_bag import load_context_2022_12_11_bag
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.utils import catchtime


def evaluate_ensemble(
        configs: List[dict],
        train_datasets: List[str],
        test_datasets: List[str],
        ensemble_size: int,
        num_folds: int = 10,
        backend: str = "native",
        bag: bool = False,
):
    print(f"Evaluating on {backend}")
    load_ctx = load_context_2022_12_11_bag if bag else load_context_2022_10_13

    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_ctx(
        load_zeroshot_pred_proba=True, lazy_format=True,
    )
    train_scorer = EnsembleSelectionConfigScorer.from_zsc(
        datasets=train_datasets,
        zeroshot_simulator_context=zsc,
        zeroshot_gt=zeroshot_gt,
        zeroshot_pred_proba=zeroshot_pred_proba,
        ensemble_size=ensemble_size,
        max_fold=num_folds,
        backend=backend,
    )
    train_error = train_scorer.score(configs)
    if len(test_datasets) > 0:
        test_scorer = EnsembleSelectionConfigScorer.from_zsc(
            datasets=test_datasets,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=ensemble_size,
            max_fold=num_folds,
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

    reporter = Reporter()
    with catchtime("evaluate ensemble"):
        train_error, test_error = evaluate_ensemble(
            configs=configs,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            num_folds=num_folds,
            ensemble_size=ensemble_size,
            backend=backend,
        )
        metrics = {
            "train_error": train_error
        }
        if len(test_datasets) > 0:
            metrics["test_error"] = test_error
        reporter(**metrics)
