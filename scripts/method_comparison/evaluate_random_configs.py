"""
Evaluate random ensmble configurations on train and test datasets which can be used to measure correlation between
train and test scores.
"""
import logging
import random
import string
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from autogluon_zeroshot.contexts import get_context
from autogluon_zeroshot.loaders import Paths
from autogluon_zeroshot.utils import catchtime
from scripts.method_comparison.evaluate_ensemble import evaluate_ensemble

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

def random_string(length: int) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))


def split_datasets(datasets, n_splits):
    np.random.shuffle(datasets)
    if n_splits == 1:
        indices = np.arange(len(datasets))
        splits = [(indices[:len(indices) // 2], indices[len(indices) // 2:])]
    else:
        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        splits = kf.split(datasets)
    return splits


def eval_randomconfig(
        train_datasets,
        test_datasets,
        configs,
        num_base_models,
):
    train_datasets_folds = zsc.get_dataset_folds(train_datasets)
    test_datasets_folds = zsc.get_dataset_folds(test_datasets)
    random_perm = np.random.permutation(len(configs))
    random_configs = [configs[i] for i in random_perm[:num_base_models]]
    train_error, _ = evaluate_ensemble(
        configs=random_configs,
        train_datasets=train_datasets_folds,
        test_datasets=[],
        num_folds=num_folds_fit,
        ensemble_size=ensemble_size,
        backend="ray",
    )
    _, test_error = evaluate_ensemble(
        configs=random_configs,
        train_datasets=[],
        test_datasets=test_datasets_folds,
        num_folds=10,
        ensemble_size=ensemble_size,
        backend="ray",
    )
    return train_error, test_error, random_configs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--expname", type=str)
    input_args, _ = parser.parse_known_args()

    if input_args.expname is None:
        expname = random_string(5)
    else:
        expname = input_args.expname

    ensemble_size = 20
    num_evaluations = 100
    num_base_models = 10
    n_splits = [2, 5]
    num_folds_fits = [5, 10]
    bag = True

    if bag:
        context_name = 'BAG_D104_F10_C608_FULL'
    else:
        context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)

    with catchtime("load"):
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(load_predictions=False)
    configs = zsc.get_configs()
    datasets = np.array(zsc.get_datasets())

    results = []

    exps = [
        (n_split, (fold, (train_index, test_index)), num_folds_fit, eval_index)
        for n_split in n_splits
        for fold, (train_index, test_index) in enumerate(split_datasets(datasets, n_split))
        for num_folds_fit in num_folds_fits
        for eval_index in range(num_evaluations)
    ]

    for n_split, (fold, (train_index, test_index)), num_folds_fit, eval_index in tqdm(exps):
        print(f"fold: {fold}/{n_split} eval {eval_index}/{num_evaluations} with {num_folds_fit} folds to fit")
        train_datasets = list(datasets[train_index])
        test_datasets = list(datasets[test_index])
        train_error, test_error, random_configs = eval_randomconfig(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            configs=configs,
            num_base_models=num_base_models,
        )
        results.append({
            'fold': fold,
            'train-score': train_error,
            'test-score': test_error,
            'selected_configs': random_configs,
            'num_folds_fit': num_folds_fit,
            'n_splits': n_split,
        })
        print(results[-1])
        csv_filename = Paths.results_root / f"random-evals-{expname}.csv"
        print(f"update results in {csv_filename}")
        pd.DataFrame(results).to_csv(csv_filename, index=False)
    print(results)
