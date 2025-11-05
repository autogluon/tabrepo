"""Example of loading a custom task for TabArena.

This file assumes the dataset is preprocessed and saved as a CSV file to `dataset_file`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from tabarena.benchmark.task import UserTask

TASK_CACHE_DIR = str(Path(__file__).parent / "tabarena_out" / "local_tasks")
"""Output for artefacts from the evaluation results of the custom model."""


def get_tasks_for_tabarena(
    dataset_file: str = "biopsie_preprocessed_full_cohort.csv",
) -> UserTask:
    """Generate a local task to be used by TabArena for the Biopsy dataset."""
    dataset = pd.read_csv(Path(__file__).parent / dataset_file)
    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Create a stratified 10-repeated 3-fold split (any other split can be used as well)
    n_repeats, n_splits = 10, 3
    sklearn_splits = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42).split(
        X=dataset.drop(columns=["target"]), y=dataset["target"]
    )
    # Transform the splits into a standard dictionary format expected by TabArena
    splits = {}
    for split_i, (train_idx, test_idx) in enumerate(sklearn_splits):
        repeat_i = split_i // n_splits
        fold_i = split_i % n_splits
        if repeat_i not in splits:
            splits[repeat_i] = {}
        splits[repeat_i][fold_i] = (train_idx.tolist(), test_idx.tolist())

    user_task = UserTask(
        task_name=f"BiopsyCancerPrediction_{dataset_file}",
        task_cache_path=Path(TASK_CACHE_DIR),
    )
    oml_task = user_task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task
