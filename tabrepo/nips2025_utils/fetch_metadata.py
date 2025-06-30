from __future__ import annotations

from pathlib import Path

import pandas as pd

from autogluon.common.loaders import load_pd
import tabrepo


def _get_n_repeats(n_instances: int) -> int:
    """
    Get the number of n_repeats for the full benchmark run based on the 2025 paper.
    If < 2500 samples, n_repeats = 10, else n_repeats = 3

    Parameters
    ----------
    n_instances: int

    Returns
    -------
    n_repeats: int
    """
    if n_instances < 2500:
        n_repeats = 10
    else:
        n_repeats = 3
    return n_repeats


def load_task_metadata(paper: bool = True, subset: str = None) -> pd.DataFrame:
    """
    Load the task metadata for all datasets in the TabArena benchmark.

    Parameters
    ----------
    paper: bool, default True
        If True, returns the task_metadata for the 51 NeurIPS 2025 TabArena paper datasets
        If False, returns the task_metadata for 61 datasets prior to filtering down to 51 datasets for the paper.
    subset: {None, "TabPFNv2", "TabICL"}, default None
        If None, returns all tasks.
        If "TabPFNv2", filters to tasks compatible with TabPFNv2:
            <=10k samples, <=500 features, <=10 classes
            33/51 datasets
        If "TabICL", filters to tasks compatible with TabICL:
            <=100k samples, <=500 features, classification
            36/51 datasets

    Returns
    -------
    task_metadata: pd.DataFrame
        Metadata about each dataset in the benchmark.

    """
    tabrepo_root = str(Path(tabrepo.__file__).parent.parent)
    if paper:
        path = f"{tabrepo_root}/tabrepo/nips2025_utils/metadata/task_metadata_tabarena51.csv"
    else:
        path = f"{tabrepo_root}/tabrepo/nips2025_utils/metadata/task_metadata_tabarena61.csv"
    task_metadata = load_pd.load(path=path)

    task_metadata["n_folds"] = 3
    task_metadata["n_repeats"] = task_metadata["NumberOfInstances"].apply(_get_n_repeats)
    task_metadata["n_features"] = (task_metadata["NumberOfFeatures"] - 1).astype(int)
    task_metadata["n_samples_test_per_fold"] = (task_metadata["NumberOfInstances"] / task_metadata["n_folds"]).astype(int)
    task_metadata["n_samples_train_per_fold"] = (task_metadata["NumberOfInstances"] - task_metadata["n_samples_test_per_fold"]).astype(int)

    task_metadata["dataset"] = task_metadata["name"]

    if subset is None:
        pass
    elif subset == "TabPFNv2":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 10000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]
    elif subset == "TabICL":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 100000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] > 0]
    else:
        raise AssertionError(f"Unknown subset: {subset}")

    return task_metadata
