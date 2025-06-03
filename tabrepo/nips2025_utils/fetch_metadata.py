from __future__ import annotations

from pathlib import Path

import pandas as pd

from autogluon.common.loaders import load_pd
import tabrepo


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
            <=100k samples, <=500 features, <=10 classes, classification
            36/51 datasets

    Returns
    -------
    task_metadata: pd.DataFrame
        Metadata about each dataset in the benchmark.

    """
    tabrepo_root = str(Path(tabrepo.__file__).parent.parent)
    if paper:
        path = f"{tabrepo_root}/tabflow/metadata/task_metadata_tabarena51.csv"
    else:
        path = f"{tabrepo_root}/tabflow/metadata/task_metadata_tabarena61.csv"
    task_metadata = load_pd.load(path=path)
    task_metadata["dataset"] = task_metadata["name"]

    if subset is None:
        pass
    if subset == "TabPFNv2":
        task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 15000]
        task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 501]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]
    elif subset == "TabICL":
        task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 150000]
        task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 501]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] > 0]
    else:
        raise AssertionError(f"Unknown subset: {subset}")

    return task_metadata
