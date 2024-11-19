from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .abstract_repository import AbstractRepository


def convert_time_infer_s_from_sample_to_batch(df: pd.DataFrame, repo: "AbstractRepository") -> pd.DataFrame:
    """
    Temp: Multiply by 0.1 since 90% of the instances are used for training and 10% for test
    # TODO: Change this in future, not all tasks will use 90% train / 10% test. Instead keep track of train/test rows per dataset_fold pair.
    """
    df = df.copy(deep=True)
    if "dataset" in df.columns:
        df["time_infer_s"] = df["time_infer_s"] * df["dataset"].map(
            repo.task_metadata.set_index("dataset")["NumberOfInstances"]
        ) * 0.1
    else:
        df["time_infer_s"] = df["time_infer_s"] * df.index.get_level_values("dataset").map(
            repo.task_metadata.set_index("dataset")["NumberOfInstances"]
        ) * 0.1
    return df


def convert_time_infer_s_from_batch_to_sample(df: pd.DataFrame, repo: "AbstractRepository") -> pd.DataFrame:
    """
    Temp: Multiply by 0.1 since 90% of the instances are used for training and 10% for test
    # TODO: Change this in future, not all tasks will use 90% train / 10% test. Instead keep track of train/test rows per dataset_fold pair.
    """
    df = df.copy(deep=True)
    if "dataset" in df.columns:
        df["time_infer_s"] = df["time_infer_s"] / (df["dataset"].map(
            repo.task_metadata.set_index("dataset")["NumberOfInstances"]
        ) * 0.1)
    else:
        df["time_infer_s"] = df["time_infer_s"] / (df.index.get_level_values("dataset").map(
            repo.task_metadata.set_index("dataset")["NumberOfInstances"]
        ) * 0.1)
    return df