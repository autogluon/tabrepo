from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import openml
from openml.datasets.functions import create_dataset
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    TaskType,
)

if TYPE_CHECKING:
    import pandas as pd


# Patch Functions for OpenML Dataset
def _get_dataset(self, **kwargs) -> openml.datasets.OpenMLDataset:
    return self.local_dataset


class UserTask:
    """A user-defined task to run on custom datasets or tasks."""

    def __init__(
        self,
        *,
        task_name: str,
        dataset: pd.DataFrame,
        target_feature: str,
        problem_type: Literal["classification", "regression"],
        splits: dict[int, dict[int, tuple[list, list]]],
    ):
        """Initialize a user-defined task.

        Parameters
        ----------
        task_name: str
            The name of the task. Make sure this is a unique name on your system,
            as we create the cache based on this name.
        dataset: pd.DataFrame
            The dataset to be used for the task. It should be a pandas DataFrame
            with features and target variable. Moreover, make sure the DataFrame
            has the correct dtypes for each column, as this will be used
            to infer the metadata of features. Thus, make sure that:
                - Numerical features are of a number type.
                - Categorical features are of type 'category'.
                - Text features are of a string type.
                - Date features are of a date type.
        target_feature: str
            The name of the target feature in the dataset. This must be a column
            in the dataset DataFrame.
        problem_type: Literal['classification', 'regression']
            The type of problem to be solved. It can be either 'classification'
            or 'regression'.
        splits: dict[int, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]]
            A dictionary the train-tests splits per repeat and fold.
            These splits represent the outer splits that are used to evaluate models,
            and not the inner splits used for tuning/validation/HPO.

            The structure is:
            {
                repeat_id: {
                    split_id: {
                        (train_indices, test_indices)
                    }
                    ...
                }
                ...
            }
            where train_indices and test_indices are lists of indices, starting from 0.

            Note the following assumptions:
                - The indices in train_indices and test_indices do not overlap.
                - Per repeat_id, one can have multiple split_ids, but the test_indices
                  should not overlap across split_ids.
                - Splits across repeat_ids should have the same structure (e.g., if
                  there is only one split in repeat_id 0, there should be only one split
                  in all other repeat_ids).
        """
        self.task_name = task_name
        self._dataset = dataset
        self.target_feature = target_feature
        self.problem_type = problem_type
        self.splits = splits
        self._task_name_hash = hashlib.sha256(
            self.task_name.encode("utf-8")
        ).hexdigest()

        self._validate_splits(splits=splits, n_samples=len(dataset))

    @staticmethod
    def _validate_splits(
        *, splits: dict[int, dict[int, tuple[list, list]]], n_samples: int
    ) -> None:
        """Validate the splits passed by the user."""
        if not isinstance(splits, dict):
            raise ValueError("Splits must be a dictionary.")

        found_structure = None
        for repeat_id, split_dict in splits.items():
            if not isinstance(split_dict, dict):
                raise ValueError(f"Splits for repeat {repeat_id} must be a dictionary.")
            test_indices_per_repeat = set()
            for split_id, (train_indices, test_indices) in split_dict.items():
                if not isinstance(train_indices, list) or not isinstance(
                    test_indices, list
                ):
                    raise ValueError(f"Indices for split {split_id} must be lists.")
                if not all(
                    isinstance(idx, int) for idx in train_indices + test_indices
                ):
                    raise ValueError(
                        f"All indices in split {split_id} must be integers."
                    )
                if len(train_indices) == 0 or len(test_indices) == 0:
                    raise ValueError(
                        f"Train and test indices in split {split_id} must not be empty."
                    )
                if set(train_indices) & set(test_indices):
                    raise ValueError(
                        f"Train and test indices in split {split_id} must not overlap."
                    )
                if any(np.array(train_indices + test_indices) < 0):
                    raise ValueError(
                        f"Indices in split {split_id} must be non-negative."
                    )
                if any(np.array(train_indices + test_indices) >= n_samples):
                    raise ValueError(
                        f"Indices in split {split_id} must not exceed the dataset size (0 to {n_samples - 1})."
                    )
                if test_indices_per_repeat & set(test_indices):
                    raise ValueError(
                        f"Test indices in split {split_id} must not overlap with previous splits in repeat {repeat_id}."
                    )
                test_indices_per_repeat.update(test_indices)

            if found_structure is None:
                found_structure = len(split_dict)
            elif found_structure != len(split_dict):
                raise ValueError("All repeats must have the same number of splits.")

    @property
    def task_id(self) -> int:
        """Generate a unique task ID based on the task name and a UUID.
        This is used to identify the task, for example, when caching results.
        """
        return int(self._task_name_hash, 16) % 10**10

    @property
    def _local_dataset_id(self) -> str:
        return self._task_name_hash

    @property
    def _local_cache_path(self) -> Path:
        return (
            Path(openml.config._root_cache_directory)
            / "local"
            / "datasets"
            / self._local_dataset_id
        )

    # TODO: support local OpenML tasks inside of OpenML code...
    def to_openml_task(self) -> OpenMLSupervisedTask:
        """Convert the user-defined task to a local (unpublished) OpenMLSupervisedTask."""
        task_type = (
            TaskType.SUPERVISED_CLASSIFICATION
            if self.problem_type == "classification"
            else TaskType.SUPERVISED_REGRESSION
        )
        extra_kwargs = {}
        if task_type == TaskType.SUPERVISED_CLASSIFICATION:
            task_cls = OpenMLClassificationTask  # type: ignore
            extra_kwargs["class_labels"] = list(
                np.unique(self._dataset[self.target_feature])
            )
        elif task_type == TaskType.SUPERVISED_REGRESSION:
            task_cls = OpenMLRegressionTask  # type: ignore
        else:
            raise NotImplementedError(f"Task type {task_type:d} not supported.")

        local_dataset = create_dataset(
            name=f"LocalDataset-{self.task_name}",
            description=None,
            creator=None,
            contributor=None,
            collection_date=None,
            language=None,
            licence=None,
            attributes="auto",
            data=self._dataset,
            default_target_attribute=self.target_feature,
            ignore_attribute=None,
            citation="N/A",
            row_id_attribute=None,
            original_data_url=None,
            paper_url=None,
            version_label=None,
            update_comment=None,
        )
        # Cache data to disk
        parquet_file = self._local_cache_path / "data.pq"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        self._dataset.to_parquet(parquet_file)
        del self._dataset  # Free memory

        # We only need local_dataset.get_data() from the OpenMLDataset, thus, we make
        # sure with the code below that get_data() returns the data.
        local_dataset.parquet_file = parquet_file
        local_dataset.data_file = "ignored"  # not used for local datasets

        # Create the task
        task = task_cls(
            task_id=self.task_id,
            task_type_id=task_type,
            task_type="None",  # Placeholder, not used for local tasks
            data_set_id=-1,  # Placeholder, not used for local tasks
            target_name=self.target_feature,
            **extra_kwargs,
        )
        task.local_dataset = local_dataset
        task.get_dataset = _get_dataset.__get__(task, OpenMLSupervisedTask)

        # Transform TabArena splits to OpenMLSplit format
        openml_splits = {}
        for repeat in self.splits:
            openml_splits[repeat] = OrderedDict()
            for fold in self.splits[repeat]:
                openml_splits[repeat][fold] = OrderedDict()
                # We do not support learning curves tasks, so no need for samples.
                openml_splits[repeat][fold][0] = (
                    np.array(self.splits[repeat][fold][0], dtype=int),
                    np.array(self.splits[repeat][fold][1], dtype=int),
                )

        task.split = openml.tasks.split.OpenMLSplit(
            name="User-Splits",
            description="User-defined splits for a custom task.",
            split=openml_splits,
        )

        return task
