from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from typing_extensions import Self

from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.single_best_config_scorer import SingleBestConfigScorer
from ..utils.cache import SaveLoadMixin


class AbstractRepository(ABC, SaveLoadMixin):
    def __init__(
        self,
        zeroshot_context: ZeroshotSimulatorContext,
        config_fallback: str | None = None
    ):
        self._zeroshot_context = zeroshot_context
        self._config_fallback = None
        self.set_config_fallback(config_fallback)

    def set_config_fallback(self, config_fallback: str = None):
        if config_fallback:
            assert config_fallback in self.configs()
        self._config_fallback = config_fallback

    def print_info(self):
        self._zeroshot_context.print_info()

    @property
    def _dataset_to_tid_dict(self) -> Dict[str, int]:
        return self._zeroshot_context.dataset_to_tid_dict

    @property
    def _tid_to_dataset_dict(self) -> Dict[int, str]:
        return {v: k for k, v in self._dataset_to_tid_dict.items()}

    def subset(self,
               datasets: List[str] = None,
               folds: List[int] = None,
               configs: List[str] = None,
               problem_types: List[str] = None,
               force_to_dense: bool = True,
               inplace: bool = False,
               verbose: bool = True,
               ) -> Self:
        """
        Method to subset the repository object and force to a dense representation.

        :param datasets: The list of datasets to subset. Ignored if unspecified.
        :param folds: The list of folds to subset. Ignored if unspecified.
        :param configs: The list of configs to subset. Ignored if unspecified.
        :param problem_types: The list of problem types to subset. Ignored if unspecified.
        :param force_to_dense: If True, forces the output to dense representation.
        :param inplace: If True, will perform subset logic inplace.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return subsetted repo if inplace=False or self after inplace updates in this call.
        """
        if not inplace:
            return copy.deepcopy(self).subset(
                datasets=datasets,
                folds=folds,
                configs=configs,
                problem_types=problem_types,
                force_to_dense=force_to_dense,
                inplace=True,
                verbose=verbose,
            )
        if folds is not None:
            self._zeroshot_context.subset_folds(folds=folds)
        if configs is not None:
            self._zeroshot_context.subset_configs(configs=configs)
        if datasets is not None:
            self._zeroshot_context.subset_datasets(datasets=datasets)
        if problem_types is not None:
            self._zeroshot_context.subset_problem_types(problem_types=problem_types)
        if force_to_dense:
            self.force_to_dense(inplace=True, verbose=verbose)
        return self

    @abstractmethod
    def force_to_dense(self, inplace: bool = False, verbose: bool = True) -> Self:
        """
        Method to force the repository to a dense representation inplace.

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        :param inplace: If True, will perform logic inplace.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return dense repo if inplace=False or self after inplace updates in this call.
        """
        raise NotImplementedError

    @property
    def _df_metadata(self) -> pd.DataFrame:
        return self._zeroshot_context.df_metadata

    def tids(self, problem_type: str = None) -> List[int]:
        """
        Note: returns the taskid of the datasets rather than the string name.

        :param problem_type: If specified, only datasets with the given problem_type are returned.
        """
        return self._zeroshot_context.get_tids(problem_type=problem_type)

    def datasets(self, problem_type: str = None, union: bool = True) -> List[str]:
        """
        Return all valid datasets.
        By default, will return all datasets that appear in any config at least once.

        Parameters
        ----------
        problem_type : str, default = None
            If specified, will only consider datasets with the given problem type
        union: bool, default = True
            If True, will return the union of datasets present in each config.
            If False, will return the intersection of datasets present in each config.

        Returns
        -------
        A list of dataset names satisfying the above conditions.
        """
        return self._zeroshot_context.get_datasets(problem_type=problem_type, union=union)

    def configs(self, *, datasets: List[str] = None, tasks: List[str] = None, union: bool = True) -> List[str]:
        """
        Return all valid configs.
        By default, will return all configs that appear in any task at least once.

        Parameters
        ----------
        datasets : List[str], default = None
            If specified, will only consider the configs present in the given datasets
        tasks: List[str], default = None
            If specified, will only consider the configs present in the given tasks
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns
        -------
        A list of config names satisfying the above conditions.
        """
        return self._zeroshot_context.get_configs(datasets=datasets, tasks=tasks, union=union)

    def dataset_to_tid(self, dataset: str) -> int:
        return self._dataset_to_tid_dict[dataset]

    def tid_to_dataset(self, tid: int) -> str:
        return self._tid_to_dataset_dict.get(tid, "Not found")

    def metrics(self, datasets: List[str] = None, folds: List[int] = None, configs: List[str] = None) -> pd.DataFrame:
        """
        :param datasets:
        :param folds:
        :param configs: list of configs to query metrics
        :return: pd.DataFrame of metrics for each dataset-fold-framework combination.

        Output is a multi-index pandas DataFrame ("dataset", "fold", "framework").
        Each row is a result for a particular config on a given task.
        If a config does not have a result for a given task, it will not have a row present in the DataFrame for that task.
        Columns:
            metric_error : Test error of the config
            metric_error_val : Validation error of the config
            time_train_s : Training time of the config
            time_infer_s : Inference time of the config
            rank : Rank of the config
        """
        df = self._zeroshot_context.df_configs_ranked.set_index(["dataset", "fold", "framework"], drop=True)[
            ["metric_error", "metric_error_val", "time_train_s", "time_infer_s", "rank"]
        ]
        if datasets is None:
            datasets = self.datasets()

        mask = df.index.get_level_values("dataset").isin(datasets)
        if folds is not None:
            mask = mask & df.index.get_level_values("fold").isin(folds)
        if configs is not None:
            mask = mask & df.index.get_level_values("framework").isin(configs)
        df = df[mask]

        return df

    def dataset_fold_config_pairs(self, datasets: List[str] = None, folds: List[int] = None, configs: List[str] = None) -> list:
        """
        Returns a list of all (dataset, fold, config) tuples available in the repo.

        Parameters
        ----------
        datasets: List[str], default None
            Filters the output to only contain datasets in the specified list.
        folds: List[int], default None
            Filters the output to only contain folds in the specified list.
        configs: List[str], default None
            Filters the output to only contain configs in the specified list.
        """
        return self.metrics(datasets=datasets, folds=folds, configs=configs).index.tolist()

    def predict_test(self, dataset: str, fold: int, config: str, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the test set for a given configuration on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        config: str
            The model config to get predictions from. Must be a value in `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_rows, n_classes).
            If False, will return binary predictions in shape (n_rows), with the value being class 1 (the positive class).

            You can convert from (n_rows, n_classes) -> (n_rows) via `predictions[:, 1]`.
            You can convert from (n_rows) -> (n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions on the test set with shape (n_rows, n_classes) for multiclass or (n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        """
        return self.predict_test_multi(dataset=dataset, fold=fold, configs=[config], binary_as_multiclass=binary_as_multiclass).squeeze()

    def predict_val(self, dataset: str, fold: int, config: str, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        config: str
            The model config to get predictions from. Must be a value in `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_rows, n_classes).
            If False, will return binary predictions in shape (n_rows), with the value being class 1 (the positive class).

            You can convert from (n_rows, n_classes) -> (n_rows) via `predictions[:, 1]`.
            You can convert from (n_rows) -> (n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions on the validation set with shape (n_rows, n_classes) for multiclass or (n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        """
        return self.predict_val_multi(dataset=dataset, fold=fold, configs=[config], binary_as_multiclass=binary_as_multiclass).squeeze()

    @abstractmethod
    def predict_test_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the test set for a given list of configurations on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        configs: List[str], default = None
            The model configs to get predictions from.
            If None, will use `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_configs, n_rows, n_classes).
            If False, will return binary predictions in shape (n_configs, n_rows), with the value being class 1 (the positive class).

            You can convert from (n_configs, n_rows, n_classes) -> (n_configs, n_rows) via `predictions[:, :, 1]`.
            You can convert from (n_configs, n_rows) -> (n_configs, n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_configs, n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions with shape (n_configs, n_rows, n_classes) for multiclass or (n_configs, n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        The output order will be the same order as `configs`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_val_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        """
        Returns the predictions on the validation set for a given list of configurations on a given dataset and fold

        Parameters
        ----------
        dataset: str
            The dataset to get predictions from. Must be a value in `self.datasets()`.
        fold: int
            The fold of the dataset to get predictions from.
        configs: List[str], default = None
            The model configs to get predictions from.
            If None, will use `self.configs()`.
        binary_as_multiclass: bool, default = False
            If True, will return binary predictions in shape (n_configs, n_rows, n_classes).
            If False, will return binary predictions in shape (n_configs, n_rows), with the value being class 1 (the positive class).

            You can convert from (n_configs, n_rows, n_classes) -> (n_configs, n_rows) via `predictions[:, :, 1]`.
            You can convert from (n_configs, n_rows) -> (n_configs, n_rows, n_classes) via `np.stack([1 - predictions, predictions], axis=predictions.ndim)`.

            The internal representation is of form (n_configs, n_rows) as it requires less memory,
            so there is a conversion overhead introduced when `binary_as_multiclass=True`.

        Returns
        -------
        The model predictions with shape (n_configs, n_rows, n_classes) for multiclass or (n_configs, n_rows) in case of regression.
        For binary, shape depends on `binary_as_multiclass` value.
        The output order will be the same order as `configs`.
        """
        raise NotImplementedError

    def dataset_metadata(self, dataset: str) -> dict:
        metadata = self._df_metadata[self._df_metadata["dataset"] == dataset]
        return dict(zip(metadata.columns, metadata.values[0]))

    def dataset_info(self, dataset: str) -> dict:
        """
        Parameters
        ----------
        dataset: str

        Returns
        -------
        Dictionary with two keys:
            "metric": The evaluation metric name used for scoring on the dataset
            "problem_type": The problem type of the dataset
        """
        return self._zeroshot_context.df_metrics.loc[dataset].to_dict()

    @property
    def folds(self) -> List[int]:
        return self._zeroshot_context.folds

    def n_folds(self) -> int:
        return len(self.folds)

    def n_datasets(self) -> int:
        return len(self.datasets())

    def n_configs(self) -> int:
        return len(self.configs())

    def task_name_from_tid(self, tid: int, fold: int) -> str:
        return self._zeroshot_context.task_name_from_tid(tid=tid, fold=fold)

    def task_name(self, dataset: str, fold: int) -> str:
        return self.task_name_from_tid(tid=self.dataset_to_tid(dataset), fold=fold)

    def task_to_dataset(self, task: str) -> str:
        return self._zeroshot_context.task_to_dataset_dict[task]

    def task_to_fold(self, task: str) -> int:
        return self._zeroshot_context.task_to_fold(task=task)

    def _construct_single_best_config_scorer(self, **kwargs) -> SingleBestConfigScorer:
        return SingleBestConfigScorer.from_repo(repo=self, **kwargs)

    def _convert_binary_to_multiclass(self, predictions: np.ndarray, dataset: str) -> np.ndarray:
        """
        Converts binary predictions in (n_rows) format to (n_rows, n_classes) format.
        Converts binary predictions in (n_configs, n_rows) format to (n_configs, n_rows, n_classes) format.
        Skips conversion if dataset's problem_type != "binary".

        Parameters
        ----------
        predictions: np.ndarray
            The predictions to convert.
        dataset: str
            The dataset the predictions originate from.

        Returns
        -------
        Returns converted predictions if binary, else returns original predictions.
        """
        if self.dataset_info(dataset)["problem_type"] == "binary":
            return np.stack([1 - predictions, predictions], axis=predictions.ndim)
        else:
            return predictions
