from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from .. import repository
from ..predictions.tabular_predictions import TabularModelPredictions
from ..simulation.configuration_list_scorer import ConfigurationListScorer
from ..simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels, EnsembleSelectionConfigScorer
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.single_best_config_scorer import SingleBestConfigScorer
from ..utils.cache import SaveLoadMixin


class EvaluationRepository(SaveLoadMixin):
    """
    Simple Repository class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.
    """
    def __init__(
            self,
            zeroshot_context: ZeroshotSimulatorContext,
            tabular_predictions: TabularModelPredictions,
            ground_truth: GroundTruth,
            config_fallback: str = None,
    ):
        """
        :param zeroshot_context:
        :param tabular_predictions:
        :param ground_truth:
        :param config_fallback: if specified, used to replace the result of a configuration that is missing, if not
        specified an error is thrown when querying a config that does not exist. A cheap baseline such as the result
        of a mean predictor can be used for the fallback.
        """
        self._tabular_predictions: TabularModelPredictions = tabular_predictions
        self._zeroshot_context: ZeroshotSimulatorContext = zeroshot_context
        self._ground_truth = ground_truth
        if self._tabular_predictions is not None:
            assert all(self._zeroshot_context.dataset_to_tid_dict[x] in self._tid_to_dataset_dict for x in self._tabular_predictions.datasets)
        self.set_config_fallback(config_fallback)

    def set_config_fallback(self, config_fallback: str = None):
        if config_fallback:
            assert config_fallback in self.configs()
        self._config_fallback = config_fallback

    def to_zeroshot(self) -> repository.EvaluationRepositoryZeroshot:
        """
        Returns a version of the repository as an EvaluationRepositoryZeroshot object.

        :return: EvaluationRepositoryZeroshot object
        """
        from .evaluation_repository_zeroshot import EvaluationRepositoryZeroshot
        self_zeroshot = copy.copy(self)  # Shallow copy so that the class update does not alter self
        self_zeroshot.__class__ = EvaluationRepositoryZeroshot
        return self_zeroshot

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
               ) -> "EvaluationRepository":
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

    # TODO: Make a better docstring, confusing what this `exactly` does
    # TODO: Add `is_dense` method to assist in unit tests + sanity checks
    def force_to_dense(self, inplace: bool = False, verbose: bool = True) -> "EvaluationRepository":
        """
        Method to force the repository to a dense representation inplace.

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        :param inplace: If True, will perform logic inplace.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return dense repo if inplace=False or self after inplace updates in this call.
        """
        if not inplace:
            return copy.deepcopy(self).force_to_dense(inplace=True, verbose=verbose)

        # TODO: Move these util functions to simulations or somewhere else to avoid circular imports
        from tabrepo.contexts.utils import intersect_folds_and_datasets, force_to_dense, prune_zeroshot_gt
        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(self._zeroshot_context, self._tabular_predictions, self._ground_truth)

        # TODO do we still need it? At the moment, we are using the fallback for models so this may not be necessary
        #  anymore
        # force_to_dense(self._tabular_predictions,
        #                first_prune_method='task',
        #                second_prune_method='dataset',
        #                verbose=verbose)

        self._zeroshot_context.subset_configs(self._tabular_predictions.models)
        datasets = [d for d in self._tabular_predictions.datasets if d in self._dataset_to_tid_dict]
        self._zeroshot_context.subset_datasets(datasets)

        self._tabular_predictions.restrict_models(self._zeroshot_context.get_configs())
        self._ground_truth = prune_zeroshot_gt(zeroshot_pred_proba=self._tabular_predictions,
                                               zeroshot_gt=self._ground_truth,
                                               dataset_to_tid_dict=self._dataset_to_tid_dict,
                                               verbose=verbose, )
        return self

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
    
    def compare_metrics(self, results_df: pd.DataFrame, datasets: List[str] = None, folds: List[int] = None, configs: List[str] = None) -> pd.DataFrame:
        df_exp = results_df.reset_index().set_index(["dataset", "fold", "framework"])[
            ["metric_error", "metric_error_val", "time_train_s", "time_infer_s"]
        ]

        #TO DO: use self._zeroshot_context.df_configs to match fit df and tab repo df
        # Will yield incorrect result - still in progress
        df_tr = self.metrics(datasets=datasets, folds=folds, configs=configs)
        #Future Question - Dropping rank as it is not in the results_df
        df_tr.drop(columns=['rank'], inplace=True)
        df_tr = df_tr.reset_index().set_index(["dataset", "fold","framework"])
        df = pd.concat([df_exp, df_tr], axis=0)
        df = df.sort_index()

        # Future question - for plotting how do match the 15 cols in results_df with the tabrepo columns?
        return df

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
        predictions = self._tabular_predictions.predict_test(
            dataset=dataset,
            fold=fold,
            models=configs,
            model_fallback=self._config_fallback,
        )
        return self._convert_binary_to_multiclass(dataset=dataset, predictions=predictions) if binary_as_multiclass else predictions

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
        predictions = self._tabular_predictions.predict_val(
            dataset=dataset,
            fold=fold,
            models=configs,
            model_fallback=self._config_fallback,
        )
        return self._convert_binary_to_multiclass(dataset=dataset, predictions=predictions) if binary_as_multiclass else predictions

    def labels_test(self, dataset: str, fold: int) -> np.array:
        return self._ground_truth.labels_test(dataset=dataset, fold=fold)

    def labels_val(self, dataset: str, fold: int) -> np.array:
        return self._ground_truth.labels_val(dataset=dataset, fold=fold)

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

    def evaluate_ensemble(
        self,
        datasets: List[str],
        configs: List[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        rank: bool = True,
        folds: Optional[List[int]] = None,
        backend: str = "ray",
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        :param datasets: list of datasets to compute errors on.
        :param configs: list of config to consider for ensembling. Uses all configs if None.
        :param ensemble_size: number of members to select with Caruana.
        :param ensemble_cls: class used for the ensemble model.
        :param ensemble_kwargs: kwargs to pass to the init of the ensemble class.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :param backend: Options include ["native", "ray"].
        :return: Tuple:
            Pandas Series of ensemble test errors per task, with multi-index (dataset, fold).
            Pandas DataFrame of ensemble weights per task, with multi-index (dataset, fold). Columns are the names of each config.
        """
        if folds is None:
            folds = self.folds
        if configs is None:
            configs = self.configs()
        tasks = [
            self.task_name(dataset=dataset, fold=fold)
            for dataset in datasets
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            tasks=tasks,
            ensemble_size=ensemble_size,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            backend=backend,
        )

        dict_errors, dict_ensemble_weights = scorer.compute_errors(configs=configs)

        if rank:
            dict_scores = scorer.compute_ranks(errors=dict_errors)
            out = dict_scores
        else:
            out = dict_errors

        dataset_folds = [(self.task_to_dataset(task=task), self.task_to_fold(task=task)) for task in tasks]
        ensemble_weights = [dict_ensemble_weights[task] for task in tasks]
        out_list = [out[task] for task in tasks]

        multiindex = pd.MultiIndex.from_tuples(dataset_folds, names=["dataset", "fold"])

        df_name = "rank" if rank else "error"
        df_out = pd.Series(data=out_list, index=multiindex, name=df_name)
        df_ensemble_weights = pd.DataFrame(data=ensemble_weights, index=multiindex, columns=configs)

        return df_out, df_ensemble_weights

    def _construct_config_scorer(self,
                                 config_scorer_type: str = 'ensemble',
                                 **config_scorer_kwargs) -> ConfigurationListScorer:
        if config_scorer_type == 'ensemble':
            return self._construct_ensemble_selection_config_scorer(**config_scorer_kwargs)
        elif config_scorer_type == 'single':
            return self._construct_single_best_config_scorer(**config_scorer_kwargs)
        else:
            raise ValueError(f'Invalid config_scorer_type: {config_scorer_type}')

    def _construct_ensemble_selection_config_scorer(self,
                                                    ensemble_size: int = 10,
                                                    backend='ray',
                                                    **kwargs) -> EnsembleSelectionConfigScorer:
        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            zeroshot_simulator_context=self._zeroshot_context,
            zeroshot_gt=self._ground_truth,
            zeroshot_pred_proba=self._tabular_predictions,
            ensemble_size=ensemble_size,  # 100 is better, but 10 allows to simulate 10x faster
            backend=backend,
            **kwargs,
        )
        return config_scorer

    def _construct_single_best_config_scorer(self, **kwargs) -> SingleBestConfigScorer:
        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=self._zeroshot_context,
            **kwargs,
        )
        return config_scorer

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

    @classmethod
    def from_context(cls, version: str = None, prediction_format: str = "memmap"):
        return load_repository(version=version, prediction_format=prediction_format)


def load_repository(version: str, *, load_predictions: bool = True, cache: bool | str = False, prediction_format: str = "memmap") -> EvaluationRepository:
    """
    Load the specified EvaluationRepository. Will automatically download all required inputs if they do not already exist on local disk.

    Parameters
    ----------
    version: str
        The name of the context to load.
    load_predictions: bool, default = True
        If True, loads the config predictions.
        If False, does not load the config predictions (disabling the ability to simulate ensembling).
    cache: bool | str, default = False
        Valid values: [True, False, "overwrite"]
        If True, will load directly from a cached repository or cache the loaded evaluation repository to accelerate future load calls.
        Setting to True may lead to incompatibility in loading repositories from different versions of the codebase.
        If "overwrite", will overwrite the existing cache and cache the new version.
    prediction_format: str, default = "memmap"
        Options: ["memmap", "memopt", "mem"]
        Determines the way the predictions are represented in the repo.
        It is recommended to keep the value as "memmap" for optimal performance.

    Returns
    -------
    EvaluationRepository object for the given context.
    """
    from tabrepo.contexts import get_subcontext
    if cache is not False:
        kwargs = dict()
        if isinstance(cache, str) and cache == "overwrite":
            kwargs["ignore_cache"] = True
            kwargs["exists"] = "overwrite"
        repo = get_subcontext(version).load(load_predictions=load_predictions, prediction_format=prediction_format, **kwargs)
    else:
        repo = get_subcontext(version).load_from_parent(load_predictions=load_predictions, prediction_format=prediction_format)
    return repo
