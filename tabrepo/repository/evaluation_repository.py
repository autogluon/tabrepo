from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from tabrepo.simulation.configuration_list_scorer import ConfigurationListScorer
from tabrepo.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from tabrepo.simulation.ground_truth import GroundTruth
from tabrepo.simulation.simulation_context import ZeroshotSimulatorContext
from tabrepo.simulation.single_best_config_scorer import SingleBestConfigScorer
from tabrepo.predictions.tabular_predictions import TabularModelPredictions
from tabrepo.utils.cache import SaveLoadMixin
from tabrepo.utils import catchtime
from tabrepo import repository


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
    ):
        self._tabular_predictions: TabularModelPredictions = tabular_predictions
        self._zeroshot_context: ZeroshotSimulatorContext = zeroshot_context
        self._ground_truth = ground_truth
        if self._tabular_predictions is not None:
            assert all(self._zeroshot_context.dataset_to_tid_dict[x] in self._tid_to_name for x in self._tabular_predictions.datasets)

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
    def _name_to_tid(self) -> Dict[str, int]:
        return self._zeroshot_context.dataset_to_tid_dict

    @property
    def _tid_to_name(self) -> Dict[int, str]:
        return {v: k for k, v in self._name_to_tid.items()}

    def subset(self,
               folds: List[int] = None,
               models: List[str] = None,
               tids: List[Union[str, int]] = None,
               problem_types: List[str] = None,
               verbose: bool = True,
               ):
        """
        Method to subset the repository object and force to a dense representation.

        :param folds: The list of folds to subset. Ignored if unspecified.
        :param models: The list of models to subset. Ignored if unspecified.
        :param tids: The list of dataset task ids to subset. Ignored if unspecified.
        :param problem_types: The list of problem types to subset. Ignored if unspecified.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return self after in-place updates in this call.
        """
        if folds:
            self._zeroshot_context.subset_folds(folds=folds)
        if models:
            self._zeroshot_context.subset_models(models=models)
        if tids:
            # TODO: Align `_zeroshot_context` naming of datasets -> tids
            self._zeroshot_context.subset_tids(tids=tids)
        if problem_types:
            self._zeroshot_context.subset_problem_types(problem_types=problem_types)
        self.force_to_dense(verbose=verbose)
        return self

    # TODO: Add `is_dense` method to assist in unit tests + sanity checks
    def force_to_dense(self, verbose: bool = True):
        """
        Method to force the repository to a dense representation inplace.

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return self after in-place updates in this call.
        """
        # TODO: Move these util functions to simulations or somewhere else to avoid circular imports
        from tabrepo.contexts.utils import intersect_folds_and_datasets, force_to_dense, prune_zeroshot_gt
        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(self._zeroshot_context, self._tabular_predictions, self._ground_truth)

        force_to_dense(self._tabular_predictions,
                       first_prune_method='task',
                       second_prune_method='dataset',
                       verbose=verbose)

        self._zeroshot_context.subset_models(self._tabular_predictions.models)
        datasets = [d for d in self._tabular_predictions.datasets if d in self._name_to_tid]
        self._zeroshot_context.subset_datasets(datasets)
        self._tabular_predictions.restrict_models(self._zeroshot_context.get_configs())
        self._ground_truth = prune_zeroshot_gt(zeroshot_pred_proba=self._tabular_predictions,
                                               zeroshot_gt=self._ground_truth,
                                               dataset_to_tid_dict=self._name_to_tid,
                                               verbose=verbose,)
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

    def datasets(self, problem_type: str = None) -> List[str]:
        return self._zeroshot_context.get_datasets(problem_type=problem_type)

    def list_models_available(self, tid: int) -> List[str]:
        # TODO rename with new name, and keep naming convention of tabular_predictions to allow filtering over folds,
        #  datasets, specify whether all need to be present etc
        """
        :param tid:
        :return: the list of configurations that are available on all folds of the given dataset.
        """
        res = set(self._tabular_predictions.list_models_available(datasets=[tid]))
        for fold in self.folds:
            df = self._zeroshot_context.df_results_by_dataset_vs_automl
            task = self.task_name(tid=tid, fold=fold)
            methods = set(df.loc[df.dataset == task, "framework"].unique())
            res = res.intersection(methods)
        return list(sorted(res))

    # TODO: Unify with `list_models_available`
    def list_models(self) -> List[str]:
        """
        List all models that appear in at least one task.

        :return: the list of configurations that are available in at least one task.
        """
        return self._zeroshot_context.get_configs()

    def dataset_to_tid(self, dataset: str) -> int:
        return self._name_to_tid[dataset]

    def tid_to_dataset(self, tid: int) -> str:
        return self._tid_to_name.get(tid, "Not found")

    def eval_metrics(self, tid: int, config_names: List[str], fold: int, check_all_found: bool = True) -> List[dict]:
        """
        :param tid:
        :param config_names: list of configs to query metrics
        :param fold:
        :return: list of metrics for each configuration
        """
        df = self._zeroshot_context.df_results_by_dataset_vs_automl
        task = self.task_name(tid=tid, fold=fold)
        mask = (df["task"] == task) & (df["framework"].isin(config_names))
        output_cols = ["framework", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "rank",]
        if check_all_found:
            assert sum(mask) == len(config_names), \
                f"expected one evaluation occurence for each configuration {config_names} for {tid}, " \
                f"{fold} but found {sum(mask)}."
        return [dict(zip(output_cols, row)) for row in df.loc[mask, output_cols].values]

    # FIXME: Rename
    def val_predictions(self, tid: int, config_name: str, fold: int) -> np.array:
        """
        Returns the predictions on the validation set for a given configuration on a given dataset and fold
        :return: the model predictions with shape (n_rows, n_classes) or (n_rows) in case of regression
        """
        return self.predict_val(tid=tid, fold=fold, configs=[config_name]).squeeze()

    # FIXME: Rename
    def test_predictions(self, tid: int, config_name: str, fold: int) -> np.array:
        """
        Returns the predictions on a test set for a given configuration on a given dataset and fold
        :return: the model predictions with shape (n_rows, n_classes) or (n_rows) in case of regression
        """
        return self.predict_test(tid=tid, fold=fold, configs=[config_name]).squeeze()

    def predict_test(self, tid: int, fold: int, configs: List[str] = None) -> np.ndarray:
        dataset = self.tid_to_dataset(tid)
        return self._tabular_predictions.predict_test(
            dataset=dataset,
            fold=fold,
            models=configs,
        )

    def predict_val(self, tid: int, fold: int, configs: List[str] = None) -> np.ndarray:
        dataset = self.tid_to_dataset(tid)
        return self._tabular_predictions.predict_val(
            dataset=dataset,
            fold=fold,
            models=configs,
        )

    def labels_test(self, tid: int, fold: int) -> np.array:
        return self._ground_truth.labels_test(tid=tid, fold=fold)

    def labels_val(self, tid: int, fold: int) -> np.array:
        return self._ground_truth.labels_val(tid=tid, fold=fold)

    def dataset_metadata(self, tid: int) -> dict:
        metadata = self._df_metadata[self._df_metadata.tid == tid]
        return dict(zip(metadata.columns, metadata.values[0]))

    def dataset_info(self, tid: int) -> dict:
        """
        Parameters
        ----------
        tid

        Returns
        -------
        Dictionary with two keys:
            "metric": The evaluation metric name used for scoring on the dataset
            "problem_type": The problem type of the dataset
        """
        dataset = self.tid_to_dataset(tid=tid)
        return self._zeroshot_context.df_metrics.loc[dataset].to_dict()

    @property
    def folds(self) -> List[int]:
        return self._zeroshot_context.folds

    def n_folds(self) -> int:
        return len(self.folds)

    def n_datasets(self) -> int:
        return len(self.datasets())

    def n_models(self) -> int:
        return len(self.list_models())

    @staticmethod
    def task_name(tid: int, fold: int) -> str:
        return f"{tid}_{fold}"

    def task_name_from_dataset(self, dataset: str, fold: int) -> str:
        return self.task_name(tid=self.dataset_to_tid(dataset), fold=fold)

    def evaluate_ensemble(
        self,
        tids: List[int],
        config_names: List[str],
        ensemble_size: int,
        rank: bool = True,
        folds: Optional[List[int]] = None,
        backend: str = "ray",
    ) -> Tuple[np.array, Dict[str, np.array]]:
        """
        :param tids: list of dataset tids to compute errors on.
        :param config_names: list of config to consider for ensembling.
        :param ensemble_size: number of members to select with Caruana.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :return: Tuple:
            2D array of scores whose rows are datasets and columns are folds.
            Dictionary of task_name -> model weights in the ensemble. Model weights are stored in a numpy array,
                with weights corresponding to the order of `config_names`.
        """
        if folds is None:
            folds = self.folds
        tasks = [
            self.task_name(tid=tid, fold=fold)
            for tid in tids
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            datasets=tasks,
            ensemble_size=ensemble_size,
            backend=backend,
        )

        dict_errors, dict_ensemble_weights = scorer.compute_errors(configs=config_names)
        if rank:
            dict_scores = scorer.compute_ranks(errors=dict_errors)
            out = dict_scores
        else:
            out = dict_errors

        out_numpy = np.array([[
                out[self.task_name(tid=tid, fold=fold)
            ] for fold in folds
        ] for tid in tids])

        return out_numpy, dict_ensemble_weights

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

    @classmethod
    def from_context(cls, version: str = None, lazy_format: bool = True):
        return load(version=version, lazy_format=lazy_format)


def load(version: str = None) -> EvaluationRepository:
    from tabrepo.contexts import get_context
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = get_context(version).load(load_predictions=True)
    r = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )
    r = r.force_to_dense(verbose=True)
    return r


# TODO: git shelve ADD BACK
if __name__ == '__main__':
    from tabrepo.contexts.context_artificial import load_repo_artificial

    with catchtime("loading repo and evaluating one ensemble config"):
        dataset_name = "abalone"
        config_name = "NeuralNetFastAI_r1"
        # repo = EvaluationRepository.load(version="2022_10_13")

        repo = load_repo_artificial()
        tid = repo.dataset_to_tid(dataset=dataset_name)
        print(repo.datasets()[:3])  # ['abalone', 'ada', 'adult']
        print(repo.tids()[:3])  # [2073, 3945, 7593]

        print(tid)  # 360945
        print(list(repo.list_models_available(tid))[:3])  # ['LightGBM_r181', 'CatBoost_r81', 'ExtraTrees_r33']
        print(repo.eval_metrics(tid=tid, config_names=[config_name], fold=2))  # {'time_train_s': 0.4008138179779053, 'metric_error': 25825.49788, ...
        print(repo.val_predictions(tid=tid, config_name=config_name, fold=2).shape)
        print(repo.test_predictions(tid=tid, config_name=config_name, fold=2).shape)
        print(repo.dataset_metadata(tid=tid))  # {'tid': 360945, 'ttid': 'TaskType.SUPERVISED_REGRESSION
        print(repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5, backend="native"))  # [[7.20435338 7.04106921 7.11815431 7.08556309 7.18165966 7.1394064  7.03340405 7.11273415 7.07614767 7.21791022]]
        print(repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name],
                                     ensemble_size=5, folds=[2], backend="native"))  # [[7.11815431]]
