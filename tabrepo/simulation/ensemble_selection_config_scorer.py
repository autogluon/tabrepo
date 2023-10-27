from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray

from autogluon.core.metrics import get_metric, Scorer
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from .configuration_list_scorer import ConfigurationListScorer
from .ground_truth import GroundTruth

from .simulation_context import ZeroshotSimulatorContext
from .simulation_context import TabularModelPredictions
from ..utils.rank_utils import RankScorer
from ..utils import task_to_tid_fold
from ..metrics import _fast_log_loss, _fast_roc_auc


@ray.remote
def compute_error_ray(config_scorer, configs: List[str], task: str) -> (float, dict):
    error, ensemble_weights = config_scorer.evaluate_task(task=task, models=configs)
    return error, ensemble_weights

class EnsembleScorer:
    def __init__(self,
                 zeroshot_pp: TabularModelPredictions,
                 tid_to_dataset_dict: dict,
                 zeroshot_gt: GroundTruth,
                 task_metrics_metadata,
                 ensemble_method: callable = EnsembleSelection,
                 ensemble_method_kwargs: dict = None,
                 proxy_fit_metric_map: dict = None,
                 use_fast_metrics: bool = True,
                 ):
        if proxy_fit_metric_map is None:
            proxy_fit_metric_map = dict()
        if ensemble_method_kwargs is None:
            ensemble_method_kwargs = dict()
        ensemble_method_kwargs = copy.deepcopy(ensemble_method_kwargs)
        if "ensemble_size" not in ensemble_method_kwargs:
            ensemble_method_kwargs["ensemble_size"] = 100
        self.ensemble_method: callable = ensemble_method
        self.ensemble_method_kwargs = ensemble_method_kwargs
        self.zeroshot_pp: TabularModelPredictions = zeroshot_pp
        self.zeroshot_gt = zeroshot_gt
        self.task_metrics_metadata = task_metrics_metadata
        self.tid_to_dataset_dict = tid_to_dataset_dict
        self.proxy_fit_metric_map = proxy_fit_metric_map
        self.use_fast_metrics = use_fast_metrics

    def _get_metric_from_name(self, metric_name: str, problem_type: str) -> Scorer:
        if self.use_fast_metrics:
            return self._get_fast_metric_if_exist(metric_name=metric_name, problem_type=problem_type)
        else:
            return get_metric(metric=metric_name, problem_type=problem_type)

    def _get_fast_metric_if_exist(self, metric_name: str, problem_type: str) -> Scorer:
        """
        # TODO: Add docstring
        # TODO: Consider making this more standardized.
        #  Currently fast_log_loss needs a bit of special preprocessing of the data and isn't a straightforward replace.
        """
        if metric_name == 'log_loss':
            # TODO: Can be even faster if we transform pred_val and pred_test
            #  as a preprocessing step to TabularModelPredictions.
            #  This would avoid ever having to pay the preprocessing time cost, and would massively reduce memory usage.
            eval_metric = _fast_log_loss.fast_log_loss
        elif metric_name == 'roc_auc':
            eval_metric = _fast_roc_auc.fast_roc_auc_cpp
        else:
            eval_metric = get_metric(metric=metric_name, problem_type=problem_type)
        return eval_metric

    def get_preds_from_models(self, dataset: str, fold: int, models: List[str]):
        pred_val = self.zeroshot_pp.predict_val(dataset=dataset, fold=fold, models=models)
        pred_test = self.zeroshot_pp.predict_test(dataset=dataset, fold=fold, models=models)
        return pred_val, pred_test

    def evaluate_task(self, dataset: str, fold: int, models: List[str]) -> Tuple[float, np.array]:
        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        y_val = self.zeroshot_gt.labels_val(dataset=dataset, fold=fold)
        y_test = self.zeroshot_gt.labels_test(dataset=dataset, fold=fold)

        pred_val, pred_test = self.get_preds_from_models(dataset=dataset, fold=fold, models=models)

        if problem_type == 'binary':
            # Force binary prediction probabilities to 1 dimensional prediction probabilites of the positive class
            # if it is in multiclass format
            if len(pred_val.shape) == 3:
                pred_val = pred_val[:, :, 1]
            if len(pred_test.shape) == 3:
                pred_test = pred_test[:, :, 1]

        fit_metric_name = self.proxy_fit_metric_map.get(metric_name, metric_name)

        eval_metric = self._get_metric_from_name(metric_name=metric_name, problem_type=problem_type)
        fit_eval_metric = self._get_metric_from_name(metric_name=fit_metric_name, problem_type=problem_type)

        if hasattr(fit_eval_metric, 'preprocess_bulk'):
            y_val, pred_val = fit_eval_metric.preprocess_bulk(y_val, pred_val)

        weighted_ensemble = self.ensemble_method(
            problem_type=problem_type,
            metric=fit_eval_metric,
            **self.ensemble_method_kwargs,
        )

        weighted_ensemble.fit(predictions=pred_val, labels=y_val)

        if hasattr(eval_metric, 'preprocess_bulk'):
            y_test, pred_test = eval_metric.preprocess_bulk(y_test, pred_test)

        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_test)
        err = eval_metric.error(y_test, y_test_pred)

        ensemble_weights: np.array = weighted_ensemble.weights_

        return err, ensemble_weights


# FIXME: Add temperature scaling!!
class EnsembleSelectionConfigScorer(ConfigurationListScorer):
    def __init__(self,
                 datasets: List[str],
                 zeroshot_gt: Dict[str, Dict[int, Dict[str, Any]]],
                 zeroshot_pred_proba: TabularModelPredictions,
                 ranker: RankScorer,
                 tid_to_dataset_name_dict: Dict[int, str],
                 task_metrics_metadata: Dict[int, Dict[str, str]],
                 ensemble_size=100,
                 ensemble_selection_kwargs=None,
                 backend: str = 'native',
                 use_fast_metrics: bool = True,
                 proxy_fit_metric_map: Optional[Union[dict, str]] = None,  # TODO: Add unit test
                 ):
        """
        A scorer object to evaluate configs via simulating ensemble selection.

        :param datasets: The list of datasets to consider for scoring. TODO: Convert to tasks?
        :param zeroshot_gt: The ground truth information and task metadata for all tasks.
        :param zeroshot_pred_proba: The TabularModelPredictions object that contains the predictions of all configs on all tasks.
        :param ranker: The ranking object used to compute scores on each task.
        :param dataset_name_to_tid_dict: Mapping of dataset names to corresponding TIDs. TODO: Remove?
        :param dataset_name_to_fold_dict: Mapping of dataset names to available folds. TODO: Remove?
        :param task_metrics_metadata: dictionary containing metric information and problem type for all tasks
        :param ensemble_size: The maximum ensemble selection iterations when fitting the ensemble. TODO: Remove?
        :param ensemble_selection_kwargs: kwargs to pass to the init of the ensemble selection model.
        :param max_fold: The maximum number of folds to consider for each dataset. TODO: Remove?
        :param backend: Options include ["native", "ray"].
        :param use_fast_metrics: If True, will leverage optimized eval metrics to speed up config scoring.
        :param proxy_fit_metric_map:
            If eval_metric is among the keys in the `proxy_fit_metric_map` dictionary,
            the value eval_metric will be used during the weighted ensemble fitting process as a proxy.
            For example, the proxy metric could be faster to compute while producing a similar end result.
            If None: Do not use proxy metrics, equivalent to {}.
            If 'roc_auc_to_log_loss': set to {'roc_auc': 'log_loss'}, making 'log_loss' a proxy to 'roc_auc'
        """
        super(EnsembleSelectionConfigScorer, self).__init__(datasets=datasets)
        if zeroshot_gt is None:
            raise ValueError(f'zeroshot_gt cannot be None!')
        if zeroshot_pred_proba is None:
            raise ValueError(f'zeroshot_pred_proba cannot be None!')
        self.zeroshot_gt = zeroshot_gt
        self.zeroshot_pred_proba = zeroshot_pred_proba
        self.ranker = ranker
        self.tid_to_dataset_name_dict = tid_to_dataset_name_dict
        self.ensemble_size = ensemble_size
        if ensemble_selection_kwargs is None:
            ensemble_selection_kwargs = {}
        self.ensemble_selection_kwargs = ensemble_selection_kwargs
        assert backend in ['native', 'ray']
        self.backend = backend
        self.use_fast_metrics = use_fast_metrics
        if proxy_fit_metric_map is None:
            proxy_fit_metric_map = {}
        elif isinstance(proxy_fit_metric_map, str):
            assert proxy_fit_metric_map == 'roc_auc_to_log_loss'
            proxy_fit_metric_map = {'roc_auc': 'log_loss'}  # log_loss is fast to compute and a good proxy for roc_auc
        self.proxy_fit_metric_map = proxy_fit_metric_map

        ensemble_selection_kwargs = copy.deepcopy(ensemble_selection_kwargs)
        ensemble_selection_kwargs["ensemble_size"] = ensemble_size

        self.ensemble_scorer = EnsembleScorer(
            zeroshot_pp=zeroshot_pred_proba,
            zeroshot_gt=zeroshot_gt,
            task_metrics_metadata=task_metrics_metadata,
            ensemble_method_kwargs=ensemble_selection_kwargs,
            proxy_fit_metric_map=proxy_fit_metric_map,
            use_fast_metrics=use_fast_metrics,
            tid_to_dataset_dict=tid_to_dataset_name_dict
        )

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        if 'datasets' not in kwargs:
            kwargs['datasets'] = zeroshot_simulator_context.get_tasks()

        dataset_to_tid_dict = zeroshot_simulator_context.dataset_to_tid_dict
        task_metrics_metadata = zeroshot_simulator_context.df_metrics
        task_metrics_metadata = {
            dataset: task_metrics_metadata.loc[dataset].to_dict()
            for dataset in task_metrics_metadata.index if dataset in dataset_to_tid_dict
        }

        return cls(
            ranker=zeroshot_simulator_context.rank_scorer,
            tid_to_dataset_name_dict=zeroshot_simulator_context.tid_to_dataset_dict,
            task_metrics_metadata=task_metrics_metadata,
            **kwargs,
        )

    def evaluate_task(self, task: str, models: List[str]) -> Tuple[float, np.array]:
        tid, fold = task_to_tid_fold(task=task)
        dataset = self.tid_to_dataset_name_dict[tid]
        return self.ensemble_scorer.evaluate_task(dataset=dataset, fold=fold, models=models)

    def compute_errors(self, configs: List[str]) -> Tuple[Dict[str, float], Dict[str, np.array]]:
        """
        Compute and return test errors and ensemble weights for all tasks on the user-specified list of configs.

        :param configs: List of model config names to ensemble and compute test errors with.
        :return: Tuple:
            Dictionary of task_name -> test evaluation metric error of the ensemble.
            Dictionary of task_name -> model weights in the ensemble. Model weights are stored in a numpy array,
                with weights corresponding to the order of `configs`.
        """
        if self.backend == 'ray':
            return self.compute_errors_ray(configs=configs)
        errors = dict()
        ensemble_weights = dict()
        for task in self.datasets:
            errors[task], ensemble_weights[task] = self.evaluate_task(task=task, models=configs)
        return errors, ensemble_weights

    # speedup can be obtained by only sending minimum zeroshot pred proba info for each task by using lazy format
    def compute_errors_ray(self, configs: List[str]) -> Tuple[Dict[str, float], Dict[str, np.array]]:
        # Create and execute all tasks in parallel
        if not ray.is_initialized():
            ray.init()
        config_scorer = ray.put(self)
        results = []
        for i in range(len(self.datasets)):
            results.append(compute_error_ray.remote(
                config_scorer,
                configs,
                self.datasets[i],
            ))
        results_list = ray.get(results)
        errors_list = [r[0] for r in results_list]
        ensemble_weights_list = [r[1] for r in results_list]
        errors = {self.datasets[i]: errors_list[i] for i in range(len(self.datasets))}
        ensemble_weights = {self.datasets[i]: ensemble_weights_list[i] for i in range(len(self.datasets))}
        return errors, ensemble_weights

    def compute_ranks(self, errors: Dict[str, float]) -> Dict[str, float]:
        ranks = {}
        for dataset, error in errors.items():
            rank = self.ranker.rank(dataset, error)  # FIXME: Use score or error?
            ranks[dataset] = rank
        return ranks

    def compute_rank_mean(self, errors: Dict[str, float]) -> float:
        ranks = self.compute_ranks(errors=errors)
        average_rank = np.mean(list(ranks.values()))
        return average_rank

    def score(self, configs: List[str]) -> float:
        errors, ensemble_weights = self.compute_errors(configs=configs)
        rank = self.compute_rank_mean(errors)
        return rank

    def score_per_dataset(self, configs: List[str]) -> Dict[str, float]:
        errors, ensemble_weights = self.compute_errors(configs=configs)
        return self.compute_ranks(errors=errors)

    def subset(self, datasets):
        return self.__class__(
            datasets=datasets,
            zeroshot_gt=self.zeroshot_gt,
            zeroshot_pred_proba=self.zeroshot_pred_proba,
            ranker=self.ranker,
            ensemble_size=self.ensemble_size,
            ensemble_selection_kwargs=self.ensemble_selection_kwargs,
            tid_to_dataset_name_dict=self.tid_to_dataset_name_dict,
            task_metrics_metadata=self.ensemble_scorer.task_metrics_metadata,
        )
