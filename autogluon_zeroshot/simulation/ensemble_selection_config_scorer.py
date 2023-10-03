from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray

from autogluon.core.metrics import get_metric, Scorer
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from .configuration_list_scorer import ConfigurationListScorer

from .simulation_context import ZeroshotSimulatorContext
from .simulation_context import TabularModelPredictions, TabularPicklePredictions
from ..utils.rank_utils import RankScorer
from ..metrics import _fast_log_loss, _fast_roc_auc


@ray.remote
def compute_error_ray(config_scorer, configs: List[str], task: str) -> (float, dict):
    error, ensemble_weights = config_scorer.run_task(task=task, models=configs)
    return error, ensemble_weights


class TaskMetadata:
    def __init__(self, task_metadata: Dict[str, Any]):
        self.task_metadata = task_metadata

    def get_ground_truth(self):
        y_val = self.task_metadata["y_val"].to_numpy()
        y_test = self.task_metadata["y_test"].fillna(-1).to_numpy()
        return y_val, y_test

    def get_metric(self) -> Scorer:
        metric_name = self.task_metadata["eval_metric"]
        problem_type = self.task_metadata["problem_type"]
        return self._get_metric_from_name(metric_name=metric_name, problem_type=problem_type)

    def _get_metric_from_name(self, metric_name: str, problem_type: str) -> Scorer:
        return get_metric(metric=metric_name, problem_type=problem_type)

    @property
    def problem_type(self):
        return self.task_metadata["problem_type"]

    @property
    def metric_name(self):
        return self.task_metadata["eval_metric"]


class EnsembleScorer:
    def __init__(self,
                 zeroshot_pp: TabularModelPredictions | dict,
                 zeroshot_gt: Dict[str, Dict[int, Dict[str, Any]]],
                 ensemble_method: callable = EnsembleSelection,
                 ensemble_method_kwargs: dict = None,
                 proxy_fit_metric_map: dict = None,
                 use_fast_metrics: bool = True):
        if isinstance(zeroshot_pp, dict):
            zeroshot_pp = TabularPicklePredictions(pred_dict=zeroshot_pp)
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
        pred_val, pred_test = self.zeroshot_pp.predict(dataset=dataset, fold=fold, splits=['val', 'test'], models=models)
        return pred_val, pred_test

    def get_task_metadata(self, dataset: str, fold: int) -> TaskMetadata:
        return TaskMetadata(self.zeroshot_gt[dataset][fold])

    def run_task(self, dataset: str, fold: int, models: List[str]) -> Tuple[float, np.array]:
        task_metadata = self.get_task_metadata(dataset=dataset, fold=fold)

        metric_name = task_metadata.metric_name
        problem_type = task_metadata.problem_type

        y_val, y_test = task_metadata.get_ground_truth()
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
                 datasets: list,
                 zeroshot_gt: dict,
                 zeroshot_pred_proba: TabularModelPredictions,
                 ranker: RankScorer,
                 dataset_name_to_tid_dict: dict,
                 dataset_name_to_fold_dict: dict,
                 ensemble_size=100,
                 ensemble_selection_kwargs=None,
                 max_fold: Optional[float] = None,
                 backend: str = 'native',
                 use_fast_metrics: bool = True,
                 proxy_fit_metric_map: Optional[Union[dict, str]] = None,  # TODO: Add unit test
                 ):
        """
        TODO: Add docstring
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
        self.dataset_name_to_tid_dict = dataset_name_to_tid_dict
        self.dataset_name_to_fold_dict = dataset_name_to_fold_dict
        self.ensemble_size = ensemble_size
        if ensemble_selection_kwargs is None:
            ensemble_selection_kwargs = {}
        self.ensemble_selection_kwargs = ensemble_selection_kwargs
        self.max_fold = max_fold
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
            ensemble_method_kwargs=ensemble_selection_kwargs,
            proxy_fit_metric_map=proxy_fit_metric_map,
            use_fast_metrics=use_fast_metrics,
        )

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        if 'datasets' not in kwargs:
            kwargs['datasets'] = zeroshot_simulator_context.get_dataset_folds()
        return cls(
            ranker=zeroshot_simulator_context.rank_scorer_vs_automl,
            dataset_name_to_tid_dict=zeroshot_simulator_context.dataset_name_to_tid_dict,
            dataset_name_to_fold_dict=zeroshot_simulator_context.dataset_name_to_fold_dict,
            **kwargs,
        )

    def run_task(self, task: str, models: List[str]) -> Tuple[float, np.array]:
        fold = self.dataset_name_to_fold_dict[task]
        tid = self.dataset_name_to_tid_dict[task]
        return self.ensemble_scorer.run_task(dataset=tid, fold=fold, models=models)

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
            fold = self.dataset_name_to_fold_dict[task]
            if self.max_fold and fold >= self.max_fold:
                continue
            errors[task], ensemble_weights[task] = self.run_task(task=task, models=configs)
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
            dataset_name_to_tid_dict=self.dataset_name_to_tid_dict,
            dataset_name_to_fold_dict=self.dataset_name_to_fold_dict,
            ensemble_size=self.ensemble_size,
            ensemble_selection_kwargs=self.ensemble_selection_kwargs,
            max_fold=self.max_fold,
        )
