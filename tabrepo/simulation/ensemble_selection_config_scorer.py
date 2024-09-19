from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING

import numpy as np
import ray

from autogluon.core.metrics import get_metric, Scorer
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from .configuration_list_scorer import ConfigurationListScorer

from ..utils.rank_utils import RankScorer
from ..utils import task_to_tid_fold
from ..metrics import _fast_log_loss, _fast_roc_auc

if TYPE_CHECKING:
    from ..repository.evaluation_repository import EvaluationRepository


@ray.remote
def compute_error_ray(config_scorer, configs: List[str], task: str) -> (float, dict):
    error, ensemble_weights = config_scorer.evaluate_task(task=task, models=configs)
    return error, ensemble_weights


class EnsembleScorer:
    def __init__(self,
                 repo: "EvaluationRepository",
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
        self.repo = repo
        self.ensemble_method: callable = ensemble_method
        self.ensemble_method_kwargs = ensemble_method_kwargs
        self.task_metrics_metadata = task_metrics_metadata
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
        pred_val = self.repo.predict_val_multi(dataset=dataset, fold=fold, configs=models)
        pred_test = self.repo.predict_test_multi(dataset=dataset, fold=fold, configs=models)
        return pred_val, pred_test

    def filter_models(self, dataset: str, fold: int, models: List[str]) -> List[str]:
        """
        Filters models by user-defined logic. Used in class extensions.
        """
        return models

    def evaluate_task(self, dataset: str, fold: int, models: List[str]) -> Tuple[float, np.array]:
        n_models = len(models)
        task_metadata = self.task_metrics_metadata[dataset]
        metric_name = task_metadata["metric"]
        problem_type = task_metadata["problem_type"]

        y_val = self.repo.labels_val(dataset=dataset, fold=fold)
        y_test = self.repo.labels_test(dataset=dataset, fold=fold)

        # If filtering models, need to keep track of original model order to return ensemble weights list
        models_filtered = self.filter_models(dataset=dataset, fold=fold, models=models)
        models, models_filtered_idx = self._get_models_filtered_idx(models=models, models_filtered=models_filtered)

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

        if hasattr(fit_eval_metric, 'post_problem_type'):
            fit_problem_type = fit_eval_metric.post_problem_type
        else:
            fit_problem_type = problem_type

        weighted_ensemble = self.ensemble_method(
            problem_type=fit_problem_type,
            metric=fit_eval_metric,
            **self.ensemble_method_kwargs,
        )

        weighted_ensemble.fit(predictions=pred_val, labels=y_val)

        if hasattr(eval_metric, 'preprocess_bulk'):
            y_test, pred_test = eval_metric.preprocess_bulk(y_test, pred_test)

        if hasattr(eval_metric, 'post_problem_type'):
            predict_problem_type = eval_metric.post_problem_type
        else:
            predict_problem_type = problem_type
        weighted_ensemble.problem_type = predict_problem_type

        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_test)
        err = eval_metric.error(y_test, y_test_pred)

        ensemble_weights: np.array = weighted_ensemble.weights_

        # ensemble_weights has to be updated, need to be in the original models order
        ensemble_weights_fixed = np.zeros(n_models, dtype=np.float64)
        ensemble_weights_fixed[models_filtered_idx] = ensemble_weights
        ensemble_weights = ensemble_weights_fixed

        return err, ensemble_weights

    def _get_models_filtered_idx(self, models: list[str], models_filtered: list[str]) -> Tuple[list[str], list[int]]:
        """
        Returns the filtered list of models and the index mapping of the filtered models to the original `models` list.
        """
        models_filtered_set = set(models_filtered)

        # Preserve `models` order without duplicates (optimized)
        models_seen = set()
        # not (m in models_seen or models_seen.add(m) only adds `m` to models_seen if `m` was not already in models_seen.
        models_filtered = [m for m in models if (m in models_filtered_set) and not (m in models_seen or models_seen.add(m))]

        if len(models_filtered_set) < len(models_filtered):
            # Duplicate names in `models`, have special handling
            models_idx = {}
            for i, m in enumerate(models):
                if m not in models_idx:
                    models_idx[m] = []
                models_idx[m].append(i)
            models_filtered_idx = [models_idx[m].pop(0) for m in models_filtered]
        else:
            models_filtered_idx = [models.index(m) for m in models_filtered]
        return models_filtered, models_filtered_idx


class EnsembleScorerMaxModels(EnsembleScorer):
    """
    Identical to EnsembleScorer, with the addition of `max_models` and `max_models_per_type`.

    Parameters
    ----------
    max_models: int, default = None
        If specified, will limit ensemble candidates to the top `max_models` highest validation score models.
        This logic is applied after the filtering from `max_models_per_type`.
    max_models_per_type: int | str, default = None
        If specified, will limit ensemble candidates of a given model type to the top `max_models_per_type` highest validation score models.
        If "auto", scales dynamically with the number of rows in the dataset.
    """
    def __init__(self, repo: "EvaluationRepository", max_models: int = None, max_models_per_type: int | str = None, **kwargs):
        super().__init__(repo=repo, **kwargs)
        assert self.repo is not None
        if max_models is not None:
            assert max_models >= 0
        if max_models_per_type is not None:
            if isinstance(max_models_per_type, str):
                assert max_models_per_type == "auto"
            else:
                assert max_models_per_type >= 0
        self.max_models = max_models
        self.max_models_per_type = max_models_per_type

    def filter_models(self, dataset: str, fold: int, models: List[str]) -> List[str]:
        """
        Filters models by user-defined logic. Used in class extensions.
        """
        if self.max_models is not None or self.max_models_per_type is not None:
            if self.max_models_per_type is not None and isinstance(self.max_models_per_type, str) and self.max_models_per_type == "auto":
                max_models_per_type = self._get_max_models_per_type_auto(dataset=dataset)
            else:
                max_models_per_type = self.max_models_per_type
            models = self.repo._zeroshot_context.get_top_configs(
                dataset=dataset,
                fold=fold,
                configs=models,
                max_models=self.max_models,
                max_models_per_type=max_models_per_type,
            )
        return models

    def _get_max_models_per_type_auto(self, dataset: str) -> int:
        """
        Logic to mimic AutoGluon's default setting for `max_models_per_type`.
        """
        # TODO: Make it easier to get this info without accessing private variables in repo
        df_metadata = self.repo._zeroshot_context.df_metadata
        num_rows = int(df_metadata[df_metadata["dataset"] == dataset].iloc[0]["NumberOfInstances"] * 9 / 10)
        if num_rows < 1000:
            max_models_per_type = 1
        elif num_rows < 5000:
            max_models_per_type = 2
        elif num_rows < 10000:
            max_models_per_type = 3
        elif num_rows < 15000:
            max_models_per_type = 4
        elif num_rows < 20000:
            max_models_per_type = 5
        elif num_rows < 25000:
            max_models_per_type = 6
        elif num_rows < 30000:
            max_models_per_type = 7
        elif num_rows < 35000:
            max_models_per_type = 8
        elif num_rows < 40000:
            max_models_per_type = 9
        elif num_rows < 45000:
            max_models_per_type = 10
        elif num_rows < 50000:
            max_models_per_type = 11
        else:
            max_models_per_type = 12
        return max_models_per_type


# FIXME: Add temperature scaling!!
class EnsembleSelectionConfigScorer(ConfigurationListScorer):
    def __init__(self,
                 tasks: List[str],
                 repo: "EvaluationRepository",
                 ranker: RankScorer,
                 tid_to_dataset_name_dict: Dict[int, str],
                 task_metrics_metadata: Dict[int, Dict[str, str]],
                 ensemble_size=100,
                 ensemble_selection_kwargs=None,
                 backend: str = 'native',
                 use_fast_metrics: bool = True,
                 proxy_fit_metric_map: Optional[Union[dict, str]] = None,  # TODO: Add unit test
                 ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
                 ensemble_kwargs: dict = None,
                 ):
        """
        A scorer object to evaluate configs via simulating ensemble selection.

        :param tasks: The list of tasks to consider for scoring.
        :param ranker: The ranking object used to compute scores on each task.
        :param task_metrics_metadata: dictionary containing metric information and problem type for all tasks
        :param ensemble_size: The maximum ensemble selection iterations when fitting the ensemble. TODO: Remove?
        :param ensemble_selection_kwargs: kwargs to pass to the init of the ensemble selection model.
        :param backend: Options include ["native", "ray"].
        :param use_fast_metrics: If True, will leverage optimized eval metrics to speed up config scoring.
        :param proxy_fit_metric_map:
            If eval_metric is among the keys in the `proxy_fit_metric_map` dictionary,
            the value eval_metric will be used during the weighted ensemble fitting process as a proxy.
            For example, the proxy metric could be faster to compute while producing a similar end result.
            If None: Do not use proxy metrics, equivalent to {}.
            If 'roc_auc_to_log_loss': set to {'roc_auc': 'log_loss'}, making 'log_loss' a proxy to 'roc_auc'
        :param: ensemble_cls: The ensemble class to use for fitting and scoring.
        :param: ensemble_kwargs: The kwargs to pass to the init call of `ensemble_cls`.
        """
        super().__init__(tasks=tasks)
        if ensemble_kwargs is None:
            ensemble_kwargs = {}
        self.repo = repo
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

        self.ensemble_scorer = ensemble_cls(
            repo=repo,
            task_metrics_metadata=task_metrics_metadata,
            ensemble_method_kwargs=ensemble_selection_kwargs,
            proxy_fit_metric_map=proxy_fit_metric_map,
            use_fast_metrics=use_fast_metrics,
            **ensemble_kwargs,
        )

    @classmethod
    def from_repo(cls, repo: "EvaluationRepository", **kwargs):
        zeroshot_simulator_context = repo._zeroshot_context
        if 'tasks' not in kwargs:
            kwargs['tasks'] = zeroshot_simulator_context.get_tasks()

        dataset_to_tid_dict = zeroshot_simulator_context.dataset_to_tid_dict
        task_metrics_metadata = zeroshot_simulator_context.df_metrics
        task_metrics_metadata = {
            dataset: task_metrics_metadata.loc[dataset].to_dict()
            for dataset in task_metrics_metadata.index if dataset in dataset_to_tid_dict
        }

        return cls(
            repo=repo,
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
        for task in self.tasks:
            errors[task], ensemble_weights[task] = self.evaluate_task(task=task, models=configs)
        return errors, ensemble_weights

    # speedup can be obtained by only sending minimum zeroshot pred proba info for each task by using lazy format
    def compute_errors_ray(self, configs: List[str]) -> Tuple[Dict[str, float], Dict[str, np.array]]:
        # Create and execute all tasks in parallel
        if not ray.is_initialized():
            ray.init()
        config_scorer = ray.put(self)
        results = []
        for i in range(len(self.tasks)):
            results.append(compute_error_ray.remote(
                config_scorer,
                configs,
                self.tasks[i],
            ))
        results_list = ray.get(results)
        errors_list = [r[0] for r in results_list]
        ensemble_weights_list = [r[1] for r in results_list]
        errors = {self.tasks[i]: errors_list[i] for i in range(len(self.tasks))}
        ensemble_weights = {self.tasks[i]: ensemble_weights_list[i] for i in range(len(self.tasks))}
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

    def subset(self, tasks):
        return self.__class__(
            tasks=tasks,
            repo=self.repo,
            ranker=self.ranker,
            ensemble_size=self.ensemble_size,
            ensemble_selection_kwargs=self.ensemble_selection_kwargs,
            tid_to_dataset_name_dict=self.tid_to_dataset_name_dict,
            task_metrics_metadata=self.ensemble_scorer.task_metrics_metadata,
        )
