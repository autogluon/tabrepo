from typing import List, Optional

import numpy as np
import ray

from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from .configuration_list_scorer import ConfigurationListScorer

from .simulation_context import ZeroshotSimulatorContext
from .simulation_context import TabularModelPredictions
from ..utils.rank_utils import RankScorer
from ..metrics import _fast_log_loss
from ..metrics import _fast_roc_auc


@ray.remote
def compute_error_ray(config_scorer, configs, dataset) -> float:
    error = config_scorer.run_dataset(dataset=dataset, models=configs)
    return error


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
                 ):
        """
        TODO: Add docstring
        :param use_fast_metrics: If True, will leverage optimized eval metrics to speed up config scoring.
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

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        return cls(
            ranker=zeroshot_simulator_context.rank_scorer_vs_automl,
            dataset_name_to_tid_dict=zeroshot_simulator_context.dataset_name_to_tid_dict,
            dataset_name_to_fold_dict=zeroshot_simulator_context.dataset_name_to_fold_dict,
            **kwargs,
        )

    def _get_fast_metric_if_exist(self,
                                  metric_name: str,
                                  problem_type: str,
                                  y_val: np.array,
                                  pred_val: np.array,
                                  y_test: np.array,
                                  pred_test: np.array,
                                  ):
        """
        # TODO: Add docstring
        # TODO: Consider making this more standardized.
        #  Currently fast_log_loss needs a bit of special preprocessing of the data and isn't a straightforward replace.
        # TODO: Add fast_roc_auc
        # TODO: Add fast_rmse
        """
        if metric_name == 'log_loss' and problem_type == 'multiclass':
            # TODO: Can be even faster if we transform pred_val and pred_test
            #  as a preprocessing step to TabularModelPredictions.
            #  This would avoid ever having to pay the preprocessing time cost, and would massively reduce memory usage.
            eval_metric = _fast_log_loss.fast_log_loss
            pred_val = _fast_log_loss.extract_true_class_prob_bulk(y_true=y_val, y_pred_bulk=pred_val)
            pred_test = _fast_log_loss.extract_true_class_prob_bulk(y_true=y_test, y_pred_bulk=pred_test)
        elif metric_name == 'roc_auc':
            y_val = y_val.astype(np.bool8)
            y_test = y_test.astype(np.bool8)
            pred_val = pred_val.astype(np.float32)
            pred_test = pred_test.astype(np.float32)
            eval_metric = _fast_roc_auc.fast_roc_auc_cpp
        else:
            eval_metric = get_metric(metric_name)
        return eval_metric, y_val, pred_val, y_test, pred_test

    def run_dataset(self, dataset, models):
        fold = self.dataset_name_to_fold_dict[dataset]
        dataset = self.dataset_name_to_tid_dict[dataset]

        problem_type = self.zeroshot_gt[dataset][fold]['problem_type']
        metric_name = self.zeroshot_gt[dataset][fold]['eval_metric']
        y_val = self.zeroshot_gt[dataset][fold]['y_val']
        y_test = self.zeroshot_gt[dataset][fold]['y_test']

        y_val = y_val.to_numpy()
        y_test = y_test.fillna(-1).to_numpy()

        pred_val, pred_test = self.zeroshot_pred_proba.predict(dataset=dataset, fold=fold, splits=['val', 'test'], models=models)

        if self.use_fast_metrics:
            eval_metric, y_val, pred_val, y_test, pred_test = self._get_fast_metric_if_exist(
                metric_name=metric_name,
                problem_type=problem_type,
                y_val=y_val,
                pred_val=pred_val,
                y_test=y_test,
                pred_test=pred_test,
            )
        else:
            eval_metric = get_metric(metric_name)

        weighted_ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            problem_type=problem_type,
            metric=eval_metric,
            **self.ensemble_selection_kwargs,
        )

        weighted_ensemble.fit(predictions=pred_val, labels=y_val)
        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_test)
        err = eval_metric.error(y_test, y_test_pred)

        # FIXME
        # y_val_pred = weighed_ensemble.predict_proba(a)
        # errval = eval_metric._optimum - eval_metric(y_val, y_val_pred)  # FIXME: proba or pred, figure out
        # print(dataset, errval)

        return err

    def compute_errors(self, configs: list):
        if self.backend == 'ray':
            return self.compute_errors_ray(configs=configs)
        errors = {}
        for dataset in self.datasets:
            fold = self.dataset_name_to_fold_dict[dataset]
            if self.max_fold and fold >= self.max_fold:
                continue
            errors[dataset] = self.run_dataset(dataset=dataset, models=configs)
        return errors

    # speedup can be obtained by only sending minimum zeroshot pred proba info for each task by using lazy format
    def compute_errors_ray(self, configs: list):
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
        errors_list = ray.get(results)
        errors = {self.datasets[i]: errors_list[i] for i in range(len(self.datasets))}
        return errors

    def compute_ranks(self, errors: dict):
        ranks = {}
        for dataset, error in errors.items():
            rank = self.ranker.rank(dataset, error)  # FIXME: Use score or error?
            ranks[dataset] = rank
        return ranks

    def compute_rank_mean(self, errors: dict):
        ranks = self.compute_ranks(errors=errors)
        average_rank = np.mean(list(ranks.values()))
        return average_rank

    def score(self, configs: list):
        errors = self.compute_errors(configs=configs)
        rank = self.compute_rank_mean(errors)
        return rank

    def score_per_dataset(self, configs: list):
        errors = self.compute_errors(configs=configs)
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
