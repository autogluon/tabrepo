from typing import Optional

import numpy as np
import ray

from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from .configuration_list_scorer import ConfigurationListScorer

from .simulation_context import ZeroshotSimulatorContext
from .simulation_context import TabularModelPredictions
from ..utils.rank_utils import RankScorer


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
                 ):
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

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        return cls(
            ranker=zeroshot_simulator_context.rank_scorer_vs_automl,
            dataset_name_to_tid_dict=zeroshot_simulator_context.dataset_name_to_tid_dict,
            dataset_name_to_fold_dict=zeroshot_simulator_context.dataset_name_to_fold_dict,
            **kwargs,
        )

    def run_dataset(self, dataset, models):
        fold = self.dataset_name_to_fold_dict[dataset]
        dataset = self.dataset_name_to_tid_dict[dataset]

        problem_type = self.zeroshot_gt[dataset][fold]['problem_type']
        metric_name = self.zeroshot_gt[dataset][fold]['eval_metric']
        eval_metric = get_metric(metric_name)
        y_val = self.zeroshot_gt[dataset][fold]['y_val']
        y_test = self.zeroshot_gt[dataset][fold]['y_test']

        pred_proba_dict_val, pred_proba_dict_test = self.zeroshot_pred_proba.predict(dataset=dataset, fold=fold, splits=['val', 'test'], models=models)
        weighted_ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            problem_type=problem_type,
            metric=eval_metric,
            **self.ensemble_selection_kwargs,
        )

        weighted_ensemble.fit(predictions=pred_proba_dict_val, labels=y_val)
        if eval_metric.needs_pred:
            y_test_pred = weighted_ensemble.predict(pred_proba_dict_test)
        else:
            y_test_pred = weighted_ensemble.predict_proba(pred_proba_dict_test)
        y_test = y_test.fillna(-1)
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
