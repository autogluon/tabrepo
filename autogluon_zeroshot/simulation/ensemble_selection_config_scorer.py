import numpy as np

from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

from .simulation_context import ZeroshotSimulatorContext
from ..utils.rank_utils import RankScorer


# FIXME: Add temperature scaling!!
class EnsembleSelectionConfigScorer:
    def __init__(self,
                 datasets: list,
                 zeroshot_gt: dict,
                 zeroshot_pred_proba: dict,
                 ranker: RankScorer,
                 dataset_name_to_tid_dict: dict,
                 dataset_name_to_fold_dict: dict,
                 ensemble_size=100,
                 ensemble_selection_kwargs=None):
        self.datasets = datasets
        self.zeroshot_gt = zeroshot_gt
        self.zeroshot_pred_proba = zeroshot_pred_proba
        self.ranker = ranker
        self.dataset_name_to_tid_dict = dataset_name_to_tid_dict
        self.dataset_name_to_fold_dict = dataset_name_to_fold_dict
        self.ensemble_size = ensemble_size
        if ensemble_selection_kwargs is None:
            ensemble_selection_kwargs = {}
        self.ensemble_selection_kwargs = ensemble_selection_kwargs

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

        pred_proba_dict_val = self.zeroshot_pred_proba[dataset][fold]['pred_proba_dict_val']
        pred_proba_dict_test = self.zeroshot_pred_proba[dataset][fold]['pred_proba_dict_test']
        weighted_ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            problem_type=problem_type,
            metric=eval_metric,
            **self.ensemble_selection_kwargs,
        )

        a = []
        for m in models:
            a.append(pred_proba_dict_val[m])
        weighted_ensemble.fit(predictions=a, labels=y_val)
        b = []
        for m in models:
            b.append(pred_proba_dict_test[m])
        y_test_pred = weighted_ensemble.predict_proba(b)
        y_test = y_test.fillna(-1)
        err = eval_metric._optimum - eval_metric(y_test, y_test_pred)  # FIXME: proba or pred, figure out

        # FIXME
        # y_val_pred = weighed_ensemble.predict_proba(a)
        # errval = eval_metric._optimum - eval_metric(y_val, y_val_pred)  # FIXME: proba or pred, figure out
        # print(dataset, errval)

        return err

    def compute_errors(self, configs: list):
        errors = {}
        for dataset in self.datasets:
            errors[dataset] = self.run_dataset(dataset=dataset, models=configs)
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
        )
