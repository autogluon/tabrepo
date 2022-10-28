import numpy as np

from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

from autogluon_zeroshot.utils.rank_utils import RankScorer


# FIXME: Add temperature scaling!!
class EnsembleSelectionConfigScorer:
    def __init__(self,
                 datasets: list,
                 folds: list,
                 zeroshot_gt: dict,
                 zeroshot_pred_proba: dict,
                 ranker: RankScorer,
                 ensemble_size=100,
                 ensemble_selection_kwargs=None):
        self.datasets = datasets
        self.folds = folds
        self.zeroshot_gt = zeroshot_gt
        self.zeroshot_pred_proba = zeroshot_pred_proba
        self.ranker = ranker
        self.ensemble_size = ensemble_size
        if ensemble_selection_kwargs is None:
            ensemble_selection_kwargs = {}
        self.ensemble_selection_kwargs = ensemble_selection_kwargs

    def run_dataset(self, dataset, fold, models):
        problem_type = self.zeroshot_gt[dataset][fold]['problem_type']
        metric_name = self.zeroshot_gt[dataset][fold]['eval_metric']
        eval_metric = get_metric(metric_name)
        y_val = self.zeroshot_gt[dataset][fold]['y_val']
        y_test = self.zeroshot_gt[dataset][fold]['y_test']

        pred_proba_dict_val = self.zeroshot_pred_proba[dataset][fold]['pred_proba_dict_val']
        pred_proba_dict_test = self.zeroshot_pred_proba[dataset][fold]['pred_proba_dict_test']
        weighed_ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            problem_type=problem_type,
            metric=eval_metric,
            **self.ensemble_selection_kwargs
        )

        a = []
        for m in models:
            a.append(pred_proba_dict_val[m])
        weighed_ensemble.fit(predictions=a, labels=y_val)
        b = []
        for m in models:
            b.append(pred_proba_dict_test[m])
        y_test_pred = weighed_ensemble.predict_proba(b)
        y_test = y_test.fillna(-1)
        err = eval_metric._optimum - eval_metric(y_test, y_test_pred)  # FIXME: proba or pred, figure out

        # FIXME
        # y_val_pred = weighed_ensemble.predict_proba(a)
        # errval = eval_metric._optimum - eval_metric(y_val, y_val_pred)  # FIXME: proba or pred, figure out
        # print(dataset, errval)

        return err

    def compute_errors(self, configs: list):
        # FIXME: Use Fold
        if len(self.folds) != 1:
            raise AssertionError('self.folds not implemented for multifold yet!')
        errors = {}
        for dataset in self.datasets:
            for fold in self.folds:
                err = self.run_dataset(dataset, fold, configs)
                errors[dataset] = err
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
            folds=self.folds,
            zeroshot_gt=self.zeroshot_gt,
            zeroshot_pred_proba=self.zeroshot_pred_proba,
            ranker=self.ranker,
            ensemble_size=self.ensemble_size,
            ensemble_selection_kwargs=self.ensemble_selection_kwargs,
        )
