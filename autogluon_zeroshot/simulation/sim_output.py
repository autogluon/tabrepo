import copy
from typing import List

import numpy as np
import pandas as pd

from autogluon_zeroshot.portfolio import PortfolioCV
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer


# FIXME: Ideally this should be easier, but not really possible with current logic.
def metric_error_to_score(metric_error: float, metric: str):
    if metric.startswith('neg_'):
        metric_score = -metric_error
    elif metric == 'auc':
        metric_score = 1 - metric_error
    else:
        raise AssertionError(f'Unknown metric: {metric}')
    return metric_score


# FIXME: Make portfolio_cv its own class!
# FIXME: accurate time_infer_s
# FIXME: accurate score_val
class SimulationOutputGenerator:
    """
    Generate an output pandas DataFrame that is compatible with the AutoMLBenchmark results format.
    Useful to compare directly with AutoML frameworks and baselines.

    Given a portfolio, create an output DataFrame with the following columns:

    dataset:
        The original dataset name (as a string)
    fold:
        The fold (as an int)
    framework:
        Set to the user provided name for all rows.
        This is used to differentiate the portfolio from other portfolios.
    metric:
        Example: 'neg_rmse', 'auc', etc.
    metric_error:
        The metric error of the portfolio. Lower is better, 0 is perfect.
    metric_score:
        The metric score of the portfolio, defined identical to AutoMLBenchmark.
        Higher is better.
    problem_type
    score_val:
        # FIXME: Not correct currently, but not necessary for analysis purposes.
    tid
    time_infer_s:
        # FIXME: Not correct currently, instead figure out which configs were used in the ensemble.
    time_train_s:
        The sum of the training times of the individual models in the portfolio.

    """
    def __init__(self,
                 zsc,
                 zeroshot_gt,
                 zeroshot_pred_proba,
                 backend='ray'):
        self.zsc = zsc
        self.zeroshot_gt = zeroshot_gt
        self.zeroshot_pred_proba = zeroshot_pred_proba
        self.backend = backend

    def from_portfolio(self,
                       portfolio: List[str],
                       datasets: List[str],
                       name: str) -> pd.DataFrame:
        """
        Create from a single portfolio (Not cross-validated)
        """
        zeroshot_pred_proba = copy.deepcopy(self.zeroshot_pred_proba)

        zeroshot_pred_proba.restrict_models(portfolio)

        config_scorer_test = EnsembleSelectionConfigScorer.from_zsc(
            datasets=datasets,
            zeroshot_simulator_context=self.zsc,
            zeroshot_gt=self.zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=100,
            backend=self.backend,
        )

        df_raw_subset = self.zsc.df_raw[self.zsc.df_raw['tid_new'].isin(datasets)]
        df_raw_subset = df_raw_subset[df_raw_subset['model'].isin(portfolio)]
        df_total_train_and_infer_times = df_raw_subset[['tid_new', 'time_train_s', 'time_infer_s']].groupby('tid_new').sum()
        df_raw_subset = df_raw_subset.drop_duplicates(subset=['tid_new'])

        score_per_dataset = config_scorer_test.compute_errors(portfolio)

        df_raw_subset['metric_error'] = [score_per_dataset[row[0]] for row in zip(df_raw_subset['tid_new'])]
        df_raw_subset['metric_score'] = [metric_error_to_score(row[0], row[1]) for row in
                                         zip(df_raw_subset['metric_error'], df_raw_subset['metric'])]
        df_raw_subset['framework'] = name
        df_raw_subset = df_raw_subset.set_index('tid_new', drop=True)
        df_raw_subset['time_train_s'] = df_total_train_and_infer_times['time_train_s']

        # FIXME: time_infer_s is not correct since it assumes all models are used in final ensemble
        #  In reality the infer_speed is faster.
        df_raw_subset['time_infer_s'] = df_total_train_and_infer_times['time_infer_s']
        df_raw_subset = df_raw_subset.drop(columns=['model', 'framework_parent', 'constraint'])
        df_raw_subset = df_raw_subset.reset_index(drop=True)
        return df_raw_subset

    def from_portfolio_cv(self, portfolio_cv: PortfolioCV, name: str) -> pd.DataFrame:
        """
        Create from a cross-validated portfolio, using the results only from
        the holdout to construct an output that is not overfit.
        """
        assert portfolio_cv.are_test_folds_unique()

        portfolios = portfolio_cv.portfolios
        num_folds = len(portfolios)
        for i in range(len(portfolios)):
            print(f'Fold {portfolios[i].fold} Selected Configs: {portfolios[i].configs}')

        for i in range(len(portfolios)):
            print(f'Fold {portfolios[i].fold} Test Score: {portfolios[i].test_score}')

        print(f'Final Test Score: {portfolio_cv.get_test_score_overall()}')

        df_raw_all = []
        for f in range(num_folds):
            print(f'Computing scores for each dataset... (fold={f})')
            portfolio = portfolios[f]
            df_raw_subset = self.from_portfolio(portfolio=portfolio.configs,
                                                datasets=portfolio.test_datasets_fold,
                                                name=name)
            df_raw_all.append(df_raw_subset)
        df_raw_all = pd.concat(df_raw_all)
        return df_raw_all
