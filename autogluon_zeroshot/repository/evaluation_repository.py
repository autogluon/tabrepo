from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .repository import SimpleRepository
from ..portfolio import Portfolio, PortfolioCV
from ..simulation.configuration_list_scorer import ConfigurationListScorer
from ..simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from ..simulation.single_best_config_scorer import SingleBestConfigScorer
from ..simulation.sim_output import SimulationOutputGenerator
from ..simulation.sim_runner import run_zs_simulation


class EvaluationRepository(SimpleRepository):
    def evaluate_ensemble(
        self,
        dataset_names: List[str],
        config_names: List[str],
        ensemble_size: int,
        rank: bool = True,
        folds: Optional[List[int]] = None,
        backend: str = "ray",
    ) -> np.array:
        """
        :param dataset_names: list of dataset to compute errors on.
        :param config_names: list of config to consider for ensembling.
        :param ensemble_size: number of members to select with Caruana.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :return: 2D array of scores whose rows are datasets and columns are folds
        """
        if folds is None:
            folds = range(self.n_folds())
        dataset_fold_name = lambda dataset, fold: f"{self.dataset_to_taskid(dataset)}_{fold}"
        tasks = [
            dataset_fold_name(dataset, fold)
            for dataset in dataset_names
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            datasets=tasks,
            ensemble_size=ensemble_size,
            backend=backend,
        )
        if rank:
            dict_scores = scorer.score_per_dataset(config_names)
        else:
            dict_scores = scorer.compute_errors(configs=config_names)

        return np.array([[
                dict_scores[dataset_fold_name(dataset, fold)
            ] for fold in folds
        ] for dataset in dataset_names])

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

    # TODO: add saving support
    # TODO: add loading support
    def generate_output_from_portfolio_cv(self,
                                          portfolio_cv: PortfolioCV,
                                          name: str,
                                          config_scorer_type: str = 'ensemble',
                                          config_scorer_kwargs: dict = None) -> pd.DataFrame:
        sog = SimulationOutputGenerator(repo=self,
                                        config_scorer_type=config_scorer_type,
                                        config_scorer_kwargs=config_scorer_kwargs)
        df_result = sog.from_portfolio_cv(portfolio_cv=portfolio_cv, name=name)
        return df_result

    # TODO: add saving support
    # TODO: add loading support
    def generate_output_from_portfolio(self,
                                       portfolio: Union[Portfolio, List[str]],
                                       name: str,
                                       config_scorer_type: str = 'ensemble',
                                       config_scorer_kwargs: dict = None) -> pd.DataFrame:
        sog = SimulationOutputGenerator(repo=self,
                                        config_scorer_type=config_scorer_type,
                                        config_scorer_kwargs=config_scorer_kwargs)
        df_result = sog.from_portfolio(portfolio=portfolio, name=name)
        return df_result

    # TODO: add simulate_zeroshot_debug
    # TODO: add saving support
    # TODO: add loading support
    def simulate_zeroshot(self,
                          num_zeroshot: int = 10,
                          n_splits: int = 2,
                          backend: str = 'ray',
                          config_scorer_type: str = 'ensemble',
                          config_scorer_kwargs: dict = None) -> PortfolioCV:
        if config_scorer_kwargs is None:
            config_scorer_kwargs = {}
        if config_scorer_type == 'ensemble':
            config_scorer = self._construct_ensemble_selection_config_scorer(**config_scorer_kwargs)
        elif config_scorer_type == 'single':
            config_scorer = self._construct_single_best_config_scorer(**config_scorer_kwargs)
        else:
            raise ValueError(f'Unknown config_scorer_type: {config_scorer_type}')
        results_cv: PortfolioCV = run_zs_simulation(
            zsc=self._zeroshot_context,
            config_scorer=config_scorer,
            n_splits=n_splits,
            config_generator_kwargs={'num_zeroshot': num_zeroshot},
            backend=backend,
        )
        return results_cv


def load(version: str = None) -> EvaluationRepository:
    from autogluon_zeroshot.contexts import get_context
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = get_context(version).load(load_predictions=True, lazy_format=True)
    r = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )
    r = r.force_to_dense(verbose=True)
    return r
