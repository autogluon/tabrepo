from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl

from typing import List
from autogluon_zeroshot.simulation.configuration_list_scorer import ConfigurationListScorer
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext
from autogluon_zeroshot.simulation.tabular_predictions import TabularModelPredictions
from autogluon_zeroshot.utils import catchtime
from autogluon_zeroshot.contexts.utils import intersect_folds_and_datasets, force_to_dense, prune_zeroshot_gt
from autogluon_zeroshot.portfolio import Portfolio, PortfolioCV
from autogluon_zeroshot.simulation.sim_runner import run_zs_simulation
from autogluon_zeroshot.simulation.sim_output import SimulationOutputGenerator


class EvaluationRepository:
    def __init__(
            self,
            zeroshot_context: ZeroshotSimulatorContext,
            tabular_predictions: TabularModelPredictions,
            ground_truth: dict,
    ):
        self._tabular_predictions = tabular_predictions
        self._zeroshot_context = zeroshot_context
        self._ground_truth = ground_truth
        assert all(x in self._tid_to_name for x in self._tabular_predictions.datasets)

    def print_info(self):
        self._zeroshot_context.print_info()

    @property
    def _name_to_tid(self):
        return self._zeroshot_context.dataset_to_tid_dict

    @property
    def _tid_to_name(self):
        return {v: k for k, v in self._name_to_tid.items()}

    def save(self, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        save_pkl.save(path=path, object=self)

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        obj = load_pkl.load(path=path)
        assert isinstance(obj, cls)
        return obj

    def subset(self,
               folds: List[int] = None,
               models: List[str] = None,
               datasets: List[Union[str, int]] = None,
               verbose: bool = True,
               ):
        if folds:
            self._zeroshot_context.subset_folds(folds=folds)
        if models:
            self._zeroshot_context.subset_models(models=models)
        if datasets:
            self._zeroshot_context.subset_datasets(datasets=datasets)
        # TODO:
        # if problem_type:
        #     self._zeroshot_context.subset_problem_type(problem_type=problem_type)
        self.force_to_dense(verbose=verbose)
        return self

    def force_to_dense(self, verbose: bool = True):
        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(self._zeroshot_context, self._tabular_predictions, self._ground_truth)
        force_to_dense(self._tabular_predictions,
                       first_prune_method='task',
                       second_prune_method='dataset',
                       verbose=verbose)

        self._zeroshot_context.subset_models(self._tabular_predictions.models)
        self._zeroshot_context.subset_datasets(self._tabular_predictions.datasets)
        self._tabular_predictions.restrict_models(self._zeroshot_context.get_configs())
        self._ground_truth = prune_zeroshot_gt(zeroshot_pred_proba=self._tabular_predictions,
                                               zeroshot_gt=self._ground_truth,
                                               verbose=verbose)
        return self

    @property
    def _df_metadata(self):
        return self._zeroshot_context.df_metadata

    def dataset_names(self) -> List[str]:
        return list(sorted([self._tid_to_name[task_id] for task_id in self._tabular_predictions.datasets]))

    def task_ids(self):
        return list(sorted(self._name_to_tid.values()))

    def list_models_available(self, dataset_name: str):
        # TODO rename with new name, and keep naming convention of tabular_predictions to allow filtering over folds,
        #  datasets, specify whether all need to be present etc
        """
        :param dataset_name:
        :return: the list of configurations that are available on all folds of the given dataset.
        """
        task_id = self._name_to_tid[dataset_name]
        res = set(self._tabular_predictions.list_models_available(datasets=[task_id]))
        for fold in range(self.n_folds()):
            df = self._zeroshot_context.df_results_by_dataset_vs_automl
            methods = set(df.loc[df.dataset == f"{self.dataset_to_taskid(dataset_name)}_{fold}", "framework"].unique())
            res = res.intersection(methods)
        return list(sorted(res))

    def dataset_to_taskid(self, dataset_name: str) -> int:
        return self._name_to_tid[dataset_name]

    def taskid_to_dataset(self, taskid: int) -> str:
        return self._tid_to_name.get(taskid, "Not found")

    def eval_metrics(self, dataset_name: str, config_names: List[str], fold: int, check_all_found: bool = True) -> List[dict]:
        """
        :param dataset_name:
        :param config_names: list of configs to query metrics
        :param fold:
        :return: list of metrics for each configuration
        """
        df = self._zeroshot_context.df_results_by_dataset_vs_automl
        mask = (df.dataset == f"{self.dataset_to_taskid(dataset_name)}_{fold}") & (df.framework.isin(config_names))
        output_cols = ["framework", "time_train_s", "metric_error", "time_infer_s", "bestdiff", "loss_rescaled",
                       "time_train_s_rescaled", "time_infer_s_rescaled", "rank", "score_val"]
        if check_all_found:
            assert sum(mask) == len(config_names), \
                f"expected one evaluation occurence for each configuration {config_names} for {dataset_name}, " \
                f"{fold} but found {sum(mask)}."
        return [dict(zip(output_cols, row)) for row in df.loc[mask, output_cols].values]

    def val_predictions(self, dataset_name: str, config_name: str, fold: int):
        val_predictions, _ = self._tabular_predictions.predict(
            dataset=self.dataset_to_taskid(dataset_name),
            fold=fold,
            models=[config_name]
        )
        return val_predictions[0]

    def test_predictions(self, dataset_name: str, config_name: str, fold: int):
        _, test_predictions = self._tabular_predictions.predict(
            dataset=self.dataset_to_taskid(dataset_name),
            fold=fold,
            models=[config_name]
        )
        return test_predictions[0]

    def dataset_metadata(self, dataset_name: str) -> dict:
        metadata = self._df_metadata[self._df_metadata.name == dataset_name]
        return dict(zip(metadata.columns, metadata.values[0]))

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

    def get_datasets(self, problem_type=None):
        return self._zeroshot_context.get_datasets(problem_type=problem_type)

    def n_folds(self) -> int:
        return len(self._tabular_predictions.folds)

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


def load(version: str = None):
    from autogluon_zeroshot.contexts import get_context
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = get_context(version).load(load_predictions=True, lazy_format=False)
    return EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )


if __name__ == '__main__':
    with catchtime("loading repo and evaluating one ensemble config"):
        dataset_name = "abalone"
        config_name = "NeuralNetFastAI_r1"
        # repo = EvaluationRepository.load(version="2022_10_13")

        from autogluon_zeroshot.contexts.context_artificial import load_context_artificial
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
        repo = EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=zeroshot_pred_proba,
            ground_truth=zeroshot_gt,
        )

        print(repo.dataset_names()[:3])  # ['abalone', 'ada', 'adult']
        print(repo.task_ids()[:3])  # [2073, 3945, 7593]

        print(repo.dataset_to_taskid(dataset_name))  # 360945
        print(list(repo.list_models_available(dataset_name))[:3])  # ['LightGBM_r181', 'CatBoost_r81', 'ExtraTrees_r33']
        print(repo.eval_metrics(dataset_name=dataset_name, config_names=[config_name], fold=2))  # {'time_train_s': 0.4008138179779053, 'metric_error': 25825.49788, ...
        print(repo.val_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape)
        print(repo.test_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape)
        print(repo.dataset_metadata(dataset_name=dataset_name))  # {'tid': 360945, 'ttid': 'TaskType.SUPERVISED_REGRESSION
        print(repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name], ensemble_size=5, backend="native"))  # [[7.20435338 7.04106921 7.11815431 7.08556309 7.18165966 7.1394064  7.03340405 7.11273415 7.07614767 7.21791022]]
        print(repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name],
                                     ensemble_size=5, folds=[2], backend="native"))  # [[7.11815431]]


