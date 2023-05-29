from typing import Union

from typing import List
from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext
from autogluon_zeroshot.simulation.tabular_predictions import TabularModelPredictions
from autogluon_zeroshot.utils.cache import SaveLoadMixin
from autogluon_zeroshot.utils import catchtime


class SimpleRepository(SaveLoadMixin):
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
        # TODO: Move these util functions to simulations or somewhere else to avoid circular imports
        from autogluon_zeroshot.contexts.utils import intersect_folds_and_datasets, force_to_dense, prune_zeroshot_gt
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

    def get_datasets(self, problem_type=None):
        return self._zeroshot_context.get_datasets(problem_type=problem_type)

    def n_folds(self) -> int:
        return len(self._tabular_predictions.folds)


# TODO: git shelve ADD BACK
if __name__ == '__main__':
    from autogluon_zeroshot.contexts.context_artificial import load_context_artificial
    from autogluon_zeroshot.repository.evaluation_repository import EvaluationRepository
    with catchtime("loading repo and evaluating one ensemble config"):
        dataset_name = "abalone"
        config_name = "NeuralNetFastAI_r1"
        # repo = EvaluationRepository.load(version="2022_10_13")

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


