from __future__ import annotations

import itertools
from typing import Tuple, Type

import numpy as np
import pandas as pd

from .time_utils import get_runtime
from ..simulation.ensemble_selection_config_scorer import EnsembleScorer, EnsembleScorerMaxModels, EnsembleSelectionConfigScorer
from ..utils.parallel_for import parallel_for


class EnsembleMixin:
    # TODO: rank=False by default, include way more information like fit time and infer time?
    # TODO: Add time_train_s
    # TODO: Add infer_limit
    def evaluate_ensemble(
        self,
        datasets: list[str],
        configs: list[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        rank: bool = True,
        folds: list[int] | None = None,
        backend: str = "ray",
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        :param datasets: list of datasets to compute errors on.
        :param configs: list of config to consider for ensembling. Uses all configs if None.
        :param ensemble_size: number of members to select with Caruana.
        :param ensemble_cls: class used for the ensemble model.
        :param ensemble_kwargs: kwargs to pass to the init of the ensemble class.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :param backend: Options include ["native", "ray"].
        :return: Tuple:
            Pandas Series of ensemble test errors per task, with multi-index (dataset, fold).
            Pandas DataFrame of ensemble weights per task, with multi-index (dataset, fold). Columns are the names of each config.
        """
        if folds is None:
            folds = self.folds
        if configs is None:
            configs = self.configs()
        tasks = [
            self.task_name(dataset=dataset, fold=fold)
            for dataset in datasets
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            tasks=tasks,
            ensemble_size=ensemble_size,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            backend=backend,
        )

        dict_errors, dict_ensemble_weights = scorer.compute_errors(configs=configs)

        if rank:
            dict_scores = scorer.compute_ranks(errors=dict_errors)
            out = dict_scores
        else:
            out = dict_errors

        dataset_folds = [(self.task_to_dataset(task=task), self.task_to_fold(task=task)) for task in tasks]
        ensemble_weights = [dict_ensemble_weights[task] for task in tasks]
        out_list = [out[task] for task in tasks]

        multiindex = pd.MultiIndex.from_tuples(dataset_folds, names=["dataset", "fold"])

        df_name = "rank" if rank else "metric_error"
        df_out = pd.Series(data=out_list, index=multiindex, name=df_name)
        df_ensemble_weights = pd.DataFrame(data=ensemble_weights, index=multiindex, columns=configs)

        return df_out, df_ensemble_weights

    # FIXME: Delete this, move logic into evaluate_ensemble once evaluate_ensemble returns two DataFrames.
    def evaluate_ensemble_with_time(
        self,
        dataset: str,
        fold: int,
        configs: list[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        time_limit: float = None,
        fit_order: str = "original",
        seed: int = 0,
        rank: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if configs is None:
            configs = self.configs()

        if time_limit is not None:
            if fit_order == "random":
                rng = np.random.default_rng(seed=seed)
                configs_fit_order = list(rng.permuted(configs))
            else:
                configs_fit_order = configs

            from .time_utils import filter_configs_by_runtime
            tid = self.dataset_to_tid(dataset)
            configs = filter_configs_by_runtime(repo=self, tid=tid, fold=fold, config_names=configs_fit_order, max_cumruntime=time_limit)

            if len(configs) == 0:
                if self._config_fallback is None:
                    if len(configs_fit_order) > 0:
                        raise AssertionError(
                            f"Can't fit an ensemble with no configs when self._config_fallback is None "
                            f"(No configs are trainable in the provided time_limit={time_limit}.)"
                        )
                    else:
                        raise AssertionError(f"Can't fit an ensemble with no configs when self._config_fallback is None.")
                configs = [self._config_fallback]

        df_out, df_ensemble_weights = self.evaluate_ensemble(
            datasets=[dataset],
            configs=configs,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            ensemble_size=ensemble_size,
            rank=rank,
            folds=[fold],
            backend="native",
        )

        # FIXME: Make this for loop faster, it is noticeably slow currently
        # FIXME: add metric_error_val?
        # select configurations used in the ensemble as infer time only depends on the models with non-zero weight.
        fail_if_missing = self._config_fallback is None
        task_time_map = {}
        for dataset, fold in df_out.index:
            tid = self.dataset_to_tid(dataset=dataset)
            config_weights = df_ensemble_weights.loc[(dataset, fold)]
            config_selected_ensemble = [
                config for config in configs if config_weights[config] != 0
            ]

            runtimes = get_runtime(
                repo=self,
                tid=tid,
                fold=fold,
                config_names=configs,
                runtime_col='time_train_s',
                fail_if_missing=fail_if_missing,
            )
            latencies = get_runtime(
                repo=self,
                tid=tid,
                fold=fold,
                config_names=config_selected_ensemble,
                runtime_col='time_infer_s',
                fail_if_missing=fail_if_missing,
            )
            time_train_s = sum(runtimes.values())
            time_infer_s = sum(latencies.values())

            task_time_map[(dataset, fold)] = {"time_train_s": time_train_s, "time_infer_s": time_infer_s}
        df_out = df_out.to_frame()
        df_task_time = pd.DataFrame(task_time_map).T
        df_out[["time_train_s", "time_infer_s"]] = df_task_time
        df_datasets_info = self.datasets_info(datasets=[dataset])
        df_out = df_out.join(df_datasets_info, how="inner")
        df_datasets_to_tids = self.datasets_to_tids(datasets=[dataset]).to_frame()
        df_datasets_to_tids.index.name = "dataset"
        df_out = df_out.join(df_datasets_to_tids, how="inner")

        return df_out, df_ensemble_weights

    def evaluate_ensemble_with_time_multi(
        self,
        datasets: list[str] = None,
        folds: list[int] = None,
        configs: list[str] = None,
        *,
        ensemble_cls: Type[EnsembleScorer] = EnsembleScorerMaxModels,
        ensemble_kwargs: dict = None,
        ensemble_size: int = 100,
        time_limit: float = None,
        fit_order: str = "original",
        seed: int = 0,
        rank: bool = True,
        backend: str = "ray",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if backend == "native":
            backend = "sequential"
        if configs is None:
            configs = self.configs()
        if folds is None:
            folds = self.folds
        if datasets is None:
            datasets = self.datasets()

        context = dict(
            self=self,
            configs=configs,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            ensemble_size=ensemble_size,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            rank=rank,
        )

        inputs = list(itertools.product(datasets, folds))
        inputs = [{"dataset": dataset, "fold": fold} for dataset, fold in inputs]

        list_rows = parallel_for(
            self.__class__.evaluate_ensemble_with_time,
            inputs=inputs,
            context=context,
            engine=backend,
        )

        df_out = pd.concat([l[0] for l in list_rows], axis=0)
        df_ensemble_weights = pd.concat([l[1] for l in list_rows], axis=0)  # FIXME: Is this guaranteed same columns in each?

        return df_out, df_ensemble_weights

    def _construct_ensemble_selection_config_scorer(self,
                                                    ensemble_size: int = 10,
                                                    backend='ray',
                                                    **kwargs) -> EnsembleSelectionConfigScorer:
        config_scorer = EnsembleSelectionConfigScorer.from_repo(
            repo=self,
            ensemble_size=ensemble_size,  # 100 is better, but 10 allows to simulate 10x faster
            backend=backend,
            **kwargs,
        )
        return config_scorer
