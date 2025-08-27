from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Self

import pandas as pd

from tabrepo.benchmark.result import BaselineResult, ConfigResult
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.method_processor import get_info_from_result, generate_task_metadata, load_raw
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator

if TYPE_CHECKING:
    from tabrepo.repository import EvaluationRepository


class EndToEndMulti:
    def __init__(
        self,
        results_lst: list[BaselineResult | dict],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
    ):
        # raw
        self.results_lst: list[BaselineResult] = EndToEnd.clean_raw(results_lst=results_lst)

        result_types_dict = {}
        for r in self.results_lst:
            cur_result = get_info_from_result(result=r)
            cur_tuple = (cur_result["method_type"], cur_result["model_type"])
            if cur_tuple not in result_types_dict:
                result_types_dict[cur_tuple] = []
            result_types_dict[cur_tuple].append(r)

        unique_types = list(result_types_dict.keys())

        self.end_to_end_lst = []
        for cur_type in unique_types:
            # TODO: Avoid fetching task_metadata repeatedly from OpenML in each iteration
            cur_results_lst = result_types_dict[cur_type]
            cur_end_to_end = EndToEnd(results_lst=cur_results_lst, task_metadata=task_metadata, cache=cache)
            self.end_to_end_lst.append(cur_end_to_end)

    def to_results(self) -> EndToEndResultsMulti:
        return EndToEndResultsMulti(
            end_to_end_results_lst=[end_to_end.to_results() for end_to_end in self.end_to_end_lst],
        )


class EndToEnd:
    def __init__(
        self,
        results_lst: list[BaselineResult | dict],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
    ):
        # raw
        self.results_lst: list[BaselineResult] = self.clean_raw(results_lst=results_lst)
        self.method_metadata: MethodMetadata = MethodMetadata.from_raw(
            results_lst=self.results_lst
        )

        if cache:
            print(
                f'Artifacts for method {self.method_metadata.method} will be saved in: "{self.method_metadata.path}"'
            )

        if cache:
            self.method_metadata.to_yaml()
            self.method_metadata.cache_raw(results_lst=self.results_lst)

        if task_metadata is None:
            tids = list({r.task_metadata["tid"] for r in self.results_lst})
            task_metadata = generate_task_metadata(tids=tids)
        self.task_metadata = task_metadata

        # processed
        self.repo: EvaluationRepository = self.method_metadata.generate_repo(
            results_lst=self.results_lst,
            task_metadata=self.task_metadata,
            cache=cache,
        )

        # results
        tabarena_context = TabArenaContext()
        self.hpo_results, self.model_results = tabarena_context.simulate_repo(
            method=self.method_metadata,
            repo=self.repo,
            use_rf_config_fallback=False,
            cache=cache,
        )

    @classmethod
    def clean_raw(cls, results_lst: list[BaselineResult | dict]) -> list[BaselineResult]:
        return [r if not isinstance(r, dict) else BaselineResult.from_dict(result=r) for r in results_lst]

    @classmethod
    def from_path_raw(cls, path_raw: str | Path, name: str = None, name_suffix: str = None) -> Self:
        results_lst: list[BaselineResult] = load_raw(path_raw=path_raw)
        if name is not None or name_suffix is not None:
            for r in results_lst:
                r.update_name(name=name, name_suffix=name_suffix)
                if isinstance(r, ConfigResult):
                    r.update_model_type(name=name, name_suffix=name_suffix)
        return cls(results_lst=results_lst)

    @classmethod
    def from_cache(cls, method: str, artifact_name: str | None = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(
            method=method, artifact_name=artifact_name
        )
        results_lst = method_metadata.load_raw()
        return cls(results_lst=results_lst)

    def to_results(self) -> EndToEndResults:
        return EndToEndResults(
            method_metadata=self.method_metadata,
            hpo_results=self.hpo_results,
            model_results=self.model_results,
        )


class EndToEndResultsMulti:
    def __init__(
        self,
        end_to_end_results_lst: list[EndToEndResults],
    ):
        self.end_to_end_results_lst = end_to_end_results_lst

    def compare_on_tabarena(
        self,
        output_dir: str | Path,
        *,
        filter_dataset_fold: bool = False,
        df_results_extra: pd.DataFrame = None,
        subset: str | None | list = None,
        new_result_prefix: str | None = None,
        use_model_results: bool = False,
    ) -> pd.DataFrame:
        """Compare results on TabArena leaderboard.

        Args:
            output_dir (str | Path): Directory to save the results.
            subset (str | None | list): Subset of tasks to evaluate on.
                Options are "classification", "regression", "lite"  for TabArena-Lite,
                "tabicl", "tabpfn", "tabpfn/tabicl", or None for all tasks.
                Or a list of subset names to filter for.
            new_result_prefix (str | None): If not None, add a prefix to the new
                results to distinguish new results from the original TabArena results.
                Use this, for example, if you re-run a model from TabArena.
        """

        df_metrics, baselines_extra = self.get_results(use_model_results=use_model_results)

        return EndToEndResults._compare_on_tabarena(
            output_dir=output_dir,
            df_metrics=df_metrics,
            filter_dataset_fold=filter_dataset_fold,
            baselines_extra=baselines_extra,
            df_results_extra=df_results_extra,
            subset=subset,
            new_result_prefix=new_result_prefix,
        )

    def get_results(self, use_model_results: bool) -> tuple[pd.DataFrame, list[str]]:
        df_metrics_lst = []
        baselines_extra_lst = []
        for result in self.end_to_end_results_lst:
            df_metrics, baselines_extra = result.get_results(use_model_results=use_model_results)
            df_metrics_lst.append(df_metrics)
            baselines_extra_lst.append(baselines_extra)
        df_metrics = pd.concat(df_metrics_lst, ignore_index=True)
        baselines_extra = [b for baselines_extra in baselines_extra_lst for b in baselines_extra]
        return df_metrics, baselines_extra


class EndToEndResults:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        hpo_results: pd.DataFrame = None,
        model_results: pd.DataFrame = None,
    ):
        self.method_metadata = method_metadata
        if hpo_results is None and self.method_metadata.method_type == "config":
            hpo_results = self.method_metadata.load_hpo_results()
        if model_results is None:
            model_results = self.method_metadata.load_model_results()
        self.hpo_results = hpo_results
        self.model_results = model_results

    @classmethod
    def from_cache(cls, method: str, artifact_name: str | None = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(
            method=method, artifact_name=artifact_name
        )
        return cls(method_metadata=method_metadata)

    def compare_on_tabarena(
        self,
        output_dir: str | Path,
        *,
        filter_dataset_fold: bool = False,
        df_results_extra: pd.DataFrame = None,
        subset: str | None | list = None,
        new_result_prefix: str | None = None,
        use_model_results: bool = False,
    ) -> pd.DataFrame:
        """Compare results on TabArena leaderboard.

        Args:
            output_dir (str | Path): Directory to save the results.
            subset (str | None | list): Subset of tasks to evaluate on.
                Options are "classification", "regression", "lite"  for TabArena-Lite,
                "tabicl", "tabpfn", "tabpfn/tabicl", or None for all tasks.
                Or a list of subset names to filter for.
            new_result_prefix (str | None): If not None, add a prefix to the new
                results to distinguish new results from the original TabArena results.
                Use this, for example, if you re-run a model from TabArena.
        """

        df_metrics, baselines_extra = self.get_results(use_model_results=use_model_results)

        return self._compare_on_tabarena(
            output_dir=output_dir,
            df_metrics=df_metrics,
            filter_dataset_fold=filter_dataset_fold,
            baselines_extra=baselines_extra,
            df_results_extra=df_results_extra,
            subset=subset,
            new_result_prefix=new_result_prefix,
        )

    def get_results(self, use_model_results: bool) -> tuple[pd.DataFrame, list[str]]:
        if self.method_metadata.method_type == "config":
            if use_model_results:
                df_metrics = self.model_results
                baselines_extra = list(df_metrics["method"].unique())
            else:
                df_metrics = self.hpo_results
                baselines_extra = []
        else:
            df_metrics = self.model_results
            baselines_extra = list(df_metrics["method"].unique())
        return df_metrics, baselines_extra

    # TODO: Move to a separate file
    @classmethod
    def _compare_on_tabarena(
            cls,
            output_dir: str | Path,
            df_metrics: pd.DataFrame,
            baselines_extra: list[str] = None,
            *,
            filter_dataset_fold: bool = False,
            df_results_extra: pd.DataFrame = None,
            subset: str | list[str] | None = None,
            new_result_prefix: str | None = None,
    ) -> pd.DataFrame:
        df_metrics = df_metrics.copy(deep=True)
        if df_results_extra is not None:
            df_results_extra = df_results_extra.copy(deep=True)

        output_dir = Path(output_dir)

        tabarena_context = TabArenaContext()

        fillna_method = "RF (default)"
        paper_results = tabarena_context.load_results_paper(download_results="auto")

        if new_result_prefix is not None:
            for col in ["method", "config_type", "ta_name", "ta_suite"]:
                df_metrics[col] = new_result_prefix + df_metrics[col]

        if filter_dataset_fold:
            paper_results = cls._filter_to_valid_tasks(
                df_to_filter=paper_results,
                df_filter=df_metrics,
            )
            if df_results_extra is not None:
                df_results_extra = cls._filter_to_valid_tasks(
                    df_to_filter=df_results_extra,
                    df_filter=df_metrics,
                )

        # FIXME: Nick: After imputing: ta_name, ta_suite, config_type, etc. are incorrect,
        #  need to use original, not filled values
        #  This doesn't impact the evaluation, but could introduce bugs in future if we use these columns
        #  Fixing this is do-able, but requires some complex pandas tricks, so I haven't had time to implement it yet
        df_metrics = TabArenaContext.fillna_metrics(
            df_metrics=df_metrics,
            df_fillna=paper_results[paper_results["method"] == fillna_method],
        )

        df_results = pd.concat([paper_results, df_metrics], ignore_index=True)

        if df_results_extra is not None:
            if filter_dataset_fold:
                df_results = cls._filter_to_valid_tasks(
                    df_to_filter=df_results,
                    df_filter=df_results_extra,
                )

            df_results_extra = TabArenaContext.fillna_metrics(
                df_metrics=df_results_extra,
                df_fillna=df_results[df_results["method"] == fillna_method],
            )

            df_results = pd.concat([df_results, df_results_extra], ignore_index=True)
            baselines_extra += list(df_results_extra[df_results_extra["method_type"].isin(['baseline', 'portfolio'])]["method"].unique())

        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            df_results = cls._subset_tasks(df_results=df_results, subset=subset)

        # Handle imputation of names
        imputed_names = list(df_results["method"][df_results["imputed"] > 0].unique())
        if len(imputed_names) == 0:
            imputed_names = None
        if imputed_names is not None:
            from tabrepo.paper.paper_utils import get_method_rename_map

            # remove suffix
            imputed_names = [n.split(" (")[0] for n in imputed_names]
            imputed_names = [get_method_rename_map().get(n, n) for n in imputed_names]
            imputed_names = list(set(imputed_names))
            if "KNN" in imputed_names:
                imputed_names.remove("KNN")
            print(f"Model for which results were imputed: {imputed_names}")

        baselines = [
            "AutoGluon 1.3 (4h)",
        ]
        baselines += baselines_extra

        plotter = TabArenaEvaluator(
            output_dir=output_dir,
        )
        return plotter.eval(
            df_results=df_results,
            baselines=baselines,
            plot_extra_barplots=False,
            plot_times=True,
            plot_other=False,
            imputed_names=imputed_names,
        )

    @classmethod
    def _filter_to_valid_tasks(cls, *, df_to_filter: pd.DataFrame, df_filter: pd.DataFrame) -> pd.DataFrame:
        dataset_fold_map = df_filter.groupby("dataset")["fold"].apply(set)

        def is_in(dataset: str, fold: int) -> bool:
            return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

        # filter `df_to_filter` to only the dataset, fold pairs that are present in `df_filter`
        is_in_lst = [
            is_in(dataset, fold) for dataset, fold in zip(
                df_to_filter["dataset"],
                df_to_filter["fold"],
            )]
        df_filtered = df_to_filter[is_in_lst]
        return df_filtered

    @classmethod
    def _subset_tasks(cls, df_results: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
        from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata

        df_results = df_results.copy(deep=True)
        for filter_subset in subset:
            if filter_subset == "classification":
                df_results = df_results[
                    df_results["problem_type"].isin(["binary", "multiclass"])
                ]
            elif filter_subset == "regression":
                df_results = df_results[df_results["problem_type"] == "regression"]
            elif filter_subset == "medium+":
                task_metadata = load_task_metadata()
                task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 10000]
                valid_datasets = task_metadata["dataset"].unique()
                df_results = df_results[df_results["dataset"].isin(valid_datasets)]
            elif filter_subset == "small":
                task_metadata = load_task_metadata()
                task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 10000]
                valid_datasets = task_metadata["dataset"].unique()
                df_results = df_results[df_results["dataset"].isin(valid_datasets)]
            elif filter_subset == "lite":
                df_results = df_results[df_results["fold"] == 0]
            elif filter_subset == "tabicl":
                allowed_dataset = load_task_metadata(subset="TabICL")[
                    "dataset"
                ].tolist()
                df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
            elif filter_subset == "tabpfn":
                allowed_dataset = load_task_metadata(subset="TabPFNv2")[
                    "dataset"
                ].tolist()
                df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
            elif filter_subset == "tabpfn/tabicl":
                ad_tabicl = load_task_metadata(subset="TabICL")["dataset"].tolist()
                ad_tabpfn = load_task_metadata(subset="TabPFNv2")["dataset"].tolist()
                allowed_dataset = list(set(ad_tabicl).intersection(set(ad_tabpfn)))
                df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
            else:
                raise ValueError(f"Invalid subset {subset} name!")
            df_results = df_results.reset_index(drop=True)
        return df_results
