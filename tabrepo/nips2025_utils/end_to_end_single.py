from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Self

import pandas as pd

from tabrepo.benchmark.result import BaselineResult, ConfigResult
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.compare import compare_on_tabarena
from tabrepo.nips2025_utils.method_processor import generate_task_metadata, load_raw
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    from tabrepo.repository import EvaluationRepository


class EndToEndSingle:
    def __init__(
        self,
        results_lst: list[BaselineResult | dict],
        method_metadata: MethodMetadata,
        task_metadata: pd.DataFrame,
        repo: EvaluationRepository,
        hpo_results: pd.DataFrame | None,
        model_results: pd.DataFrame | None,
    ):
        self.results_lst = results_lst
        self.method_metadata = method_metadata
        self.task_metadata = task_metadata
        self.repo = repo
        self.hpo_results = hpo_results
        self.model_results = model_results

    @classmethod
    def clean_raw(cls, results_lst: list[BaselineResult | dict]) -> list[BaselineResult]:
        return [r if not isinstance(r, dict) else BaselineResult.from_dict(result=r) for r in results_lst]

    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult | dict],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        name: str | None = None,
        name_suffix: str | None = None,
    ) -> Self:
        # raw
        results_lst: list[BaselineResult] = cls.clean_raw(results_lst=results_lst)
        results_lst = cls._rename(results_lst=results_lst, name=name, name_suffix=name_suffix)
        method_metadata: MethodMetadata = MethodMetadata.from_raw(
            results_lst=results_lst
        )

        if cache:
            print(
                f'Artifacts for method {method_metadata.method} will be saved in: "{method_metadata.path}"'
            )

        if cache:
            method_metadata.to_yaml()
            method_metadata.cache_raw(results_lst=results_lst)

        if task_metadata is None:
            tids = list({r.task_metadata["tid"] for r in self.results_lst})
            task_metadata = generate_task_metadata(tids=tids)

        # processed
        repo: EvaluationRepository = method_metadata.generate_repo(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
        )

        # results
        tabarena_context = TabArenaContext()
        hpo_results, model_results = tabarena_context.simulate_repo(
            method=method_metadata,
            repo=repo,
            use_rf_config_fallback=False,
            cache=cache,
        )

        return cls(
            results_lst=results_lst,
            method_metadata=method_metadata,
            task_metadata=task_metadata,
            repo=repo,
            hpo_results=hpo_results,
            model_results=model_results,
        )

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path,
        name: str = None,
        name_suffix: str = None,
    ) -> Self:
        results_lst: list[BaselineResult] = load_raw(path_raw=path_raw)
        return cls.from_raw(results_lst=results_lst, name=name, name_suffix=name_suffix)

    @classmethod
    def _rename(
        cls,
        results_lst: list[BaselineResult],
        name: str = None,
        name_suffix: str = None,
        inplace: bool = True
    ) -> list[BaselineResult]:
        if not inplace:
            results_lst = copy.deepcopy(results_lst)
        if name is not None or name_suffix is not None:
            for r in results_lst:
                r.update_name(name=name, name_suffix=name_suffix)
                if isinstance(r, ConfigResult):
                    r.update_model_type(name=name, name_suffix=name_suffix)
        return results_lst

    @classmethod
    def from_cache(cls, method: str, artifact_name: str | None = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(
            method=method, artifact_name=artifact_name
        )
        results_lst = method_metadata.load_raw()
        return cls(results_lst=results_lst)

    def to_results(self) -> EndToEndResultsSingle:
        return EndToEndResultsSingle(
            method_metadata=self.method_metadata,
            hpo_results=self.hpo_results,
            model_results=self.model_results,
        )


class EndToEndResultsSingle:
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
        only_valid_tasks: bool = False,
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
        results = self.get_results(
            new_result_prefix=new_result_prefix,
            use_model_results=use_model_results,
            fillna=not only_valid_tasks,
        )

        return compare_on_tabarena(
            new_results=results,
            output_dir=output_dir,
            only_valid_tasks=only_valid_tasks,
            subset=subset,
        )

    def get_results(
        self,
        new_result_prefix: str | None = None,
        use_model_results: bool = False,
        fillna: bool = False,
    ) -> pd.DataFrame:
        """
        Get data to compare results on TabArena leaderboard.
            Args:
                new_result_prefix (str | None): If not None, add a prefix to the new
                    results to distinguish new results from the original TabArena results.
                    Use this, for example, if you re-run a model from TabArena.
        """
        use_model_results = self.method_metadata.method_type != "config" or use_model_results

        if use_model_results:
            df_results = self.model_results
        else:
            df_results = self.hpo_results

        if new_result_prefix is not None:
            for col in ["method", "config_type", "ta_name", "ta_suite"]:
                df_results[col] = new_result_prefix + df_results[col]

        if fillna:
            df_results = self.fillna_results_on_tabarena(df_results=df_results)

        return df_results

    @classmethod
    def fillna_results_on_tabarena(cls, df_results: pd.DataFrame) -> pd.DataFrame:
        tabarena_context = TabArenaContext()
        fillna_method = "RF (default)"
        fillna_method_name = "RandomForest"

        df_fillna = tabarena_context.load_results_paper(methods=[fillna_method_name])
        df_fillna = df_fillna[df_fillna["method"] == fillna_method]
        assert not df_fillna.empty

        # FIXME: Nick: After imputing: ta_name, ta_suite, config_type, etc. are incorrect,
        #  need to use original, not filled values
        #  This doesn't impact the evaluation, but could introduce bugs in future if we use these columns
        #  Fixing this is do-able, but requires some complex pandas tricks, so I haven't had time to implement it yet
        return TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=df_fillna,
        )
