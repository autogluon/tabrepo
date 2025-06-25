from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from typing_extensions import Self

from tabrepo.benchmark.result import BaselineResult
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.method_processor import generate_task_metadata, load_raw
from tabrepo.repository import EvaluationRepository
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator


class EndToEnd:
    def __init__(self, results_lst: list[BaselineResult]):
        cache = True

        # raw
        self.results_lst: list[BaselineResult] = results_lst
        self.method_metadata: MethodMetadata = MethodMetadata.from_raw(results_lst=self.results_lst)

        if cache:
            print(f'Artifacts for method {self.method_metadata.method} will be saved in: "{self.method_metadata.path}"')

        if cache:
            self.method_metadata.to_yaml()
            self.method_metadata.cache_raw(results_lst=self.results_lst)

        tids = list(set(r.task_metadata["tid"] for r in self.results_lst))
        self.task_metadata = generate_task_metadata(tids=tids)

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
            use_rf_config_fallback=False,
            cache=cache,
        )

    @classmethod
    def from_path_raw(cls, path_raw: str | Path) -> Self:
        results_lst: list[BaselineResult] = load_raw(path_raw=path_raw)
        return cls(results_lst=results_lst)

    @classmethod
    def from_cache(cls, method: str, artifact_name: str = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(method=method, artifact_name=artifact_name)
        results_lst = method_metadata.load_raw()
        return cls(results_lst=results_lst)


class EndToEndResults:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        hpo_results: pd.DataFrame = None,
        model_results: pd.DataFrame = None,
    ):
        self.method_metadata = method_metadata
        if hpo_results is None:
            hpo_results = self.method_metadata.load_hpo_results()
        if model_results is None:
            model_results = self.method_metadata.load_model_results()
        self.hpo_results = hpo_results
        self.model_results = model_results

    @classmethod
    def from_cache(cls, method: str, artifact_name: str = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(method=method, artifact_name=artifact_name)
        return cls(method_metadata=method_metadata)

    def compare_on_tabarena(self, output_dir: str | Path, *, subset: str | None = None) -> pd.DataFrame:
        """"Compare results on TabArena leaderboard.

        Args:
            output_dir (str | Path): Directory to save the results.
            subset (str | None): Subset of tasks to evaluate on.
                Options are "classification", "regression", "lite" for TabArena-Lite,
                or None for all tasks.
        """
        output_dir = Path(output_dir)

        tabarena_context = TabArenaContext()

        fillna_method = "RF (default)"
        paper_results = tabarena_context.load_results_paper(download_results="auto")

        # FIXME: Nick: After imputing: ta_name, ta_suite, config_type, etc. are incorrect,
        #  need to use original, not filled values
        #  This doesn't impact the evaluation, but could introduce bugs in future if we use these columns
        #  Fixing this is do-able, but requires some complex pandas tricks, so I haven't had time to implement it yet
        hpo_results = TabArenaContext.fillna_metrics(
            df_metrics=self.hpo_results,
            df_fillna=paper_results[paper_results["method"] == fillna_method]
        )

        df_results = pd.concat([paper_results, hpo_results], ignore_index=True)

        if subset == "classification":
            df_results = df_results[df_results["problem_type"].isin(["binary","multiclass"])].reset_index(drop=True)
        elif subset == "regression":
            df_results = df_results[df_results["problem_type"] == "regression"].reset_index(drop=True)
        elif "lite":
            df_results = df_results[df_results["fold"] == 0].reset_index(drop=True)


        plotter = TabArenaEvaluator(
            output_dir=output_dir,
        )
        imputed_names = list(df_results["method"][df_results["imputed"] > 0].unique())
        if len(imputed_names) == 0:
            imputed_names = None
        leaderboard = plotter.eval(
            df_results=df_results,
            plot_extra_barplots=False,
            plot_times=True,
            plot_other=False,
            imputed_names=imputed_names,
        )
        return leaderboard
