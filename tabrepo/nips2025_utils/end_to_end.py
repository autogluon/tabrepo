from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Self

import pandas as pd

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.method_processor import generate_task_metadata, load_raw
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.paper.tabarena_evaluator import TabArenaEvaluator

if TYPE_CHECKING:
    from tabrepo.benchmark.result import BaselineResult
    from tabrepo.repository import EvaluationRepository


class EndToEnd:
    def __init__(self, results_lst: list[BaselineResult]):
        cache = True

        # raw
        self.results_lst: list[BaselineResult] = results_lst
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

        tids = list({r.task_metadata["tid"] for r in self.results_lst})
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
    def from_cache(cls, method: str, artifact_name: str | None = None) -> Self:
        if artifact_name is None:
            artifact_name = method
        method_metadata = MethodMetadata.from_yaml(
            method=method, artifact_name=artifact_name
        )
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
        subset: str | None | list = None,
        new_result_prefix: str | None = None,
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
        output_dir = Path(output_dir)

        tabarena_context = TabArenaContext()

        fillna_method = "RF (default)"
        paper_results = tabarena_context.load_results_paper(download_results="auto")

        if new_result_prefix is not None:
            for col in ["method", "config_type", "ta_name", "ta_suite"]:
                self.hpo_results[col] = new_result_prefix + self.hpo_results[col]

        # FIXME: Nick: After imputing: ta_name, ta_suite, config_type, etc. are incorrect,
        #  need to use original, not filled values
        #  This doesn't impact the evaluation, but could introduce bugs in future if we use these columns
        #  Fixing this is do-able, but requires some complex pandas tricks, so I haven't had time to implement it yet
        hpo_results = TabArenaContext.fillna_metrics(
            df_metrics=self.hpo_results,
            df_fillna=paper_results[paper_results["method"] == fillna_method],
        )

        df_results = pd.concat([paper_results, hpo_results], ignore_index=True)

        if subset is not None:
            from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata

            if isinstance(subset, str):
                subset = [subset]

            for filter_subset in subset:
                if filter_subset == "classification":
                    df_results = df_results[
                        df_results["problem_type"].isin(["binary", "multiclass"])
                    ]
                elif filter_subset == "regression":
                    df_results = df_results[df_results["problem_type"] == "regression"]
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

        plotter = TabArenaEvaluator(
            output_dir=output_dir,
        )
        return plotter.eval(
            df_results=df_results,
            plot_extra_barplots=False,
            plot_times=True,
            plot_other=False,
            imputed_names=imputed_names,
        )
