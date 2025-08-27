from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing_extensions import Self

from tabrepo.benchmark.result import BaselineResult
from tabrepo.nips2025_utils.compare import compare_on_tabarena
from tabrepo.nips2025_utils.end_to_end_single import EndToEndSingle, EndToEndResultsSingle
from tabrepo.nips2025_utils.method_processor import generate_task_metadata, get_info_from_result, load_raw


class EndToEnd:
    def __init__(
        self,
        end_to_end_lst: list[EndToEndSingle],
    ):
        self.end_to_end_lst = end_to_end_lst

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
        results_lst: list[BaselineResult] = EndToEndSingle.clean_raw(results_lst=results_lst)

        if task_metadata is None:
            tids = list({r.task_metadata["tid"] for r in results_lst})
            task_metadata = generate_task_metadata(tids=tids)

        result_types_dict = {}
        for r in results_lst:
            cur_result = get_info_from_result(result=r)
            cur_tuple = (cur_result["method_type"], cur_result["model_type"])
            if cur_tuple not in result_types_dict:
                result_types_dict[cur_tuple] = []
            result_types_dict[cur_tuple].append(r)

        unique_types = list(result_types_dict.keys())

        end_to_end_lst = []
        for cur_type in unique_types:
            # TODO: Avoid fetching task_metadata repeatedly from OpenML in each iteration
            cur_results_lst = result_types_dict[cur_type]
            cur_end_to_end = EndToEndSingle.from_raw(
                results_lst=cur_results_lst,
                task_metadata=task_metadata,
                cache=cache,
                name=name,
                name_suffix=name_suffix,
            )
            end_to_end_lst.append(cur_end_to_end)
        return cls(end_to_end_lst=end_to_end_lst)

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path,
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        name: str = None,
        name_suffix: str = None,
    ) -> Self:
        results_lst: list[BaselineResult] = load_raw(path_raw=path_raw)
        return cls.from_raw(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
            name=name,
            name_suffix=name_suffix,
        )

    @classmethod
    def from_cache(cls, methods: list[str | tuple[str, str]]) -> Self:
        end_to_end_lst = []
        for method in methods:
            if isinstance(method, tuple):
                method, artifact_name = method
            else:
                artifact_name = None
            end_to_end_single = EndToEndSingle.from_cache(method=method, artifact_name=artifact_name)
            end_to_end_lst.append(end_to_end_single)
        return cls(end_to_end_lst=end_to_end_lst)

    def to_results(self) -> EndToEndResults:
        return EndToEndResults(
            end_to_end_results_lst=[end_to_end.to_results() for end_to_end in self.end_to_end_lst],
        )


class EndToEndResults:
    def __init__(
        self,
        end_to_end_results_lst: list[EndToEndResultsSingle],
    ):
        self.end_to_end_results_lst = end_to_end_results_lst

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
        df_results_lst = []
        for result in self.end_to_end_results_lst:
            df_results_lst.append(result.get_results(
                new_result_prefix=new_result_prefix,
                use_model_results=use_model_results,
                fillna=fillna,
            ))
        df_results = pd.concat(df_results_lst, ignore_index=True)
        return df_results

    @classmethod
    def from_cache(cls, methods: list[str | tuple[str, str]]) -> Self:
        end_to_end_results_lst = []
        for method in methods:
            if isinstance(method, tuple):
                method, artifact_name = method
            else:
                artifact_name = None
            end_to_end_results = EndToEndResultsSingle.from_cache(method=method, artifact_name=artifact_name)
            end_to_end_results_lst.append(end_to_end_results)
        return cls(end_to_end_results_lst=end_to_end_results_lst)
