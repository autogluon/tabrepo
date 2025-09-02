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
    """
    End-to-end pipeline for processing and evaluating a **single** method's results.

    This class orchestrates:
      1. Inferring method metadata from raw per-task results.
      2. Building an :class:`EvaluationRepository` with processed artifacts.
      3. Simulating HPO and ensembling under TabArena protocols.
      4. Producing per-task evaluation tables (e.g., metrics, train/infer times).

    Most users should not instantiate this class directly. Prefer
    :meth:`EndToEndSingle.from_raw` or :meth:`EndToEndSingle.from_path_raw`.
    If you are evaluating multiple methods, use :class:`EndToEnd`, which
    manages a list of ``EndToEndSingle`` instances.

    Parameters
    ----------
    method_metadata : MethodMetadata
        Resolved metadata describing the method, its cache locations, and naming.
    repo : EvaluationRepository
        Repository of processed artifacts built from the raw runs and task metadata.
    model_results : pd.DataFrame or None
        Raw per-task model results prior to HPO / model selection.
        These are the original results without simulation on TabArena.
        ``None`` if not yet computed.
    hpo_results : pd.DataFrame or None
        TabArena HPO simulation results (one row per (task, config, seed)).
        ``None`` if not yet computed.

    Attributes
    ----------
    method_metadata : MethodMetadata
        Method identity and on-disk artifact layout (e.g., ``path``, ``path_raw``).
    repo : EvaluationRepository
        Processed repository backing downstream analyses and comparisons.
    model_results : pandas.DataFrame or None
        Raw per-task model results prior to HPO / model selection.
    hpo_results : pandas.DataFrame or None
        Output of TabArena HPO and ensemble simulation for this method.
    task_metadata : pd.DataFrame
        (Property) Task-level metadata table provided by the repository.

    Notes
    -----
    **Caching & Side Effects**
    - The factory constructors (:meth:`from_raw`, :meth:`from_path_raw`) can
      write artifacts to disk when ``cache=True`` and/or ``cache_raw=True``:
        * Method metadata YAML to ``method_metadata.path_metadata``.
        * Raw run pickles under ``method_metadata.path_raw``.
        * Processed repository files under ``method_metadata.path_processed``
    - Naming overrides (``name`` / ``name_suffix``) are applied to all configs
      for consistency. If a unique name cannot be assigned (e.g., multiple
      distinct configs while forcing a single ``name``), underlying helpers may raise.

    See Also
    --------
    EndToEnd
        Multi-method manager that constructs and coordinates multiple
        ``EndToEndSingle`` pipelines.
    EndToEndResultsSingle
        Lightweight container returned by :meth:`to_results`.
    MethodMetadata
        Serialized description of a method and its artifact layout.
    TabArenaContext
        Simulator used internally for HPO/model selection under TabArena.

    Examples
    --------
    Basic usage from raw objects::

        results = [BaselineResult(...), ...]  # one per task/run
        e2e = EndToEndSingle.from_raw(results, cache=True, cache_raw=True)
        res = e2e.to_results()
        print(res.model_results.head())

    From an on-disk raw directory of per-run ``results.pkl`` files::

        e2e = EndToEndSingle.from_path_raw("artifacts/my_method/raw", cache=True)
        res = e2e.to_results()

    Loading a previously cached method::

        e2e = EndToEndSingle.from_cache("MyMethodName")
        res = e2e.to_results()

    """
    def __init__(
        self,
        method_metadata: MethodMetadata,
        repo: EvaluationRepository,
        model_results: pd.DataFrame | None,
        hpo_results: pd.DataFrame | None,
    ):
        self.method_metadata = method_metadata
        self.repo = repo
        self.model_results = model_results
        self.hpo_results = hpo_results

    @property
    def task_metadata(self) -> pd.DataFrame:
        return self.repo.task_metadata

    def configs_hyperparameters(self) -> dict[str, dict | None]:
        return self.repo.configs_hyperparameters()

    @classmethod
    def clean_raw(cls, results_lst: list[BaselineResult | dict]) -> list[BaselineResult]:
        return [r if not isinstance(r, dict) else BaselineResult.from_dict(result=r) for r in results_lst]

    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult | dict],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        cache_raw: bool = True,
        name: str | None = None,
        name_suffix: str | None = None,
        verbose: bool = True,
    ) -> Self:
        """
        Run logic end-to-end and cache all results:
        1. (only if using `from_path_raw`) load raw artifacts
            path_raw should be a directory containing `results.pkl` files for each run.
            In the current code, we require `path_raw` to contain the results of only 1 type of method.
        2. infer method_metadata
        3. infer task_metadata
        4. generate repo (processed data)
        5. generate results (per-task metric scores, train time, infer time, etc.)

        Parameters
        ----------
        results_lst : list[BaselineResult | dict]
            The raw results of the method on all tasks and configs.
        task_metadata : pd.DataFrame or None
            The task_metadata containing information for each task,
            such as the target evaluation metric and problem_type.
            If unspecified, will be inferred from ``results_lst``.
        cache : bool = True
            If True, will cache method metadata, processed data, and results to disk.
        cache_raw : bool = True
            If True, will cache raw data to disk.
        name : str or None = None
            If specified, will overwrite the name of the method.
            Will raise an exception if more than one config is present.
        name_suffix : str or None = None
            If specified, will be appended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
        verbose : bool = True
            If True will log info about the data processing and simulation.

        Returns
        -------
        EndToEndSingle
            An initialized EndToEndSingle class based on the provided raw results_lst.
        """
        log = print if verbose else (lambda *a, **k: None)

        # raw
        results_lst: list[BaselineResult] = cls.clean_raw(results_lst=results_lst)
        results_lst = cls._rename(results_lst=results_lst, name=name, name_suffix=name_suffix)
        method_metadata: MethodMetadata = MethodMetadata.from_raw(
            results_lst=results_lst
        )

        log(
            f"{method_metadata.method}: Creating EndToEndSingle from raw results... "
            f"(cache={cache}, cache_raw={cache_raw})"
        )

        if cache or cache_raw:
            log(f'\tArtifacts will be saved to "{method_metadata.path}"')
        if cache:
            method_metadata.to_yaml()

        if cache_raw:
            log(f'\tCaching raw results to "{method_metadata.path_raw}" ({len(results_lst)} task results)')
            method_metadata.cache_raw(results_lst=results_lst)

        if task_metadata is None:
            log(f"\tFetching task_metadata from OpenML...")
            tids = list({r.task_metadata["tid"] for r in results_lst})
            task_metadata = generate_task_metadata(tids=tids)

        log(f"\tConverting raw results into an EvaluationRepository...")
        # processed
        repo: EvaluationRepository = method_metadata.generate_repo(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=cache,
        )

        log(f"\tSimulating HPO...")
        # results
        tabarena_context = TabArenaContext()
        hpo_results, model_results = tabarena_context.simulate_repo(
            method=method_metadata,
            repo=repo,
            use_rf_config_fallback=False,
            cache=cache,
        )

        log(f"\tComplete!")
        return cls(
            method_metadata=method_metadata,
            repo=repo,
            model_results=model_results,
            hpo_results=hpo_results,
        )

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path,
        cache: bool = True,
        cache_raw: bool = True,
        name: str = None,
        name_suffix: str = None,
        verbose: bool = True,
    ) -> Self:
        results_lst: list[BaselineResult] = load_raw(path_raw=path_raw)
        return cls.from_raw(
            results_lst=results_lst,
            cache=cache,
            cache_raw=cache_raw,
            name=name,
            name_suffix=name_suffix,
            verbose=verbose,
        )

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
    def from_cache(cls, method: str | MethodMetadata, artifact_name: str | None = None) -> Self:
        if isinstance(method, MethodMetadata):
            method_metadata = method
        else:
            if artifact_name is None:
                artifact_name = method
            method_metadata = MethodMetadata.from_yaml(
                method=method, artifact_name=artifact_name,
            )
        repo = method_metadata.load_processed()
        end_to_end_results_single = EndToEndResultsSingle(method_metadata=method_metadata)
        return cls(
            method_metadata=method_metadata,
            repo=repo,
            model_results=end_to_end_results_single.model_results,
            hpo_results=end_to_end_results_single.hpo_results,
        )

    def to_results(self) -> EndToEndResultsSingle:
        return EndToEndResultsSingle(
            method_metadata=self.method_metadata,
            model_results=self.model_results,
            hpo_results=self.hpo_results,
        )


class EndToEndResultsSingle:
    def __init__(
        self,
        method_metadata: MethodMetadata,
        *,
        model_results: pd.DataFrame = None,
        hpo_results: pd.DataFrame = None,
    ):
        self.method_metadata = method_metadata
        if model_results is None:
            model_results = self.method_metadata.load_model_results()
        if hpo_results is None and self.method_metadata.method_type == "config":
            hpo_results = self.method_metadata.load_hpo_results()
        self.model_results = model_results
        self.hpo_results = hpo_results

    @classmethod
    def from_cache(cls, method: str | MethodMetadata, artifact_name: str | None = None) -> Self:
        if isinstance(method, MethodMetadata):
            method_metadata = method
        else:
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
