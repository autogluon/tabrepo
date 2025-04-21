from __future__ import annotations

import copy
from typing import Any, Literal, Type

import pandas as pd

from tabrepo.benchmark.result import AGBagResult, BaselineResult, ConfigResult
from tabrepo.benchmark.task.openml import OpenMLTaskWrapper, OpenMLS3TaskWrapper
from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample as _convert_time_infer_s_from_batch_to_sample
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionPickle, CacheFunctionDummy
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment.experiment_constructor import Experiment


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def __init__(
        self,
        expname: str,
        task_metadata: pd.DataFrame,
        cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict | None = None,
        cache_path_format: Literal["name_first", "task_first"] = "name_first",
        only_cache: bool = False,
        mode: str = "local",
        s3_bucket: str | None = None,
        debug_mode: bool = True,
        s3_dataset_cache: str | None = None,
    ):
        """

        Parameters
        ----------
        expname
        cache_cls
        cache_cls_kwargs
        cache_path_format: {"name_first", "task_first"}, default "name_first"
            Determines the folder structure for artifacts.
            "name_first" -> {expname}/data/{method}/{tid}/{fold}/
            "task_first" -> {expname}/data/tasks/{tid}/{fold}/{method}/
        mode: str, default "local"
            Either "local" or "aws". In "aws" mode, s3_bucket must be provided.
        s3_bucket: str, optional
            Required when mode="aws". The S3 bucket where artifacts will be stored.
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        s3_dataset_cache: str, optional
            Full S3 URI to the openml dataset cache (format: s3://bucket/prefix)
            If None, skip S3 download attempt
        """
        cache_cls = CacheFunctionDummy if cache_cls is None else cache_cls
        cache_cls_kwargs = {"include_self_in_call": True} if cache_cls_kwargs is None else cache_cls_kwargs

        self.expname = expname
        self.task_metadata = task_metadata
        self.cache_cls = cache_cls
        self.cache_cls_kwargs = cache_cls_kwargs
        self.cache_path_format = cache_path_format
        self.only_cache = only_cache
        self._dataset_to_tid_dict = self.task_metadata[['tid', 'dataset']].drop_duplicates(['tid', 'dataset']).set_index('dataset')['tid'].to_dict()
        self.mode = mode
        self.s3_bucket = s3_bucket
        self.debug_mode = debug_mode
        self.s3_dataset_cache = s3_dataset_cache

    @property
    def datasets(self) -> list[str]:
        return list(self._dataset_to_tid_dict.keys())

    def run(
        self,
        methods: list[Experiment],
        datasets: list[str],
        folds: list[int],
        ignore_cache: bool = False,
        raise_on_failure: bool = True,
    ) -> list[dict[str, Any]]:
        """

        Parameters
        ----------
        methods
        datasets
        folds
        ignore_cache: bool, default False
            If True, will run the experiments regardless if the cache exists already, and will overwrite the cache file upon completion.
            If False, will load the cache result if it exists for a given experiment, rather than running the experiment again.

        Returns
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        self._validate_datasets(datasets=datasets)
        self._validate_folds(folds=folds)

        tids = [self._dataset_to_tid_dict[dataset] for dataset in datasets]
        return run_experiments(
            expname=self.expname,
            tids=tids,
            folds=folds,
            methods=methods,
            task_metadata=self.task_metadata,
            ignore_cache=ignore_cache,
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            cache_path_format=self.cache_path_format,
            mode=self.mode,
            s3_bucket=self.s3_bucket,
            only_cache=self.only_cache,
            raise_on_failure=raise_on_failure,
            debug_mode=self.debug_mode,
            s3_dataset_cache=self.s3_dataset_cache
        )

    def load_results(
        self,
        methods: list[Experiment | str],
        datasets: list[str],
        folds: list[int],
        require_all: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Load results from the cache.

        Parameters
        ----------
        methods
        datasets
        folds
        require_all: bool, default True
            If True, will raise an exception if not all methods x datasets x folds have a cached result to load.
            If False, will return only the list of results with a cached result. This can be an empty list if no cached results exist.

        Returns
        -------
        results_lst
            The same output format returned by `self.run`

        """
        results_lst = []
        results_lst_exists = []
        results_lst_missing = []
        for method in methods:
            if isinstance(method, Experiment):
                method_name = method.name
            else:
                method_name = method
            for dataset in datasets:
                for fold in folds:
                    cache_exists = self._cache_exists(method_name=method_name, dataset=dataset, fold=fold)
                    cache_args = (method_name, dataset, fold)
                    if cache_exists:
                        results_lst_exists.append(cache_args)
                        print(method.name, dataset, fold)
                        print(f"\t{cache_exists}")
                    else:
                        results_lst_missing.append(cache_args)
        if require_all and results_lst_missing:
            raise AssertionError(
                f"Missing cached results for {len(results_lst_missing)}/{len(results_lst_exists) + len(results_lst_missing)} experiments! "
                f"\nTo load only the {len(results_lst_exists)} existing experiments, set `require_all=False`, "
                f"or call `exp_batch_runner.run(methods=methods, datasets=datasets, folds=folds)` to run the missing experiments."
                f"\nMissing experiments:\n\t{results_lst_missing}"
            )
        for method_name, dataset, fold in results_lst_exists:
            results_lst.append(self._load_result(method_name=method_name, dataset=dataset, fold=fold))
        return results_lst

    def generate_repo_from_experiments(
        self,
        datasets: list[str],
        folds: list[int],
        methods: list[Experiment],
        ignore_cache: bool,
        convert_time_infer_s_from_batch_to_sample: bool = True,
    ) -> EvaluationRepository:
        """

        Parameters
        ----------
        datasets
        folds
        methods
        ignore_cache
        convert_time_infer_s_from_batch_to_sample

        Returns
        -------
        EvaluationRepository

        """
        results_lst = self.run(
            datasets=datasets,
            folds=folds,
            methods=methods,
            ignore_cache=ignore_cache,
        )

        repo = self.repo_from_results(
            results_lst=results_lst,
            convert_time_infer_s_from_batch_to_sample=convert_time_infer_s_from_batch_to_sample,
        )

        return repo

    def _align_result_input_format(self, result: dict | BaselineResult) -> BaselineResult:
        """
        Converts results in old format to new format
        Keeps results in new format as-is.

        This enables the use of results in the old format alongside results in the new format.

        Parameters
        ----------
        result

        Returns
        -------

        """
        if isinstance(result, BaselineResult):
            return result
        assert isinstance(result, dict)
        result_cls = BaselineResult
        sim_artifacts = result.get("simulation_artifacts", None)
        if sim_artifacts is not None:
            assert isinstance(sim_artifacts, dict)
            dataset = result["dataset"]
            fold = result["fold"]
            result_cls = ConfigResult
            if list(sim_artifacts.keys()) == [dataset]:
                sim_artifacts = sim_artifacts[dataset][fold]
            bag_info = sim_artifacts.get("bag_info", None)
            if bag_info is not None:
                assert isinstance(bag_info, dict)
                result_cls = AGBagResult
        result_obj = result_cls(result=result, convert_format=True, inplace=False)
        return result_obj

    def _calibrate(self, result: ConfigResult) -> ConfigResult:
        problem_type = result.result["problem_type"]
        if problem_type == "multiclass":
            # FIXME: What about binary?
            result_calibrated = result.generate_calibrated(method="v2", name_suffix="_CAL")
        else:
            result_calibrated = copy.deepcopy(result)
            result_calibrated.result["framework"] = result_calibrated.result["framework"] + "_CAL"
        return result_calibrated

    def repo_from_results(
        self,
        results_lst: list[dict[str, Any] | BaselineResult],
        calibrate: bool = False,
        include_holdout: bool = False,
        convert_time_infer_s_from_batch_to_sample: bool = True,  # FIXME: Remove this, it should be False eventually
    ) -> EvaluationRepository:
        results_lst: list[BaselineResult] = [self._align_result_input_format(result) for result in results_lst]

        results_configs: list[ConfigResult] = []
        results_baselines: list[BaselineResult] = []
        for result in results_lst:
            if isinstance(result, ConfigResult):
                results_configs.append(result)
            else:
                results_baselines.append(result)

        n_configs = len(results_configs)
        if calibrate:
            results_configs_calibrated = []
            for i, result in enumerate(results_configs):
                if i % 100 == 0:
                    print(f"Calibrating: {i+1}/{n_configs}\t{result.framework}")
                results_configs_calibrated.append(self._calibrate(result=result))
            results_configs += results_configs_calibrated

        n_configs = len(results_configs)
        if include_holdout:
            for r_i, result in enumerate(results_configs):
                if isinstance(result, AGBagResult):
                    if r_i % 100 == 0:
                        print(f"Generating Holdout Results: {r_i + 1}/{n_configs}\t{result.framework}")
                    results_new: list[BaselineResult] = result.bag_artifacts()
                    results_baselines += results_new

        results_lst_df = [result.compute_df_result() for result in results_configs]
        results_lst_df_baselines = [result.compute_df_result() for result in results_baselines]
        df_configs = pd.concat(results_lst_df, ignore_index=True) if results_lst_df else None
        df_baselines = pd.concat(results_lst_df_baselines, ignore_index=True) if results_lst_df_baselines else None

        if df_configs is not None and convert_time_infer_s_from_batch_to_sample:
            df_configs = _convert_time_infer_s_from_batch_to_sample(df=df_configs, task_metadata=self.task_metadata)
        if df_baselines is not None and convert_time_infer_s_from_batch_to_sample:
            df_baselines = _convert_time_infer_s_from_batch_to_sample(df=df_baselines, task_metadata=self.task_metadata)

        configs_hyperparameters = self.get_configs_hyperparameters(results_configs=results_configs)
        results_lst_simulation_artifacts = [result.generate_old_sim_artifact() for result in results_configs]

        # TODO: per-fold pred_proba_test and pred_proba_val (indices?)
        repo: EvaluationRepository = EvaluationRepository.from_raw(
            df_configs=df_configs,
            df_baselines=df_baselines,
            results_lst_simulation_artifacts=results_lst_simulation_artifacts,
            task_metadata=self.task_metadata,
            configs_hyperparameters=configs_hyperparameters,
        )

        return repo

    def get_configs_hyperparameters(self, results_configs: list[ConfigResult]) -> dict | None:
        configs_hyperparameters = {}
        for result in results_configs:
            if result.framework in configs_hyperparameters:
                continue
            configs_hyperparameters[result.framework] = result.hyperparameters
        if not configs_hyperparameters:
            configs_hyperparameters = None
        return configs_hyperparameters

    def _cache_name(self, method_name: str, dataset: str, fold: int) -> str:
        # TODO: Windows? Use Path?
        tid = self._dataset_to_tid_dict[dataset]
        if self.cache_path_format == "name_first":
            cache_name = f"data/{method_name}/{tid}/{fold}/results"
        elif self.cache_path_format == "task_first":
            # Legacy format from early prototyping
            cache_name = f"data/tasks/{tid}/{fold}/{method_name}/results"
        else:
            raise ValueError(f"Unknown cache_path_format: {self.cache_path_format}")
        return cache_name

    def _cache_exists(self, method_name: str, dataset: str, fold: int) -> bool:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold)
        return cacher.exists

    def _load_result(self, method_name: str, dataset: str, fold: int) -> dict[str, Any]:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold)
        return cacher.load_cache()

    def _get_cacher(self, method_name: str, dataset: str, fold: int) -> AbstractCacheFunction:
        cache_name = self._cache_name(method_name=method_name, dataset=dataset, fold=fold)
        cacher = self.cache_cls(cache_name=cache_name, cache_path=self.expname, **self.cache_cls_kwargs)
        return cacher

    def _validate_datasets(self, datasets: list[str]):
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {self.datasets}"
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError(f"Duplicate datasets present! Ensure all datasets are unique.")

    def _validate_folds(self, folds: list[int]):
        if len(folds) != len(set(folds)):
            raise AssertionError(f"Duplicate folds present! Ensure all folds are unique.")


def run_experiments(
    expname: str,
    tids: list[int],
    folds: list[int],
    methods: list[Experiment],
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
    cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
    cache_cls_kwargs: dict = None,
    cache_path_format: Literal["name_first", "task_first"] = "name_first",
    mode: str = "local",
    s3_bucket: str | None = None,
    only_cache: bool = False,
    raise_on_failure: bool = True,
    debug_mode: bool = True,
    s3_dataset_cache: str | None = None,
) -> list[dict]:
    """

    Parameters
    ----------
    expname: str, Name of the experiment given by the user
    tids: list[int], List of OpenML task IDs given by the user
    folds: list[int], Number of folds present for the given task
    methods: list[Experiment], Models used for fit() and predict() in this experiment
    task_metadata: pd.DataFrame, OpenML task metadata
    ignore_cache: bool, whether to use cached results (if present)
    cache_cls: WIP
    cache_cls_kwargs: WIP
    cache_path_format: {"name_first", "task_first"}, default "name_first"
    mode: {"local", "aws"}, default "local"
    s3_bucket: str, default None
        Required when mode="aws". The S3 bucket where artifacts will be stored.
    raise_on_failure: bool, default True
        If True, will raise exceptions that occur during experiments, stopping all runs.
        If False, will ignore exceptions and continue fitting queued experiments. Experiments with exceptions will not be included in the output list.
    s3_dataset_cache: str, default None
        Full S3 URI to the openml dataset cache (format: s3://bucket/prefix)
        If None, skip S3 download attempt

    Returns
    -------
    result_lst: list[dict], containing all metrics from fit() and predict() of all the given OpenML tasks
    """
    if mode == "aws" and (s3_bucket is None or s3_bucket == ""):
        raise ValueError(f"s3_bucket parameter is required when mode is 'aws', got {s3_bucket}")

    if cache_cls is None:
        cache_cls = CacheFunctionDummy
    if cache_cls_kwargs is None:
        cache_cls_kwargs = {}

    # Modify cache path based on mode
    if mode == "local":
        base_cache_path = expname
    else:
        base_cache_path = f"s3://{s3_bucket}/{expname}"

    methods_og = methods
    methods = []
    for method in methods_og:
        # TODO: remove tuple input option, doing it to keep old scripts working
        if not isinstance(method, Experiment):
            method = Experiment(name=method[0], method_cls=method[1], method_kwargs=method[2])
        methods.append(method)

    unique_names = set()
    for method in methods:
        if method.name in unique_names:
            raise AssertionError(f"Duplicate experiment name found. All names must be unique. name: {method.name}")
        unique_names.add(method.name)

    # FIXME: dataset or name? Where does `dataset` come from, why can it be different from `name`?
    #  Using dataset for now because for some datasets like "GAMETES", the name is slightly different with `.` in `name` being replaced with `_` in `dataset`.
    #  This is probably because `.` isn't a valid name in a file in s3.
    #  TODO: What if `dataset` doesn't exist as a column? Maybe fallback to `name`? Or do the `name` -> `dataset` conversion, or use tid.
    dataset_name_column = "dataset"
    dataset_names = [task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0] for tid in tids]
    print(
        f"Running Experiments for expname: '{expname}'..."
        f"\n\tFitting {len(tids)} datasets and {len(folds)} folds for a total of {len(tids) * len(folds)} tasks"
        f"\n\tFitting {len(methods)} methods on {len(tids) * len(folds)} tasks for a total of {len(tids) * len(folds) * len(methods)} jobs..."
        f"\n\tTIDs    : {tids}"
        f"\n\tDatasets: {dataset_names}"
        f"\n\tFolds   : {folds}"
        f"\n\tMethods : {[method.name for method in methods]}"
    )
    result_lst = []
    num_datasets = len(tids)
    missing_tasks = []
    cur_experiment_idx = -1
    experiment_success_count = 0
    experiment_fail_count = 0
    experiment_missing_count = 0
    experiment_cache_exists_count = 0
    experiment_count_total = len(tids) * len(folds) * len(methods)
    for i, tid in enumerate(tids):
        task = None  # lazy task loading
        task_name = task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0]
        print(f"Starting Dataset {i+1}/{num_datasets}...")
        for fold in folds:
            for method in methods:
                cur_experiment_idx += 1
                if cache_path_format == "name_first":
                    cache_prefix = f"data/{method.name}/{tid}/{fold}"
                    cache_name = "results"
                elif cache_path_format == "task_first":
                    # Legacy format from early prototyping
                    cache_prefix = f"data/tasks/{tid}/{fold}/{method.name}"
                    cache_name = "results"
                else:
                    raise ValueError(f"Invalid cache_path_format: {cache_path_format}")
                print(
                    f"\t"
                    f"{cur_experiment_idx}/{experiment_count_total} ran | "
                    f"{experiment_success_count} success | "
                    f"{experiment_fail_count} fail | "
                    f"{experiment_cache_exists_count} cache_exists | "
                    f"{experiment_missing_count} missing | "
                    f"Fitting {task_name} on fold {fold} for method {method.name}"
                )
                cache_path = f"{base_cache_path}/{cache_prefix}"

                cacher = cache_cls(cache_name=cache_name, cache_path=cache_path, **cache_cls_kwargs)

                cache_exists = cacher.exists
                if cache_exists and not ignore_cache:
                    experiment_cache_exists_count += 1

                if only_cache:
                    if not cache_exists:
                        missing_tasks.append(cache_name)
                        experiment_missing_count += 1
                        continue
                    else:
                        out = cacher.load_cache()
                else:
                    if task is None:
                        if ignore_cache or not cache_exists:
                            if s3_dataset_cache:
                                task = OpenMLS3TaskWrapper.from_task_id(task_id=tid, s3_dataset_cache=s3_dataset_cache)
                            else:
                                task = OpenMLTaskWrapper.from_task_id(task_id=tid)

                    try:
                        out = method.run(
                            task=task,
                            fold=fold,
                            task_name=task_name,
                            cacher=cacher,
                            ignore_cache=ignore_cache,
                            debug_mode=debug_mode,
                        )
                    except Exception as exc:
                        if raise_on_failure:
                            raise
                        print(exc.__class__)
                        out = None
                if out is not None:
                    experiment_success_count += 1
                    result_lst.append(out)
                else:
                    experiment_fail_count += 1

    return result_lst


def convert_leaderboard_to_configs(leaderboard: pd.DataFrame, minimal: bool = True) -> pd.DataFrame:
    df_configs = leaderboard.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
        eval_metric="metric",
        val_error="metric_error_val",
    ))
    if minimal:
        minimal_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
            "tid",
        ]
        if "metric_error_val" in df_configs:
            minimal_columns.append("metric_error_val")
        df_configs = df_configs[minimal_columns]
    return df_configs
