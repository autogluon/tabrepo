from __future__ import annotations

import pandas as pd
from typing import Callable, List, Type

from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper
from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample as _convert_time_infer_s_from_batch_to_sample
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionPickle, CacheFunctionDummy
from tabrepo import EvaluationRepository
from experiment_runner import ExperimentRunner, OOFExperimentRunner


# TODO: Which save hierarchy?
#  1. `expname/data/tasks/{tid}/{fold}/{method}/results.pkl`  <- Current implementation
#  2. `expname/data/method/{method}/tasks/{tid}/{fold}/results.pkl`
#  3. `expname/data/{tid}/{fold}/{method}/results.pkl`
#  3. `expname/data/{method}/{tid}/{fold}/results.pkl`
# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def generate_repo_from_experiments(
        self,
        expname: str,
        tids: List[int],
        folds: List[int],
        methods: list[tuple[str, Callable, dict[str, ...]]],
        task_metadata: pd.DataFrame,
        ignore_cache: bool,
        experiment_cls: Type[OOFExperimentRunner] = OOFExperimentRunner,
        cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict = None,
        convert_time_infer_s_from_batch_to_sample: bool = False,
    ) -> EvaluationRepository:
        results_lst = run_experiments(
            expname=expname,
            tids=tids,
            folds=folds,
            methods=methods,
            experiment_cls=experiment_cls,
            cache_cls=cache_cls,
            cache_cls_kwargs=cache_cls_kwargs,
            task_metadata=task_metadata,
            ignore_cache=ignore_cache,
        )

        results_baselines = [result["df_results"] for result in results_lst if result["simulation_artifacts"] is None]
        df_baselines = pd.concat(results_baselines, ignore_index=True) if results_baselines else None

        results_configs = [result for result in results_lst if result["simulation_artifacts"] is not None]

        results_lst_simulation_artifacts = [result["simulation_artifacts"] for result in results_configs]
        results_lst_df = [result["df_results"] for result in results_configs]

        df_configs = pd.concat(results_lst_df, ignore_index=True)
        if convert_time_infer_s_from_batch_to_sample:
            df_configs = _convert_time_infer_s_from_batch_to_sample(df=df_configs, task_metadata=task_metadata)

        repo: EvaluationRepository = EvaluationRepository.from_raw(
            df_configs=df_configs,
            df_baselines=df_baselines,
            results_lst_simulation_artifacts=results_lst_simulation_artifacts,
            task_metadata=task_metadata,
        )

        return repo


# TODO: Prateek: Give a toggle for just fitting and saving the model, if not call predict as well
def run_experiments(
    expname: str,
    tids: List[int],
    folds: List[int],
    methods: list[tuple[str, Callable, dict[str, ...]]],
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
    experiment_cls: Type[ExperimentRunner] = ExperimentRunner,
    cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
    cache_cls_kwargs: dict = None,
) -> list:
    '''

    Parameters
    ----------
    expname: str, Name of the experiment given by the user
    tids: list[int], List of OpenML task IDs given by the user
    folds: list[int], Number of folds present for the given task
    methods: list[tuple[str, Callable, dict[str, ...]]], Models used for fit() and predict() in this experiment
    task_metadata: pd.DataFrame,OpenML task metadata
    ignore_cache: bool, whether to use cached results (if present)
    experiment_cls: WIP
    cache_cls: WIP
    cache_cls_kwargs: WIP

    Returns
    -------
    result_lst: list, containing all metrics from fit() and predict() of all the given OpenML tasks
    '''
    if cache_cls is None:
        cache_cls = CacheFunctionDummy
    if cache_cls_kwargs is None:
        cache_cls_kwargs = {}
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
        f"\n\tMethods : {methods}"
    )
    result_lst = []
    num_datasets = len(tids)
    for i, tid in enumerate(tids):
        task = None  # lazy task loading
        task_name = task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0]
        print(f"Starting Dataset {i+1}/{num_datasets}...")
        for fold in folds:
            for method, method_cls, method_kwargs in methods:
                cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                print(
                    f"\tFitting {task_name} on fold {fold} for method {method}"
                )

                cacher = cache_cls(cache_name=cache_name, cache_path=expname, **cache_cls_kwargs)

                if task is None:
                    if ignore_cache or not cacher.exists:
                        task = OpenMLTaskWrapper.from_task_id(task_id=tid)

                if task is not None:
                    out = cacher.cache(
                        fun=experiment_cls.init_and_run,
                        fun_kwargs=dict(
                            method_cls=method_cls,
                            task=task,
                            fold=fold,
                            task_name=task_name,
                            method=method,
                            fit_args=method_kwargs,
                        ),
                        ignore_cache=ignore_cache,
                    )
                else:
                    # load cache, no need to load task
                    out = cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)
                result_lst.append(out)

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
