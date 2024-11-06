from __future__ import annotations

import pandas as pd
from typing import Callable, List

from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper
from tabrepo.utils.cache import DummyExperiment, Experiment
from experiment_runner import ExperimentRunner


# TODO: Prateek: Give a toggle for just fitting and saving the model, if not call predict as well
def run_experiments(
    expname: str,
    tids: List[int],
    folds: List[int],
    methods: list[tuple[str, Callable, dict[str, ...]]],
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
    experiment_cls: Callable = ExperimentRunner,
    cache_cls: Callable | None = Experiment,
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
    cache_cls: WIP
    cache_cls_kwargs: WIP

    Returns
    -------
    result_lst: list, containing all metrics from fit() and predict() of all the given OpenML tasks
    '''
    if cache_cls is None:
        cache_cls = DummyExperiment
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
        task = OpenMLTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0]
        print(f"Starting Dataset {i+1}/{num_datasets}...")
        for fold in folds:
            for method, method_cls, method_kwargs in methods:
                cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                print(
                    f"\tFitting {task_name} on fold {fold} for method {method}"
                )

                experiment = cache_cls(
                    expname=expname,
                    name=cache_name,
                    run_fun=lambda: experiment_cls(
                        method_cls=method_cls,
                        task=task,
                        fold=fold,
                        task_name=task_name,
                        method=method,
                        fit_args=method_kwargs,
                    ).run(),
                    **cache_cls_kwargs,
                )
                # FIXME: The output df still needs evaluation and formatting, currently just has predictions
                # probabilities, fit and infer times
                out = experiment.data(ignore_cache=ignore_cache)
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
