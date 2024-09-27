from __future__ import annotations

import pandas as pd
from typing import Callable, List
from autogluon_benchmark.frameworks.autogluon.run import ag_eval_metric_map
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper
from tabrepo.utils.cache import DummyExperiment, Experiment


# TODO: Prateek: Give a toggle for just fitting and saving the model, if not call predict as well
# above to-do is mentioned again in fit_custom()
# TODO: Nick: This should not be part of this class.
def run_experiments(
    expname: str,
    tids: List[int],
    folds: List[int],
    methods: List[str],
    methods_dict: dict,
    method_cls,  # FIXME: Nick: This needs to be communicated on a per-method basis
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
    cache_class: Callable | None = Experiment,
    cache_class_kwargs: dict = None,
) -> list:
    '''

    Parameters
    ----------
    expname: str, Name of the experiment given by the user
    tids: list[int], List of OpenML task IDs given by the user
    folds: list[int], Number of folds present for the given task
    methods: list[str], Models used for fit() and predict() in this experiment
    methods_dict: dict, methods (models) mapped to their respective fit_args()
    task_metadata: pd.DataFrame,OpenML task metadata
    ignore_cache: bool, whether to use cached results (if present)
    method_cls: WIP
    cache_class: WIP
    cache_class_kwargs: WIP

    Returns
    -------
    result_lst: list, containing all metrics from fit() and predict() of all the given OpenML tasks
    '''
    # TODO: Prateek, Check usage
    if cache_class is None:
        cache_class = DummyExperiment
    if cache_class_kwargs is None:
        cache_class_kwargs = {}
    dataset_names = [task_metadata[task_metadata["tid"] == tid]["name"].iloc[0] for tid in tids]
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
    for tid in tids:
        task = OpenMLTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
        for fold in folds:
            for method in methods:
                cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                # TODO: Prateek, yet to support fit_args
                fit_args = methods_dict[method]
                print(
                    f"\n\tFitting {task_name} on fold {fold} for method {method}"
                )

                if isinstance(method_cls, dict):
                    cur_method_cls = method_cls[method]
                else:
                    cur_method_cls = method_cls

                experiment = cache_class(
                    expname=expname,
                    name=cache_name,
                    run_fun=lambda: run_experiment(
                        method_cls=cur_method_cls,
                        task=task,
                        fold=fold,
                        task_name=task_name,
                        method=method,
                        fit_args=fit_args,
                    ),
                    **cache_class_kwargs
                )
                # FIXME: The output df still needs evaluation and formatting, currently just has predictions
                # probabilities, fit and infer times
                out = experiment.data(ignore_cache=ignore_cache)
                result_lst.append(out)

    return result_lst


# TODO: Nick: This should not be part of this class.
def run_experiment(method_cls, task: OpenMLTaskWrapper, fold: int, task_name: str, method: str, fit_args: dict = None,
                   **kwargs):
    model = method_cls(
        problem_type=task.problem_type,
        eval_metric=ag_eval_metric_map[task.problem_type],
        **fit_args,
    )

    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold)

    out = model.fit_custom2(X=X_train, y=y_train, X_test=X_test, y_test=y_test)

    out["framework"] = method
    out["dataset"] = task_name
    out["tid"] = task.task_id
    out["fold"] = fold
    out["problem_type"] = task.problem_type
    out["eval_metric"] = model.eval_metric
    print(f"Task  Name: {out['dataset']}")
    print(f"Task    ID: {out['tid']}")
    print(f"Metric    : {out['eval_metric']}")
    print(f"Test Error: {out['test_error']:.4f}")
    print(f"Fit   Time: {out['time_fit']:.3f}s")
    print(f"Infer Time: {out['time_predict']:.3f}s")

    out.pop("predictions")
    out.pop("probabilities")

    df_results = pd.DataFrame([out])
    ordered_columns = ["dataset", "fold", "framework", "test_error", "eval_metric", "time_fit"]
    columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
    df_results = df_results[columns_reorder]
    return df_results


def convert_leaderboard_to_configs(leaderboard: pd.DataFrame, minimal: bool = True) -> pd.DataFrame:
    df_configs = leaderboard.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
        eval_metric="metric",
        val_error="metric_error_val",
    ))
    if minimal:
        df_configs = df_configs[[
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
            "tid",
        ]]
    return df_configs
