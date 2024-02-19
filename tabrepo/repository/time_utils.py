from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from tabrepo.repository.evaluation_repository import EvaluationRepository


def get_runtime(
        repo: EvaluationRepository,
        tid: int,
        fold: int,
        config_names: Optional[List[str]] = None,
        task_col: str = "task",
        runtime_col: str = 'time_train_s',
        fail_if_missing: bool = True
) -> Dict[str, float]:
    """
    :param repo:
    :param tid:
    :param fold:
    :param config_names:
    :param fail_if_missing: whether to raise an error if some configurations are missing
    :return: a dictionary with keys are elements in `config_names` and the values are runtimes of the configuration
    on the task `tid`_`fold`.
    """
    dataset = repo.tid_to_dataset(tid=tid)
    task = repo.task_name(dataset=dataset, fold=fold)
    if not config_names:
        config_names = repo.configs()
    df_metrics = repo._zeroshot_context.df_configs_ranked
    df_configs = pd.DataFrame(config_names, columns=["framework"]).merge(df_metrics[df_metrics[task_col] == task])
    runtime_configs = dict(df_configs.set_index('framework')[runtime_col])
    missing_configurations = set(config_names).difference(runtime_configs.keys())
    if len(missing_configurations) > 0:
        if fail_if_missing:
            raise ValueError(
                f"not all configurations could be found in available data for the task {task}\n" \
                f"requested: {config_names}\n" \
                f"available: {list(runtime_configs.keys())}."
            )
        else:
            # todo take mean of framework
            if repo._config_fallback is not None:
                df_configs_fallback = pd.DataFrame([repo._config_fallback], columns=["framework"]).merge(df_metrics[df_metrics[task_col] == task])
                runtime_configs_fallback = dict(df_configs_fallback.set_index('framework')[runtime_col])
                fill_value = runtime_configs_fallback[repo._config_fallback]
            else:
                fill_value = np.mean(list(runtime_configs.values()))
            print(f"Imputing missing value {fill_value} for configurations {missing_configurations} on task {task}")
            for configuration in missing_configurations:
                runtime_configs[configuration] = fill_value
    return runtime_configs


def sort_by_runtime(
    repo: EvaluationRepository,
    config_names: List[str],
    ascending: bool = True,
) -> List[str]:
    df_metrics = repo._zeroshot_context.df_configs_ranked
    config_sorted = df_metrics.pivot_table(
        index="framework", columns="tid", values="time_train_s"
    ).median(axis=1).sort_values(ascending=ascending).index.tolist()
    return [c for c in config_sorted if c in set(config_names)]


def filter_configs_by_runtime(
        repo: EvaluationRepository,
        tid: int,
        fold: int,
        config_names: List[str],
        max_cumruntime: Optional[float] = None
) -> List[str]:
    """
    :param repo:
    :param tid:
    :param fold:
    :param config_names:
    :param max_cumruntime:
    :return: A sublist of configuration from `config_names` such that the total cumulative runtime does not exceed
    `max_cumruntime`.
    """
    if not max_cumruntime:
        return config_names
    else:
        assert tid in repo.tids()
        assert fold in repo.folds
        runtime_configs = get_runtime(repo=repo, tid=tid, fold=fold, config_names=config_names, fail_if_missing=False)
        cumruntime = np.cumsum(list(runtime_configs.values()))
        # str_runtimes = ", ".join([f"{name}: {time}" for name, time in zip(runtime_configs.keys(), cumruntime)])
        # print(f"Cumulative runtime:\n {str_runtimes}")

        # gets index where cumulative runtime is bellow the target and next index is above the target
        i = np.searchsorted(cumruntime, max_cumruntime)
        return config_names[:i]

