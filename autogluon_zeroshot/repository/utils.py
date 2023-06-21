from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from autogluon_zeroshot.repository.evaluation_repository import load, EvaluationRepository


def get_runtime(
        repo: EvaluationRepository,
        tid: int,
        fold: int,
        config_names: List[str],
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
    task = repo.task_name(tid, fold)
    df_metrics = repo._zeroshot_context.df_results_by_dataset_vs_automl
    df_configs = pd.DataFrame(config_names, columns=["framework"]).merge(df_metrics[df_metrics.dataset == task])
    runtime_configs = dict(df_configs.set_index('framework')['time_train_s'])
    if fail_if_missing:
        assert all(c in runtime_configs for c in config_names)
    return runtime_configs


def sort_by_runtime(
    repo: EvaluationRepository,
    config_names: List[str],
    ascending: bool = True,
) -> List[str]:
    df_metrics = repo._zeroshot_context.df_results_by_dataset_vs_automl
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
        runtime_configs = get_runtime(repo=repo, tid=tid, fold=fold, config_names=config_names, fail_if_missing=True)
        cumruntime = np.cumsum(list(runtime_configs.values()))
        # str_runtimes = ", ".join([f"{name}: {time}" for name, time in zip(runtime_configs.keys(), cumruntime)])
        # print(f"Cumulative runtime:\n {str_runtimes}")

        # gets index where cumulative runtime is bellow the target and next index is above the target
        i = np.searchsorted(cumruntime, max_cumruntime)
        return config_names[:i]

