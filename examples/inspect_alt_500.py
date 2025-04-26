from __future__ import annotations

import numpy as np
import pandas as pd

from tabflow.cli.launch_jobs import JobManager
from tabrepo import EvaluationRepository, Evaluator

from tabrepo.benchmark.batch_utils import compute_batched_tasks_all, get_time_per_dataset_info, estimate_benchmark_runtime


def a(time_per_dataset: pd.DataFrame, folds: list[int]):
    methods_file = "../tabflow/configs/configs_alt_1400.yaml"
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)
    invalid_names = []
    invalid_names += [f"LightGBM_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"RandomForest_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"ExtraTrees_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"CatBoost_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names = set(invalid_names)
    # invalid_names += [f"LightGBM_r{i}_alt_BAG_L1" for i in range(1, 101)]
    methods = [m for m in methods if m["name"] not in invalid_names]

    batch_size_groups = [800, 400, 200, 100, 64, 32, 16, 8, 4, 2, 1]

    tasks_all = compute_batched_tasks_all(
        batch_size_groups=batch_size_groups,
        time_per_dataset=time_per_dataset,
        methods=methods,
        folds=folds,
    )
    return tasks_all


def run():
    repo_dir = "repos/tabarena_big_alt_500"  # location of local cache for fast script running

    repo = EvaluationRepository.from_dir(repo_dir)
    repo.print_info()

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    # print(f"New Baselines : {new_baselines}")
    # print(f"New Configs   : {new_configs}")
    # print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo)
    metrics = evaluator.compare_metrics(
        baselines=new_baselines,
        configs=new_configs,
        fillna=False,
    )
    metrics = metrics.reset_index(drop=False)
    time_per_dataset = get_time_per_dataset_info(metrics=metrics)

    from autogluon.common.savers import save_pd
    save_pd.save(df=time_per_dataset, path="s3://tabarena/time_per_dataset.parquet")


    time_per_dataset = time_per_dataset.sort_values(by="time_total_s_q90")

    time_per_method = metrics.groupby("framework")[["time_train_s", "time_infer_s"]].sum()
    time_per_method["time_total_s"] = time_per_method["time_train_s"] + time_per_method["time_infer_s"]
    time_per_method["time_total_frac"] = time_per_method["time_total_s"] / time_per_method["time_total_s"].sum()
    time_per_method["time_total_s_mean"] = time_per_method["time_total_s"] / len(repo.tasks())

    target_time = 3600 * 4
    batch_size_groups = [800, 400, 200, 100, 64, 32, 16, 8, 4, 2, 1]
    folds = [0]
    n_folds = len(folds)

    def get_batch_size_group(time_s_per_method):
        batch_size_target = target_time / time_s_per_method
        for batch_size in batch_size_groups:
            if batch_size_target >= batch_size:
                return batch_size
        return batch_size_groups[-1]

    time_per_dataset["batch_size_group"] = time_per_dataset["time_total_s_q90"].apply(get_batch_size_group)
    n_methods = 1401 - 400
    time_per_instance_startup = 1.8

    total_time_s = estimate_benchmark_runtime(
        n_methods=n_methods,
        n_folds=n_folds,
        time_per_dataset=time_per_dataset,
        target_time=target_time,
        time_per_instance_startup=time_per_instance_startup,
    )

    methods_file = "../tabflow/configs/configs_alt_1400.yaml"
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)
    invalid_names = []
    invalid_names += [f"LightGBM_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"RandomForest_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"ExtraTrees_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names += [f"CatBoost_r{i}_alt_BAG_L1" for i in range(1, 101)]
    invalid_names = set(invalid_names)
    # invalid_names += [f"LightGBM_r{i}_alt_BAG_L1" for i in range(1, 101)]
    methods = [m for m in methods if m["name"] not in invalid_names]

    tasks_all = compute_batched_tasks_all(
        batch_size_groups=batch_size_groups,
        time_per_dataset=time_per_dataset,
        methods=methods,
        folds=folds,
    )

    methods_dict = {m["name"]: m for m in methods}

    # for t_lst in tasks_all:
    #     for batch_lst in t_lst:
    #         for t in batch_lst:
    #             t[2] = methods_dict[t[2]]

    print()


if __name__ == '__main__':
    run()
