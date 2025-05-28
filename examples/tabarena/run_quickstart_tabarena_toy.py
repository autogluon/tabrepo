from __future__ import annotations

import os
from typing import Any

import pandas as pd

from tabrepo import EvaluationRepository, Evaluator
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.experiment.experiment_constructor import YamlExperimentSerializer


"""
Locally runs the first fold of 3 small datasets for all model families in TabRepo 2
"""

if __name__ == '__main__':
    # first you have to generate configs_all.yaml by running `run_generate_all_configs_v2.py`
    yaml_file = "configs_all.yaml"
    methods: list[AGModelBagExperiment] = YamlExperimentSerializer.from_yaml(yaml_file)

    # Load Context
    context_name = "D244_F3_C1530_30"  # 30 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./experiments/tabarena_toy"  # folder location to save all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    # The original TabRepo artifacts for the 1530 configs
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    datasets = repo_og.datasets()
    folds = repo_og.folds

    # Filter to datasets every method supports
    datasets = repo_og.datasets(problem_type="binary") + repo_og.datasets(problem_type="multiclass")
    task_metadata = repo_og.task_metadata.copy(deep=True)
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 1000]
    task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 30]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 3]
    datasets_valid = list(task_metadata["dataset"])
    datasets = [d for d in datasets if d in datasets_valid]

    # Sample for a quick demo
    datasets = datasets[:3]
    folds = [0]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=repo_og.task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # Convert the run artifacts into an EvaluationRepository
    repo: EvaluationRepository = exp_batch_runner.repo_from_results(results_lst=results_lst, convert_time_infer_s_from_batch_to_sample=False)
    repo.to_dir("repos/tabarena_toy")
    repo.print_info()

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    print(f"New Baselines : {new_baselines}")
    print(f"New Configs   : {new_configs}")
    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # Include our new configs
    # comparison_configs = comparison_configs_og + new_configs
    comparison_configs = new_configs

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo)
    metrics = evaluator.compare_metrics(
        # results_df=results_df,
        datasets=datasets,
        folds=folds,
        # baselines=baselines,
        configs=new_configs,
        fillna=False,
    )
    metrics = metrics.reset_index(drop=False)

    from tabrepo.tabarena.tabarena import TabArena
    tabarena = TabArena(
        method_col="framework",
        task_col="dataset",
        seed_column="fold",
        error_col="metric_error",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
        ],
        groupby_columns=[
            "metric",
            "problem_type",
        ]
    )

    leaderboard = tabarena.leaderboard(
        data=metrics,
        include_elo=True,
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
