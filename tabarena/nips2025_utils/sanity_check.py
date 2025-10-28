from __future__ import annotations

import pandas as pd

from tabarena import EvaluationRepository, Evaluator


def filter_to_all_valid_datasets(repo: EvaluationRepository):
    unique_methods = repo.configs() + repo.baselines()
    unique_datasets = repo.datasets()
    unique_methods_set = set(unique_methods)

    valid_datasets = []
    for dataset in unique_datasets:
        valid_configs = repo.configs(datasets=[dataset], union=False)
        valid_baselines = repo.baselines(datasets=[dataset], union=False)

        valid_methods_set = set(valid_configs + valid_baselines)
        if unique_methods_set == valid_methods_set:
            valid_datasets.append(dataset)

    repo = repo.subset(datasets=valid_datasets)
    return repo


def sanity_check(repo: EvaluationRepository, fillna: bool = True, filter_to_all_valid: bool = False, results_df: pd.DataFrame = None, results_df_extra: pd.DataFrame = None):
    repo.print_info()

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    print(f"New Baselines : {new_baselines}")
    print(f"New Configs   : {new_configs}")
    # new_configs = [n for n in new_configs if "_c" in n]
    # print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    if filter_to_all_valid:
        repo = filter_to_all_valid_datasets(repo=repo)

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo)
    metrics = evaluator.compare_metrics(
        results_df=results_df,
        baselines=new_baselines,
        configs=new_configs,
        fillna=False,
    )
    if results_df_extra is not None:
        metrics = pd.concat([metrics, results_df_extra])

    metrics = metrics.reset_index(drop=False)

    from autogluon.common.savers import save_pd
    save_pd.save(path="metrics_sim/metrics.parquet", df=metrics)

    from tabarena.tabarena.tabarena import TabArena
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


        # metrics = metrics[metrics["dataset"].isin(valid_datasets)]

    treat_folds_as_datasets = False
    if treat_folds_as_datasets:
        metrics["dataset"] = metrics["dataset"] + "_" + metrics["fold"].astype(str)
        metrics["fold"] = 0

    if fillna:
        metrics_fillna = tabarena.fillna_data(data=metrics)
    else:
        metrics_fillna = metrics

    n_datasets = len(metrics["dataset"].unique())
    print(f"Evaluating {n_datasets} datasets...")

    leaderboard = tabarena.leaderboard(
        data=metrics_fillna,
        # include_elo=True,
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)

    # repo.to_dir(repo_dir)
