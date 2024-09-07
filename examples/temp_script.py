from __future__ import annotations

import pandas as pd

from autogluon_benchmark.tasks.experiment_utils import run_experiments
from tabrepo import load_repository, EvaluationRepository


# Future Note:This function should lie in the fit()-package and not in TabRepo
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
            "metric_error_val",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
            "tid",
        ]]
    return df_configs


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)

    expname = "./initial_experiment"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    datasets = ["blood-transfusion-service-center", "Australian"]
    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    folds = repo.folds

    methods_dict = {
        "GBM_BAG": {
            "hyperparameters": {"GBM": {}},
            "fit_weighted_ensemble": False,
            "num_bag_folds": 8,
            "num_bag_sets": 1,
        },
        "RF_BAG": {
            "hyperparameters": {"RF": {}},
            "fit_weighted_ensemble": False,
            "num_bag_folds": 8,
            "num_bag_sets": 1,
        }
    }
    methods = list(methods_dict.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        methods_dict=methods_dict,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )
    results_df = pd.concat(results_lst, ignore_index=True)
    results_df = convert_leaderboard_to_configs(results_df)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    metrics = repo.compare_metrics(
        results_df,
        datasets=datasets,
        folds=folds,
        baselines=["AutoGluon_bq_4h8c_2023_11_14"],
        configs=["RandomForest_c1_BAG_L1", "LightGBM_c1_BAG_L1"],
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")
    evaluator_output = repo.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=expname,
    )
