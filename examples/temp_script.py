from __future__ import annotations

import pandas as pd

from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
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
    expname = "./initial_experiment"  # folder location of all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch
    task_metadata = load_task_metadata('task_metadata.csv')

    tids = [359955, 146818]
    folds = [0, 1, 2]

    methods_dict = {
        "TABPFN": {
            "hyperparameters": {"TABPFN": {}},
            "fit_weighted_ensemble": False,
        },
        "RF": {
            "hyperparameters": {"RF": {}},
            "fit_weighted_ensemble": False,
        }
    }
    methods = list(methods_dict.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        methods_dict=methods_dict,
        task_metadata=task_metadata,
        ignore_cache=ignore_cache,
    )
    results_df = pd.concat(results_lst, ignore_index=True)
    results_df = convert_leaderboard_to_configs(results_df)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    # Load Context
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)

    metrics = repo.compare_metrics(results_df, datasets=["blood-transfusion-service-center", "Australian"], configs=["CatBoost_r1_BAG_L1", "LightGBM_r41_BAG_L1"], folds=[0,1,2])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")
    evaluator_output = repo.plot_overall_rank_comparison(results_df=metrics, task_metadata=task_metadata, save_dir=expname)

