from __future__ import annotations

import pandas as pd

from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.tasks.experiment_utils import run_experiments
from autogluon_benchmark.evaluation.evaluator import Evaluator
from tabrepo import load_repository, get_context, EvaluationRepository


def convert_leaderboard_to_configs(leaderboard: pd.DataFrame, minimal: bool = True) -> pd.DataFrame:
    df_configs = leaderboard.rename(columns=dict(
        model="framework",
        fit_time="time_train_s",
        pred_time_test="time_infer_s"
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
    expname = "./test_results"  # folder location of all experiment artifacts
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

    results_df = results_df.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
        eval_metric="metric",
        val_error="metric_error_val",
    ))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    evaluator = Evaluator(
        # frameworks=frameworks_run,
        task_metadata=task_metadata,
        # treat_folds_as_datasets=treat_folds_as_datasets,
    )
    evaluator_output = evaluator.transform(data=results_df)

    results_ranked_df = evaluator_output.results_ranked

    context_name = "D244_F3_C1530_30"
    context = get_context(name=context_name)
    config_hyperparameters = context.load_configs_hyperparameters()
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    metrics = repo.compare_metrics(results_df, datasets=["blood-transfusion-service-center", "Australian"], configs=["CatBoost_r1_BAG_L1", "LightGBM_r41_BAG_L1"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")


