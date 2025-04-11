import pandas as pd

from autogluon.tabular import TabularPredictor
from tabrepo.benchmark.task.openml import OpenMLTaskWrapper

from tabrepo import EvaluationRepository


def get_artifacts(task: OpenMLTaskWrapper, fold: int, hyperparameters: dict, dataset: str = None, time_limit=60):
    if dataset is None:
        dataset = str(task.task_id)
    print(f"Fitting configs on dataset: {dataset}\t| fold: {fold}")
    train_data, test_data = task.get_train_test_split_combined(fold=fold)
    predictor: TabularPredictor = TabularPredictor(label=task.label).fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        ag_args_fit={"ag.max_time_limit": time_limit},
        fit_weighted_ensemble=False,
        calibrate=False,
        verbosity=0,
    )

    leaderboard = predictor.leaderboard(test_data, score_format="error")
    leaderboard["dataset"] = dataset
    leaderboard["tid"] = task.task_id
    leaderboard["fold"] = fold
    leaderboard["problem_type"] = task.problem_type
    leaderboard.rename(columns={
        "eval_metric": "metric",
        "metric_error_test": "metric_error",
    }, inplace=True)
    simulation_artifact = predictor.simulation_artifact(test_data=test_data)
    simulation_artifacts = {dataset: {fold: simulation_artifact}}
    return simulation_artifacts, leaderboard


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


"""
This tutorial showcases how to generate a small context from scratch using AutoGluon.

For the code to generate the full context, refer to https://github.com/autogluon/tabrepo/tree/main/scripts/execute_benchmarks

Required dependencies:
```bash
# Requires autogluon-benchmark
git clone https://github.com/Innixma/autogluon-benchmark.git
pip install -e autogluon-benchmark
```

This example script runs 7 configs on 3 tiny datasets with 2 folds, for a total of 42 trained models.
The full TabRepo runs 1530 configs on 244 datasets with 3 folds, using 8-fold bagging, for a total of 1,119,960 trained bagged models consisting of 8,959,680 fold models.

"""
if __name__ == '__main__':
    # list of datasets to train on
    datasets = [
        "Australian",
        "blood-transfusion",
        "meta",
    ]
    # dataset to task id map to reference the appropriate OpenML task.
    dataset_to_tid_dict = {
        "Australian": 146818,
        "blood-transfusion": 359955,
        "meta": 3623,
    }
    # time limit in seconds each config gets to train per dataset. Early stopped if exceeded.
    time_limit_per_config = 60  # 3600 in paper

    # the configs to train on each dataset
    hyperparameters = {
        "GBM": {},
        "XGB": {},
        "CAT": {},
        "FASTAI": {},
        "NN_TORCH": {},
        "RF": {},
        "XT": {},
    }

    # the folds to train on each dataset
    folds = [0, 1]

    artifacts = []
    # Fit models on the datasets and get their artifacts
    for dataset in datasets:
        task_id = dataset_to_tid_dict[dataset]
        for fold in folds:
            task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
            artifacts.append(
                get_artifacts(
                    task=task,
                    fold=fold,
                    dataset=dataset,
                    hyperparameters=hyperparameters,
                    time_limit=time_limit_per_config,
                )
            )

    results_lst_simulation_artifacts = [simulation_artifacts for simulation_artifacts, leaderboard in artifacts]

    leaderboards = [leaderboard for simulation_artifacts, leaderboard in artifacts]
    leaderboard_full = pd.concat(leaderboards)

    df_configs = convert_leaderboard_to_configs(leaderboard=leaderboard_full)
    print(df_configs)

    repo = EvaluationRepository.from_raw(df_configs=df_configs, results_lst_simulation_artifacts=results_lst_simulation_artifacts)

    # Note: Can also skip all the above code if you want to use a readily available context rather than generating from scratch:
    # repo = EvaluationRepository.from_context(version="D244_F3_C1530_30", cache=True)

    repo.print_info()

    repo = repo.to_zeroshot()

    results_cv = repo.simulate_zeroshot(num_zeroshot=3, n_splits=2, backend="seq")
    df_results = repo.generate_output_from_portfolio_cv(portfolio_cv=results_cv, name="quickstart")

    # TODO: Fix time_infer_s to only include used models in the ensemble
    # TODO: Add way to fetch model hyperparameters and generate input hyperparameters dict based on `portfolio` column.
    # TODO: Run `portfolio` on datasets to verify that the true performance matches the simulated performance.
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(df_results)
