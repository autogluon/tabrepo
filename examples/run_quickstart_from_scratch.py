import pandas as pd

from autogluon.common.savers import save_pd
from autogluon.common.utils.simulation_utils import convert_simulation_artifacts_to_tabular_predictions_dict
from autogluon.tabular import TabularPredictor
from autogluon_benchmark import OpenMLTaskWrapper

from tabrepo import EvaluationRepository
from tabrepo.repository import EvaluationRepositoryZeroshot
from tabrepo.predictions import TabularPredictionsInMemory
from tabrepo.contexts.context import BenchmarkContext, construct_context
from tabrepo.contexts.subcontext import BenchmarkSubcontext
from tabrepo.simulation.ground_truth import GroundTruth


def get_artifacts(task: OpenMLTaskWrapper, fold: int, dataset: str = None):
    if dataset is None:
        dataset = str(task.task_id)
    train_data, test_data = task.get_train_test_split_combined(fold=fold)
    predictor: TabularPredictor = TabularPredictor(label=task.label).fit(
        train_data=train_data,
        hyperparameters={
            "GBM": {},
            "XGB": {},
            "CAT": {},
            "FASTAI": {},
            "NN_TORCH": {},
            "RF": {},
            "XT": {},
        },
        time_limit=60,
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
    simulation_artifact = predictor.get_simulation_artifact(test_data=test_data)
    simulation_artifacts = {dataset: {fold: simulation_artifact}}
    return simulation_artifacts, leaderboard


def convert_leaderboard_to_configs(leaderboard: pd.DataFrame) -> pd.DataFrame:
    df_configs = leaderboard.rename(columns=dict(
        model="framework",
        fit_time="time_train_s",
        pred_time_test="time_infer_s"
    ))
    return df_configs


"""
This tutorial showcases how to generate a context from scratch using AutoGluon.

Required dependencies:
```bash
# FIXME: Requires a specific version of AutoGluon installed via this PR: https://github.com/autogluon/autogluon/pull/3555
#  This will be changed in future

# Requires autogluon-benchmark
git clone https://github.com/Innixma/autogluon-benchmark.git
pushd autogluon-benchmark
pip install -e .
popd
```
"""
if __name__ == '__main__':
    datasets = [
        "Australian",
        "blood-transfusion",
        "meta",
    ]
    dataset_to_tid_dict = {
        "Australian": 146818,
        "blood-transfusion": 359955,
        "meta": 3623,
    }

    folds = [0]

    artifacts = []
    # Fit models on the datasets and get their artifacts
    for dataset in datasets:
        task_id = dataset_to_tid_dict[dataset]
        for fold in folds:
            task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
            artifacts.append(get_artifacts(task=task, fold=fold, dataset=dataset))

    # TODO: Move into AutoGluonTaskWrapper
    simulation_artifacts_full = dict()
    leaderboards = []
    for simulation_artifacts, leaderboard in artifacts:
        leaderboards.append(leaderboard)
    leaderboard_full = pd.concat(leaderboards)
    print(leaderboard_full)
    for simulation_artifacts, leaderboard in artifacts:
        for k in simulation_artifacts.keys():
            if k not in simulation_artifacts_full:
                simulation_artifacts_full[k] = {}
            for f in simulation_artifacts[k]:
                if f in simulation_artifacts_full:
                    raise AssertionError(f"Two results exist for tid {k}, fold {f}!")
                else:
                    simulation_artifacts_full[k][f] = simulation_artifacts[k][f]

    zeroshot_pp, zeroshot_gt = convert_simulation_artifacts_to_tabular_predictions_dict(simulation_artifacts=simulation_artifacts_full)

    save_loc = "./quickstart/"
    save_loc_data_dir = save_loc + "model_predictions/"

    predictions = TabularPredictionsInMemory.from_dict(zeroshot_pp)
    ground_truth = GroundTruth.from_dict(zeroshot_gt)
    predictions.to_data_dir(data_dir=save_loc_data_dir)
    ground_truth.to_data_dir(data_dir=save_loc_data_dir)

    df_configs = convert_leaderboard_to_configs(leaderboard=leaderboard_full)
    save_pd.save(path=f"{save_loc}configs.parquet", df=df_configs)

    context: BenchmarkContext = construct_context(
        name="quickstart",
        datasets=datasets,
        folds=folds,
        local_prefix=save_loc,
        local_prefix_is_relative=False,
        has_baselines=False)
    subcontext = BenchmarkSubcontext(parent=context)

    # Note: Can also skip all the above code if you want to use a readily available context rather than generating from scratch:
    # from tabrepo.contexts import get_subcontext
    # subcontext = get_subcontext(name="D244_F3_C1416_30")

    repo: EvaluationRepository = subcontext.load_from_parent()
    repo: EvaluationRepositoryZeroshot = repo.to_zeroshot()

    results_cv = repo.simulate_zeroshot(num_zeroshot=3, n_splits=2, backend="seq")
    df_results = repo.generate_output_from_portfolio_cv(portfolio_cv=results_cv, name="quickstart")

    # TODO: Fix time_infer_s to only include used models in the ensemble
    # TODO: Add way to fetch model hyperparameters and generate input hyperparameters dict based on `portfolio` column.
    # TODO: Run `portfolio` on datasets to verify that the true performance matches the simulated performance.
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(df_results)
