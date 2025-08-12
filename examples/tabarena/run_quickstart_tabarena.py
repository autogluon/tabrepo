from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabrepo import EvaluationRepository, Evaluator
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart")  # folder location to save all experiment artifacts
    repo_dir = str(Path(__file__).parent / "repos" / "quickstart")  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata()

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    folds = [0]

    # import your model classes
    from tabrepo.benchmark.models.ag import RealMLPModel

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name="RealMLP_c1_BAG_L1_Reproduced",

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=RealMLPModel,
            model_hyperparameters={
                # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
            },  # The non-default model hyperparameters.
            num_bag_folds=8,  # num_bag_folds=8 was used in the TabArena 2025 paper
            time_limit=3600,  # time_limit=3600 was used in the TabArena 2025 paper
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    experiment_results = ExperimentResults(task_metadata=task_metadata)

    # Convert the run artifacts into an EvaluationRepository
    repo: EvaluationRepository = experiment_results.repo_from_results(results_lst=results_lst)
    repo.print_info()

    repo.to_dir(path=repo_dir)  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`

    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # get the local results
    evaluator = Evaluator(repo=repo)
    metrics: pd.DataFrame = evaluator.compare_metrics().reset_index().rename(columns={"framework": "method"})

    # load the TabArena paper results
    tabarena_context = TabArenaContext()
    tabarena_results = tabarena_context.load_results_paper(download_results="auto")
    tabarena_results = tabarena_results[[c for c in tabarena_results.columns if c in metrics.columns]]

    dataset_fold_map = metrics.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    # filter tabarena_results to only the dataset, fold pairs that are present in `metrics`
    is_in_lst = [is_in(dataset, fold) for dataset, fold in zip(tabarena_results["dataset"], tabarena_results["fold"])]
    tabarena_results = tabarena_results[is_in_lst]

    metrics = pd.concat([
        metrics,
        tabarena_results,
    ], ignore_index=True)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{metrics.head(100)}")

    calibration_framework = "RF (default)"

    from tabrepo.tabarena.tabarena import TabArena
    tabarena = TabArena(
        method_col="method",
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
        elo_kwargs={
            "calibration_framework": calibration_framework,
        },
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
