from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from tabarena import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabarena.benchmark.experiment import AGModelBagExperiment, Experiment, ExperimentBatchRunner


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_10"  # 10 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = str(Path(__file__).parent / "experiments" / "quickstart_new_config")  # folder location to save all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch

    # The original TabRepo artifacts for the 1530 configs
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    datasets = repo_og.datasets()

    # Sample for a quick demo
    datasets = datasets[:3]
    folds = [0]

    # To run everything:
    # datasets = repo_og.datasets()
    # folds = repo_og.folds

    # import your model classes (can be custom, must inherit from AbstractModel)
    from autogluon.tabular.models import LGBModel

    # import your non-autogluon model classes (must inherit from AbstractExecModel)
    from tabarena.benchmark.models.simple import SimpleLightGBM

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name="LightGBM_c1_BAG_L1_Reproduced",  # Reproduces the `LightGBM_c1_BAG_L1` model from the TabRepo 2024 paper

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=LGBModel,  # model_cls="GBM",  <- identical
            model_hyperparameters={},  # The non-default model hyperparameters.
            num_bag_folds=8,  # num_bag_folds=8 was used in the TabRepo 2024 paper
            time_limit=3600,  # time_limit=3600 was used in the TabRepo 2024 paper
        ),

        # This will be a `baseline` in EvaluationRepository, because it doesn't compute out-of-fold predictions and thus can't be used for post-hoc ensemble.
        Experiment(  # Generic wrapper for any model, including non-AutoGluon models
            name="LightGBM_Custom",

            # method_cls must inherit from `AbstractExecModel`
            # Simple LightGBM implementation that trains with 100% of the data without splitting for validation, and doesn't early stop.
            # Does not bag. There is an attribute `can_get_oof` set to False.
            method_cls=SimpleLightGBM,

            # kwargs passed to the init call of `method_cls`.
            method_kwargs={
                "hyperparameters": {"learning_rate": 0.05},
            },
        )
    ]

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
    repo: EvaluationRepository = exp_batch_runner.repo_from_results(results_lst=results_lst)

    repo.print_info()

    save_path = "repo_quickstart_new_config"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    print(f"New Baselines : {new_baselines}")
    print(f"New Configs   : {new_configs}")
    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    shared_datasets = [d for d in repo.datasets(union=False) if d in repo_og.datasets()]
    shared_folds = [f for f in repo.folds if f in repo_og.folds]
    repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo], config_fallback="ExtraTrees_c1_BAG_L1")

    # subset repo_combined to only contain datasets and folds shared by both repo and repo_og
    repo_combined = repo_combined.subset(datasets=shared_datasets, folds=shared_folds)

    repo_combined.print_info()

    comparison_configs_og = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
    ]

    # Include our new configs
    comparison_configs = comparison_configs_og + new_configs

    # Baselines to compare configs with
    # baselines = repo_combined.baselines(union=False)  # to compare with all baselines
    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
        "H2OAutoML_4h8c_2023_11_14",
        "autosklearn2_4h8c_2023_11_14",
        "flaml_4h8c_2023_11_14",
        "lightautoml_4h8c_2023_11_14",
    ]
    baselines += new_baselines

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo_combined)
    metrics = evaluator.compare_metrics(
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )

    metrics_tmp = metrics.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")
