from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabrepo import EvaluationRepository, Evaluator
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.experiment.experiment_constructor import AGModelOuterExperiment
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils import load_results


# Note: Weights are not yet publicly available.
def download_weights():
    s3_path = "s3://tabfm/mix5_mult_cat/weights/model_step_22000.pt"
    s3_path_reg = "s3://tabfm/mix5_reg.pt"
    from autogluon.common.utils.s3_utils import download_s3_file
    download_s3_file(local_path="weights/model_step_22000.pt", s3_path=s3_path)
    download_s3_file(local_path="weights/mix5_reg.pt", s3_path=s3_path_reg)


if __name__ == '__main__':
    # download_weights()  # uncomment to get the weights, otherwise the below will fail
    expname = str(Path(__file__).parent / "experiments" / "tabarena_mitra")  # folder location to save all experiment artifacts
    repo_dir = str(Path(__file__).parent / "repos" / "tabarena_mitra")
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata()
    # TODO: make a function for this
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 15000]
    task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 500]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]

    datasets = list(task_metadata["name"])
    folds = [0, 1, 2]

    threshold = 2500
    dataset_folds_repeats_lst = []
    for dataset in datasets:
        num_instances = task_metadata[task_metadata["dataset"] == dataset].iloc[0]["NumberOfInstances"]
        if num_instances < threshold:
            repeats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            repeats = [0, 1, 2]
        dataset_folds_repeats = (dataset, folds, repeats)
        dataset_folds_repeats_lst.append(dataset_folds_repeats)

    from tabrepo.benchmark.models.ag import TabPFNV2Model
    from tabrepo.benchmark.models.ag.mitra.mitra_model import MitraModel

    mitra_state_dict_params = {
        "state_dict_classification": "weights/model_step_22000.pt",
        "state_dict_regression": "weights/mix5_reg.pt",
    }

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        AGModelBagExperiment(
            name="Mitra_c1_BAG_L1",
            model_cls=MitraModel,
            model_hyperparameters={
                "epoch": 50,
                "n_estimators": 1,
                **mitra_state_dict_params,
                # TODO: Currently mitra can go out of memory on GPU in parallel mode
                # "ag_args_ensemble": {"fold_fitting_strategy": "parallel_local"},
            },
            num_bag_folds=8,
            time_limit=14400,  # TODO: Mitra currently ignores time_limit
        ),
        AGModelOuterExperiment(
            name="Mitra_c1_n1e0",
            model_cls=MitraModel,
            model_hyperparameters={
                "epoch": 0,
                "n_estimators": 1,
                **mitra_state_dict_params,
            },
        ),
        AGModelOuterExperiment(
            name="TabPFNv2_c1_n1e0",
            model_cls=TabPFNV2Model,
            model_hyperparameters={
                "n_estimators": 1,
            },
        )
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run_w_folds_per_dataset(
        dataset_folds_repeats_lst=dataset_folds_repeats_lst,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    experiment_results = ExperimentResults(task_metadata=task_metadata)

    # Convert the run artifacts into an EvaluationRepository
    repo: EvaluationRepository = experiment_results.repo_from_results(results_lst=results_lst)
    repo.print_info()

    repo.to_dir(path=repo_dir)  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`

    new_baselines = repo.baselines()
    new_configs = repo.configs()
    print(f"New Baselines : {new_baselines}")
    print(f"New Configs   : {new_configs}")
    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo)
    # metrics = evaluator.compare_metrics().reset_index().rename(columns={"framework": "method"})

    from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
    plotter = PaperRunTabArena(repo=repo, output_dir="output_dir_mitra")

    metrics = plotter.run_configs()
    metrics = metrics.rename(columns={"framework": "method"})

    dataset_fold_map = metrics.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    tabarena_results = load_results()
    tabarena_results = tabarena_results[[c for c in tabarena_results.columns if c in metrics.columns]]

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
        include_champ_delta=True,
        elo_kwargs={
            "calibration_framework": calibration_framework,
        }
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)

    metrics = metrics.rename(columns={"method": "framework"})
    plotter.eval(df_results=metrics)

    from autogluon.common.utils.s3_utils import upload_s3_folder
    folder_to_upload = plotter.output_dir
    upload_s3_folder(bucket="tabarena", prefix="mitra_tabarena_results", folder_to_upload=folder_to_upload)
