from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.models.ag.mitra.mitra_model import MitraModel
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils import load_results
from tabrepo.tabarena.tabarena import TabArena
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena


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

    task_metadata = load_task_metadata(subset="TabPFNv2")
    datasets = list(task_metadata["name"])
    dataset_folds_repeats_lst = []
    for dataset in datasets:
        dataset_info = task_metadata[task_metadata["name"] == dataset].iloc[0]
        repeats = [i for i in range(dataset_info["n_repeats"])]
        folds = [i for i in range(dataset_info["n_folds"])]
        dataset_folds_repeats = (dataset, folds, repeats)
        dataset_folds_repeats_lst.append(dataset_folds_repeats)

    mitra_state_dict_params = {
        "state_dict_classification": "workspace/weights/model_step_22000.pt",
        "state_dict_regression": "workspace/weights/mix5_reg.pt",
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
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run_w_folds_per_dataset(
        dataset_folds_repeats_lst=dataset_folds_repeats_lst,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # Convert the run artifacts into an EvaluationRepository
    repo: EvaluationRepository = ExperimentResults(task_metadata=task_metadata).repo_from_results(results_lst=results_lst)
    repo.print_info()

    repo.to_dir(path=repo_dir)  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`

    plotter = PaperRunTabArena(repo=repo, output_dir="output_dir_mitra")

    df_results = plotter.run_configs()
    df_results = df_results.rename(columns={"framework": "method"})
    dataset_fold_map = df_results.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    tabarena_results = load_results()
    tabarena_results = tabarena_results[[c for c in tabarena_results.columns if c in df_results.columns]]

    is_in_lst = [is_in(dataset, fold) for dataset, fold in zip(tabarena_results["dataset"], tabarena_results["fold"])]
    tabarena_results = tabarena_results[is_in_lst]

    df_results = pd.concat([df_results, tabarena_results], ignore_index=True)

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
        data=df_results,
        include_elo=True,
        include_champ_delta=True,
        elo_kwargs={"calibration_framework": "RF (default)"},
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)

    df_results = df_results.rename(columns={"method": "framework"})
    plotter.eval(df_results=df_results)

    from autogluon.common.utils.s3_utils import upload_s3_folder
    folder_to_upload = plotter.output_dir
    upload_s3_folder(bucket="tabarena", prefix="mitra_tabarena_results", folder_to_upload=folder_to_upload)
