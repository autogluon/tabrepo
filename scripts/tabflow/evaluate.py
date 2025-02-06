from __future__ import annotations

import argparse
import yaml
import pandas as pd
import json

from tabrepo.utils.experiment_utils import ExperimentBatchRunner
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabrepo.scripts_v5.AutoGluon_class import AGWrapper
from tabrepo.scripts_v5.ag_models.realmlp_model import RealMLPModel

# If the artifact is present, it will be used and the models will not be re-run.
if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--method_name', type=str, required=True, help="Name of the method")
    parser.add_argument('--wrapper_class', type=str, required=True, help="Wrapper class for the method")
    parser.add_argument('--fit_kwargs', type=str, required=True, help="Fit kwargs for the method in JSON format")

    args = parser.parse_args()

    # Debugging: Print raw arguments
    print("Raw arguments received:")
    print(f"datasets: {args.datasets}")
    print(f"folds: {args.folds}")
    print(f"method_name: {args.method_name}")
    print(f"wrapper_class: {args.wrapper_class}")
    print(f"fit_kwargs: {args.fit_kwargs}")

    # Load Context
    context_name = "D244_F3_C1530_30"  # 30 Datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_simple_simulator"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    # Sample for a quick demo
    # datasets = ["Australian", "blood-transfusion-service-center"]
    # folds = [0, 1]

    # To run everything:
    # datasets = repo_og.datasets()
    # folds = repo_og.folds

    datasets = args.datasets
    if -1 in args.folds:
        folds = repo_og.folds  # run on all folds
    else:
        folds = args.folds

    # Parse fit_kwargs from JSON string
    fit_kwargs = json.loads(args.fit_kwargs)
    for key, value in fit_kwargs['hyperparameters'].items():
        # TODO: Will not work if model class is given as string - like GBM in AGWrapper
        fit_kwargs['hyperparameters'][eval(key)] = fit_kwargs['hyperparameters'].pop(key)


    methods = [(args.method_name, eval(args.wrapper_class), {'fit_kwargs': fit_kwargs})]

    print(f"\nWrapper class: {args.wrapper_class}\n")
    print(f"Fit kwargs: {fit_kwargs}\n")

    tids = [repo_og.dataset_to_tid(dataset) for dataset in datasets]
    repo: EvaluationRepository = ExperimentBatchRunner().generate_repo_from_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        task_metadata=repo_og.task_metadata,
        ignore_cache=ignore_cache,
        convert_time_infer_s_from_batch_to_sample=True,
    )

    repo.print_info()

    save_path = "repo_new"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    print(f"New Configs   : {repo.configs()}")

    repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = repo_combined.subset(datasets=repo.datasets(), folds=repo.folds)

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

    comparison_configs = comparison_configs_og + [
        "RealMLP_c1_BAG_L1_v4_noes_r0",
    ]

    df_ensemble_results, df_ensemble_weights = repo_combined.evaluate_ensembles(configs=comparison_configs, ensemble_size=40)
    df_ensemble_results = df_ensemble_results.reset_index()
    df_ensemble_results["framework"] = "ensemble_with_RealMLP_c1"

    df_ensemble_results_og, df_ensemble_weights_og = repo_combined.evaluate_ensembles(configs=comparison_configs_og, ensemble_size=40)
    df_ensemble_results_og = df_ensemble_results_og.reset_index()
    df_ensemble_results_og["framework"] = "ensemble_og"

    results_df = pd.concat([
        df_ensemble_results,
        df_ensemble_results_og,
    ], ignore_index=True)

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]

    evaluator = Evaluator(repo=repo_combined)

    p = evaluator.plot_ensemble_weights(df_ensemble_weights=df_ensemble_weights, figsize=(16, 60))
    p.savefig("ensemble_weights_w_RealMLP_c1")

    metrics = evaluator.compare_metrics(
        results_df=results_df,
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )

    metrics_tmp = metrics.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")

    evaluator_output = evaluator.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=expname,
        evaluator_kwargs={
            "treat_folds_as_datasets": True,
            "frameworks_compare_vs_all": ["RealMLP_c1_BAG_L1_v4_noes_r0"],
        },
    )
