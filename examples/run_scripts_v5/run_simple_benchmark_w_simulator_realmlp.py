from __future__ import annotations

import pandas as pd

from experiment_utils import ExperimentBatchRunner
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabrepo.scripts_v5.AutoGluon_class import AGWrapper
from tabrepo.scripts_v5.ag_models.realmlp_model import RealMLPModel


# To re-use the pre-computed results if you have the file "tabrepo_artifacts_realmlp_20250201.zip":
#  cd {this_dir}
#  unzip tabrepo_artifacts_realmlp_20250201.zip
# Note: This file is currently located in "s3://tabrepo/artifacts/methods/realmlp/tabrepo_artifacts_realmlp_20250201.zip"
#  Not publicly available
# You can regenerate this artifact from scratch by running the code. On a 192 CPU core machine, this will take approximately 25 hours.
# If the artifact is present, it will be used and the models will not be re-run.
if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_200"  # 200 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_simple_simulator"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    # TODO: in future shouldn't require downloading all repo_og preds (100+ GB) before running experiments
    #  Should only need preds for ensembling part, but not for comparisons
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    # Sample for a quick demo
    # datasets = repo_og.datasets()[:3]
    # folds = [0]

    # To run everything:
    datasets = repo_og.datasets()
    folds = repo_og.folds

    # TODO: Why is RealMLP slow when running sequentially / not in a bag? Way slower than it should be. Torch threads?
    methods = [
        (
            "RealMLP_c1_BAG_L1_v4_noes_r0",  # 2025/02/01 num_cpus=192
            AGWrapper,
            {
                "fit_kwargs": {
                    "num_bag_folds": 8,
                    "num_bag_sets": 1,
                    "fit_weighted_ensemble": False,
                    "calibrate": False,
                    "verbosity": 2,
                    "hyperparameters": {
                        RealMLPModel: {
                            "random_state": 0,
                            "use_early_stopping": False,
                        },
                    },
                }
            },
        ),
    ]

    tids = [repo_og.dataset_to_tid(dataset) for dataset in datasets]
    # TODO: Make lazy load of dataset in case all experiments are complete for a given dataset
    repo: EvaluationRepository = ExperimentBatchRunner().generate_repo_from_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        task_metadata=repo_og.task_metadata,
        ignore_cache=ignore_cache,
        convert_time_infer_s_from_batch_to_sample=True,
    )

    # TODO: repo.configs_type should not be None for custom methods
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
        "RealMLP_c1_BAG_L1_v4_noes_r0",  # did 600 runs (200x3)
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
