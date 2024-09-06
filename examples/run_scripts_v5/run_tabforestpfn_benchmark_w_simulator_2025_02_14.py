from __future__ import annotations

import pandas as pd

from tabrepo import load_repository, EvaluationRepository
from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGWrapper
from tabrepo.benchmark.models.ag import TabPFNV2Model, TabPFNV2ClientModel
from tabrepo.benchmark.experiment.experiment_utils import run_experiments, convert_leaderboard_to_configs
from tabrepo.utils.cache import CacheFunctionPickle
from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample

from script_utils import load_ag11_bq_baseline


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_200"  # 100 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_tabforestpfn_simulator"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo: EvaluationRepository = load_repository(context_name, cache=True)

    # a = repo.configs(tasks=[("balance-scale", 0), ("adasda", 2)])

    # Subset to tasks supported by TabPFNv2
    task_metadata = repo.task_metadata.copy(deep=True)
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 10000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 2000]  # FIXME TMP
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 5000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 9000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] >= 2000]
    task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 500]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]

    datasets = list(task_metadata["dataset"])
    # datasets = datasets[:50]  # Capping to 50 because TabPFNv2 runs into daily limit with more

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]
    # task_metadata = task_metadata[task_metadata["NumberOfClasses"] >= 2]
    # task_metadata = task_metadata[task_metadata["NumberOfClasses"] < 2]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 1000]
    datasets = list(task_metadata["dataset"])

    # datasets = datasets[:91] + datasets[92:]

    # TabPFNv2 fails on these datasets for unknown reasons
    banned_datasets = ["Indian_pines", "topo_2_1"]

    datasets = [d for d in datasets if d not in banned_datasets]

    datasets = datasets[:44]

    folds = [0, 1, 2]

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]

    # To run everything:
    # datasets = repo.datasets()
    # folds = repo.folds

    methods = [
        ("TabPFNv2_N4_BAG_L1", AGWrapper, {"fit_kwargs": {
            "num_bag_folds": 8, "num_bag_sets": 1, "fit_weighted_ensemble": False, "calibrate": False,
            "hyperparameters": {TabPFNV2ClientModel: [{"n_estimators": 4}]},
        }}),
        # ("TabPFNv2Local_N4_BAG_L1", AGWrapper, {"fit_kwargs": {
        #     "num_bag_folds": 8, "num_bag_sets": 1, "fit_weighted_ensemble": False, "calibrate": False,
        #     "hyperparameters": {TabPFNV2Model: [{"n_estimators": 4}]},
        # }}),
        ("TabPFNv2LocalParallel_N4_BAG_L1", AGWrapper, {"fit_kwargs": {
            "num_bag_folds": 8, "num_bag_sets": 1, "fit_weighted_ensemble": False, "calibrate": False,
            "hyperparameters": {TabPFNV2Model: [{"n_estimators": 4, "ag_args_ensemble": {"fold_fitting_strategy": "parallel_local"}}]},
        }}),
    ]

    # FIXME: experiment_cls, cache_true/false, etc.
    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        cache_cls=CacheFunctionPickle,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_baselines = [result["df_results"] for result in results_lst if result["simulation_artifacts"] is None]

    if results_baselines:
        df_baselines = pd.concat(results_baselines, ignore_index=True)
        df_baselines = convert_time_infer_s_from_batch_to_sample(df_baselines, repo=repo)
    else:
        df_baselines = None

    results_configs = [result for result in results_lst if result["simulation_artifacts"] is not None]

    results_lst_simulation_artifacts = [result["simulation_artifacts"] for result in results_configs]
    results_lst_df = [result["df_results"] for result in results_configs]
    results_lst_df = [convert_leaderboard_to_configs(df) for df in results_lst_df]  # TODO: Remove later, keeping to make old runs compatible with new runs

    df_configs = pd.concat(results_lst_df, ignore_index=True)

    # TODO: Remove later, keeping to make old runs compatible with new runs
    df_configs["metric"] = df_configs["metric"].map({
        "root_mean_squared_error": "rmse",
    }).fillna(df_configs["metric"])

    df_configs = convert_time_infer_s_from_batch_to_sample(df_configs, repo=repo)

    df_processed_ag12_2024 = load_ag11_bq_baseline(datasets=datasets, folds=folds, repo=repo)
    df_baselines = pd.concat([df_baselines, df_processed_ag12_2024], ignore_index=True)

    df_configs = df_configs.drop(columns=["tid"])
    df_baselines = df_baselines.drop(columns=["tid"])

    repo_2: EvaluationRepository = EvaluationRepository.from_raw(
        df_configs=df_configs,
        df_baselines=df_baselines,
        results_lst_simulation_artifacts=results_lst_simulation_artifacts,
        task_metadata=task_metadata,
    )

    save_loc = "tabforestpfn_sim_v2"
    repo_2.to_dir(path=save_loc)

    print(f"New Configs   : {repo_2.configs()}")

    # FIXME: Allow picking ensemble based on test score, to see the difference in weights

    from tabrepo import EvaluationRepositoryCollection
    repo_combined = EvaluationRepositoryCollection(repos=[repo, repo_2], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = repo_combined.subset(datasets=repo_2.datasets(), folds=repo_2.folds)

    # FIXME: repo_combined._zeroshot_context.df_metrics contains 200 datasets when it should contain only 110

    configs = repo_combined.configs()

    repo_combined.print_info()

    # FIXME: Add boxplot of overfitting, basically, val vs test, percent loss delta
    a = repo_combined.metrics()

    configs_og = repo.configs()

    # result_ens_og, result_ens_og_weights = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs_og, ensemble_size=25, rank=False)
    #
    # weights_og = result_ens_og_weights.mean(axis=0).sort_values(ascending=False)
    # print("weights_og")
    # print(weights_og)
    #
    # result_ens, result_ens_weights = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs, ensemble_size=25, rank=False)
    # weights = result_ens_weights.mean(axis=0).sort_values(ascending=False)
    # print("weights")
    # print(weights)

    # result_ens_og_cheat, result_ens_og_weights_cheat = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs_og, ensemble_size=25, rank=False, ensemble_kwargs={"cheater": True})
    # result_ens_cheat, result_ens_weights_cheat = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs, ensemble_size=25, rank=False, ensemble_kwargs={"cheater": True})

    # weights_og = result_ens_og_weights_cheat.mean(axis=0).sort_values(ascending=False)
    # print("weights_og cheater")
    # print(weights_og)
    #
    # weights = result_ens_weights_cheat.mean(axis=0).sort_values(ascending=False)
    # print("weights cheater")
    # print(weights)

    # result_ens_og = result_ens_og.reset_index()
    # result_ens_og["framework"] = "ALL"
    # result_ens = result_ens.reset_index()
    # result_ens["framework"] = "ALL_PLUS_TabPFN"

    # result_ens_og_cheat = result_ens_og_cheat.reset_index()
    # result_ens_og_cheat["framework"] = "ALL_CHEAT"
    # result_ens_cheat = result_ens_cheat.reset_index()
    # result_ens_cheat["framework"] = "ALL_PLUS_TabForestPFN_CHEAT"

    # results_df_2 = pd.concat([
    #     result_ens,
    #     result_ens_og,
    #     # df_processed_ag12_2024,
    #     # result_ens_cheat,
    #     # result_ens_og_cheat,
    # ], ignore_index=True)

    # results_df_2 = convert_time_infer_s_from_sample_to_batch(results_df_2, repo=repo_combined)

    # print(f"AVG OG: {result_ens_og[0].mean()}")
    # print(f"AVG: {result_ens[0].mean()}")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_configs.head(100))

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        # "TabPFN_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",

        "TabPFNv2_N4_BAG_L1",
        # "TabPFNv2Local_N4_BAG_L1",
        "TabPFNv2LocalParallel_N4_BAG_L1",
    ]

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]

    from tabrepo.evaluation.evaluator import Evaluator

    evaluator = Evaluator(repo_combined)

    metrics = evaluator.compare_metrics(
        # results_df,
        # results_df_2,
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )

    metrics_tmp = metrics.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")
    evaluator_kwargs = {
        "treat_folds_as_datasets": True,
        "frameworks_compare_vs_all": [
            "TabPFNv2_N4_BAG_L1",
        ],
        "frameworks_rename": {
            "AutoGluon_bq_4h8c_2023_11_14": "AutoGluon 0.8 (4h8c)",
        },
        # "frameworks_compare_vs_all": ["TabPFNv2"],
    }
    evaluator_output = evaluator.plot_overall_rank_comparison(
        results_df=metrics,
        evaluator_kwargs=evaluator_kwargs,
        save_dir=expname,
        calibration_framework="RandomForest_c1_BAG_L1",
    )
