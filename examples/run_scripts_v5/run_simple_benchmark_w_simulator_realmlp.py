from __future__ import annotations

import pandas as pd

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner

from tabrepo.benchmark.models.ag import RealMLPModel


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

    # df_out = get_feature_info(repo_og)
    #
    # a = df_out[("int", ("bool",))]
    # print(a)
    # b = a[a > 0]
    # datasets_with_bool = list(b.index)
    #
    # # Sample for a quick demo
    datasets = repo_og.datasets()
    # datasets = repo_og.datasets(problem_type="regression")
    # datasets_filter = repo_og.datasets(problem_type="binary") + repo_og.datasets(problem_type="multiclass")
    # datasets = [d for d in datasets if d in datasets_filter] + repo_og.datasets(problem_type="regression")
    # # datasets = datasets[:173]
    # datasets_og = datasets
    # datasets = [d for d in datasets_og if d in datasets_with_bool] + [d for d in datasets_og if d not in datasets_with_bool]

    # datasets = repo_og.datasets(problem_type="regression")
    # datasets = datasets[:6]  # FIXME: ImputeF crashes on GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1 fold 0
    folds = [0, 1, 2]
    # datasets = ["Internet-Advertisements"]

    # To run everything:
    # datasets = repo_og.datasets()
    # folds = repo_og.folds

    # TODO: Why is RealMLP slow when running sequentially / not in a bag? Way slower than it should be. Torch threads?
    methods = [
        AGModelBagExperiment(  # 2025/02/01 num_cpus=192, pytabkit==1.1.3
            name="RealMLP_c1_BAG_L1_v4_noes_r0",
            model_cls=RealMLPModel,
            model_hyperparameters={
                "random_state": 0,
                "use_early_stopping": False,
                "use_ls": None,
                "bool_to_cat": False,
                "impute_bool": True,
            },
        ),
        AGModelBagExperiment(  # 2025/02/01 num_cpus=192, pytabkit==1.1.3
            name="RealMLP_c2_BAG_L1_AutoLS",
            model_cls=RealMLPModel,
            model_hyperparameters={
                "random_state": 0,
                "use_early_stopping": False,
                "use_ls": "auto",
                "bool_to_cat": False,
                "impute_bool": True,
            },
        ),
        AGModelBagExperiment(  # 2025/02/01 num_cpus=192, pytabkit==1.1.3
            name="RealMLP_c2_BAG_L1_AutoLS_AUCStop",
            model_cls=RealMLPModel,
            model_hyperparameters={
                "random_state": 0,
                "use_early_stopping": False,
                "use_ls": "auto",
                "bool_to_cat": False,
                "impute_bool": True,
                # "use_roc_auc_to_stop": True,
            },
        ),
        AGModelBagExperiment(  # 2025/02/07 num_cpus=192, pytabkit==1.2.1
            name="RealMLP_c2_BAG_L1_AutoLS_AUCStop_boolcat_impF_naT",
            model_cls=RealMLPModel,
            model_hyperparameters={
                "random_state": 0,
                "use_early_stopping": False,
                "use_ls": "auto",
                # "use_roc_auc_to_stop": True,
                "bool_to_cat": True,
                "impute_bool": False,
                "name_categories": True,
            },
        ),
        AGModelBagExperiment(  # 2025/02/12 num_cpus=192, pytabkit==1.2.1
            name="RealMLP_c2_BAG_L1_TD",
            model_cls=RealMLPModel,
            model_hyperparameters={
                "random_state": 0,
                "use_early_stopping": False,
                "use_ls": "auto",
                # "use_roc_auc_to_stop": True,
                "bool_to_cat": True,
                "impute_bool": False,
                "name_categories": True,
                # "td_s_reg": False,
            },
        ),
        # AGModelBagExperiment(  # 2025/03/05 num_cpus=192, pytabkit==1.2.1
        #     name="RealMLP_c1_BAG_L1",
        #     model_cls=RealMLPModel,
        #     model_hyperparameters={},
        # ),
    ]

    exp_batch_runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=repo_og.task_metadata,
        cache_path_format="task_first",
    )

    # results_lst = exp_batch_runner.load_results(
    #     methods=methods,
    #     datasets=datasets,
    #     folds=folds,
    # )

    results_lst = exp_batch_runner.run(
        methods=methods,
        datasets=datasets,
        folds=folds,
        ignore_cache=ignore_cache,
    )

    repo = exp_batch_runner.repo_from_results(results_lst=results_lst)

    # TODO: repo.configs_type should not be None for custom methods
    repo.print_info()

    save_path = "repo_new"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    print(f"New Configs   : {repo.configs()}")

    shared_datasets = [d for d in repo.datasets(union=False) if d in repo_og.datasets()]

    # repo_tabforestpfn = EvaluationRepository.from_dir(path="tabforestpfn_sim")
    # shared_datasets = [d for d in shared_datasets if d in repo_tabforestpfn.datasets(union=False)]
    # repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo, repo_tabforestpfn], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = repo_combined.subset(datasets=shared_datasets)
    repo_combined.set_config_fallback("ExtraTrees_c1_BAG_L1")
    evaluator = Evaluator(repo=repo_combined)

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
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_121",
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_121_bool_to_cat",
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_121_bool_to_cat_false",

        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_121_bool_to_cat2",
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_121_bool_to_cat_false2",
        # "RealMLP_c2_BAG_L1_AutoLS_bool_to_cat",
        # "RealMLP_c2_BAG_L1_AutoLS_impute_true",
        # "RealMLP_c2_BAG_L1_AutoLS_bool_to_cat_impute_true",

        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_boolcat_imputeT",  # FIXME: CRASHES?
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_boolcat_imputeF",  # FIXME: CRASHES?

        # "RealMLP_c1_BAG_L1_v4_noes_r0",  # did 600 runs (200x3)
        # "RealMLP_c2_BAG_L1_AutoLS",  # 200x3
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop",  # 200x3
        "RealMLP_c2_BAG_L1_AutoLS_AUCStop_boolcat_impF_naT",  # 200x3
        # "RealMLP_c2_BAG_L1_AutoLS_AUCStop_impF_naT",  # 175x3
        "RealMLP_c2_BAG_L1_TD",  # 200x3
        # "RealMLP_c1_BAG_L1",
    ]

    evaluator.compute_avg_config_prediction_delta(configs=comparison_configs + ["EBM_BAG_L1", "TabPFNv2_N4_BAG_L1", "TabPFN_Mix7_600000_N4_E30_FIX_BAG_L1"])

    # comparison_configs += [
    #     "TabForestPFN_N4_E10_BAG_L1",
    #     "TabForestPFN_N4_E30_BAG_L1",
    #     "TabForestPFN_N4_E50_BAG_L1",
    #     "TabForestPFN_N1_E10_S4_BAG_L1",
    #     "TabPFN_Mix7_500000_N4_E30_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E30_BAG_L1",
    #     "TabPFN_Mix7_300000_N4_E30_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E30_S4_BAG_L1",
    #     "TabPFNv2_N4_BAG_L1",
    #     "EBM_BAG_L1",
    #     "TabPFN_Mix7_600000_N1_E0_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E0_BAG_L1",
    #     "LightGBM_c1_BAG_L1_V2",
    #     "TabDPT_N1_E0_BAG_L1",
    #     "TabRMix7_500000_N1_E0_BAG_L1",
    #     "TabRMix7_500000_N4_E30_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E30_FIX_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E50_FIX_BAG_L1",
    #     "TabPFN_Mix7_600000_N4_E30_FIX_BAG_L1_COMPARISON",
    # ]



    df_ensemble_results, df_ensemble_weights = repo_combined.evaluate_ensembles(configs=comparison_configs, ensemble_size=40)
    df_ensemble_results = df_ensemble_results.reset_index()
    df_ensemble_results["framework"] = "ensemble_with_RealMLP_c1"

    df_ensemble_results_og, df_ensemble_weights_og = repo_combined.evaluate_ensembles(configs=comparison_configs_og, ensemble_size=40)
    df_ensemble_results_og = df_ensemble_results_og.reset_index()
    df_ensemble_results_og["framework"] = "ensemble_og"

    # from script_utils import load_ag11_bq_baseline
    # df_processed_ag11_2024 = load_ag11_bq_baseline(datasets=repo_combined.datasets(), folds=repo_combined.folds, repo=repo_combined)

    repo_og.set_config_fallback("ExtraTrees_c1_BAG_L1")
    df_zeroshot_portfolio_og = evaluator.zeroshot_portfolio(configs=repo_og.configs())
    df_zeroshot_portfolio_og["framework"] = "zeroshot_og"

    df_zeroshot_portfolio_w_realmlp = evaluator.zeroshot_portfolio(configs=repo_combined.configs())
    df_zeroshot_portfolio_w_realmlp["framework"] = "zeroshot_w_realmlp"

    df_zeroshot_portfolio_w_realmlp_single = evaluator.zeroshot_portfolio(configs=repo_og.configs() + ["RealMLP_c2_BAG_L1_TD"])
    df_zeroshot_portfolio_w_realmlp_single["framework"] = "zeroshot_w_realmlp_single"

    df_zeroshot_portfolio_w_realmlp_2 = evaluator.zeroshot_portfolio(configs=repo_og.configs() + ["RealMLP_c2_BAG_L1_TD", "RealMLP_c2_BAG_L1_AutoLS_AUCStop_boolcat_impF_naT"])
    df_zeroshot_portfolio_w_realmlp_2["framework"] = "zeroshot_w_realmlp_2"

    df_zeroshot_portfolio_w_realmlp_n5 = evaluator.zeroshot_portfolio(configs=repo_combined.configs(), n_portfolios=10)
    df_zeroshot_portfolio_w_realmlp_n5["framework"] = "zeroshot_w_realmlp_n10"

    df_zeroshot_portfolio_n5 = evaluator.zeroshot_portfolio(configs=repo_og.configs(), n_portfolios=10)
    df_zeroshot_portfolio_n5["framework"] = "zeroshot_og_n10"

    results_df = pd.concat([
        df_ensemble_results,
        df_ensemble_results_og,
        # df_processed_ag11_2024,
        df_zeroshot_portfolio_og,
        df_zeroshot_portfolio_w_realmlp,
        df_zeroshot_portfolio_w_realmlp_single,
        df_zeroshot_portfolio_w_realmlp_2,
        df_zeroshot_portfolio_w_realmlp_n5,
        df_zeroshot_portfolio_n5,
    ], ignore_index=True)

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
        # "AutoGluon_bq_4h8c_2024_10_25",
    ]

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
            "frameworks_compare_vs_all": [
                "RealMLP_c2_BAG_L1_TD",
                "zeroshot_w_realmlp",
            ],
        },
    )
