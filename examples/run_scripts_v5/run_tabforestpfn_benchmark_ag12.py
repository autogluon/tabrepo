from __future__ import annotations

import pandas as pd

from tabrepo import load_repository, EvaluationRepository
from tabrepo.scripts_v5.AutoGluon_class import AGWrapper
from tabrepo.scripts_v5.ag_models.ag_model import AGModelWrapper
from tabrepo.scripts_v5.ag_models.tabforestpfn_model import TabForestPFNModel
from tabrepo.scripts_v5.ag_models.ebm_model import ExplainableBoostingMachine
from tabrepo.scripts_v5.ag_models.tabpfn_v2_model import TabPFNV2Model
from tabrepo.scripts_v5.ag_models.tabdpt_model import TabDPTModel
from experiment_utils import run_experiments, convert_leaderboard_to_configs
from experiment_runner import OOFExperimentRunner
from tabrepo.utils.cache import SimulationExperiment
from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample, convert_time_infer_s_from_sample_to_batch

from script_utils import load_ag11_bq_baseline


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_200"  # 100 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_tabforestpfn_simulator_ag12"  # folder location of all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch

    repo: EvaluationRepository = load_repository(context_name, cache=True)

    # a = repo.configs(tasks=[("balance-scale", 0), ("adasda", 2)])

    # Subset to tasks supported by TabPFNv2
    task_metadata = repo.task_metadata.copy(deep=True)
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 1000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 9000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] >= 2000]
    task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]

    datasets = list(task_metadata["dataset"])
    # datasets = datasets[:50]  # Capping to 50 because TabPFNv2 runs into daily limit with more

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] >= 2]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 1000]
    datasets = list(task_metadata["dataset"])

    # datasets = datasets[:91] + datasets[92:]

    # TabPFNv2 fails on these datasets for unknown reasons
    banned_datasets = ["Indian_pines", "topo_2_1"]

    datasets = [d for d in datasets if d not in banned_datasets]

    datasets = datasets[:1]

    # datasets = ["Australian"]

    folds = [0]

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]

    # To run everything:
    # datasets = repo.datasets()
    # folds = repo.folds
    path_weights_tabpfn_real_mix7_500000 = '/home/ubuntu/workspace/tabpfn_weights/TabPFN_real_mix_7_models/model_step_500000.pt'

    path_weights_tabpfn_mix7_500000 = '/home/ubuntu/workspace/tabpfn_weights/TabPFN_mix_7_step_500000.pt'
    path_weights_tabpfn_mix7_600000 = '/home/ubuntu/workspace/tabpfn_weights/TabPFN_mix_7_step_600000.pt'
    path_weights_tabpfn_mix7_300000 = '/home/ubuntu/workspace/tabpfn_weights/TabPFN_mix_7_step_300000.pt'

    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    hyperparameters_best = get_hyperparameter_config(config_name="zeroshot")
    hyperparameters_mq = get_hyperparameter_config(config_name="default")

    import copy
    hyperparameters_best_tabpfn = copy.deepcopy(hyperparameters_best)

    hyperparameters_mq_tabpfn = copy.deepcopy(hyperparameters_mq)

    hyperparameters_best_tabpfn[TabForestPFNModel] = [
        {"n_ensembles": 1, "max_epochs": 0, "path_weights": path_weights_tabpfn_mix7_600000},
    ]

    hyperparameters_mq_tabpfn[TabForestPFNModel] = [
        {"n_ensembles": 1, "max_epochs": 0, "path_weights": path_weights_tabpfn_mix7_600000},
    ]

    hyperparameters_mq_tabpfn_N4_E30 = copy.deepcopy(hyperparameters_mq)
    # hyperparameters_mq_tabpfn_N4_E30 = {}

    hyperparameters_mq_tabpfn_N4_E30[TabForestPFNModel] = [
        # {"n_ensembles": 4, "max_epochs": 30, "path_weights": path_weights_tabpfn_mix7_600000},
        # {"n_ensembles": 1, "max_epochs": 0, "path_weights": path_weights_tabpfn_mix7_600000},
        # {"n_ensembles": 4, "max_epochs": 0, "path_weights": path_weights_tabpfn_mix7_600000},
        # {"n_ensembles": 4, "max_epochs": 30, "path_weights": path_weights_tabpfn_mix7_600000, "use_best_epoch": False},
    ]


    methods = [
        # ("AutoGluon_bq_parallel_TabPFNMix7_ZS", AGWrapper, {"fit_kwargs": {
        #     "presets": "best_quality",
        #     "hyperparameters": hyperparameters_best_tabpfn,
        #     "fit_strategy": "parallel",
        #     "time_limit": 3600,
        # }}),
        # ("AutoGluon_bq_parallel", AGWrapper, {"fit_kwargs": {
        #     "presets": "best_quality",
        #     "fit_strategy": "parallel",
        #     "time_limit": 3600,
        # }}),
        # ("AutoGluon_mq_parallel_TabPFNMix7_ZS", AGWrapper, {"fit_kwargs": {
        #     "hyperparameters": hyperparameters_mq_tabpfn,
        #     # "fit_strategy": "parallel",
        #     "time_limit": 3600,
        # }}),
        # ("AutoGluon_mq_parallel", AGWrapper, {"fit_kwargs": {
        #     "fit_strategy": "parallel",
        #     "time_limit": 3600,
        # }}),
        ("AutoGluon_mq_parallel_TabPFNMix7_N4_E30", AGWrapper, {"fit_kwargs": {
            "hyperparameters": hyperparameters_mq_tabpfn_N4_E30,
            "fit_strategy": "parallel",
            "refit_full": True,
            "time_limit": 3600,
        }}),
    ]

    # FIXME: experiment_cls, cache_true/false, etc.
    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]
    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        # experiment_cls=OOFExperimentRunner,
        cache_cls=SimulationExperiment,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_configs = []
    results_baselines = []
    for result in results_lst:
        if isinstance(result, dict):
            if result.get("simulation_artifacts", None) is None:
                results_baselines.append(result["df_results"])
            else:
                results_configs.append(result["df_results"])
        else:
            assert isinstance(result, pd.DataFrame)
            results_baselines.append(result)

    if results_baselines:
        df_baselines = pd.concat(results_baselines, ignore_index=True)
        df_baselines = convert_time_infer_s_from_batch_to_sample(df_baselines, repo=repo)
    else:
        df_baselines = None

    results_lst_simulation_artifacts = [result["simulation_artifacts"] for result in results_configs]
    results_lst_df = [result["df_results"] for result in results_configs]
    results_lst_df = [convert_leaderboard_to_configs(df) for df in results_lst_df]  # TODO: Remove later, keeping to make old runs compatible with new runs

    if results_lst_df:
        df_configs = pd.concat(results_lst_df, ignore_index=True)

        # TODO: Remove later, keeping to make old runs compatible with new runs
        df_configs["metric"] = df_configs["metric"].map({
            "root_mean_squared_error": "rmse",
        }).fillna(df_configs["metric"])

        df_configs = convert_time_infer_s_from_batch_to_sample(df_configs, repo=repo)
        df_configs = df_configs.drop(columns=["tid"])
    else:
        df_configs = None

    df_processed_ag12_2024 = load_ag11_bq_baseline(datasets=datasets, folds=folds, repo=repo)
    df_baselines = pd.concat([df_baselines, df_processed_ag12_2024], ignore_index=True)

    df_baselines = df_baselines.drop(columns=["tid"])

    if True:
        # FIXME: HACK
        df_baselines["metric_error_val"] = 0

    repo_2: EvaluationRepository = EvaluationRepository.from_raw(
        df_configs=df_baselines,
        df_baselines=df_baselines,
        results_lst_simulation_artifacts=results_lst_simulation_artifacts,
        task_metadata=task_metadata,
    )

    save_loc = "tabforestpfn_sim_ag12"
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

    if df_configs:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_configs.head(100))

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "TabPFN_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
    ]

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
        "H2OAutoML_4h8c_2023_11_14",
        "flaml_4h8c_2023_11_14",
        "lightautoml_4h8c_2023_11_14",
        "autosklearn_4h8c_2023_11_14",
        "AutoGluon_bq_mainline_4h8c_2024_10_25",
        "AutoGluon_mq_parallel_TabPFNMix7_ZS",
        "AutoGluon_mq_parallel",
        "AutoGluon_mq_parallel_TabPFNMix7_N4_E30",
    ]

    metrics = repo_combined.compare_metrics(
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
        "frameworks_compare_vs_all": [
            'AutoGluon 1.1 (4h8c)',
        ],
        "frameworks_rename": {
            "AutoGluon_bq_mainline_4h8c_2024_10_25": "AutoGluon 1.1 (4h8c)",
            "AutoGluon_bq_4h8c_2023_11_14": "AutoGluon 0.8 (4h8c)",
        },
        # "frameworks_compare_vs_all": ["TabPFNv2"],
    }
    evaluator_output = repo_combined.plot_overall_rank_comparison(
        results_df=metrics,
        evaluator_kwargs=evaluator_kwargs,
        save_dir=expname,
        calibration_framework="RandomForest_c1_BAG_L1",
    )
