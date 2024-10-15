from __future__ import annotations

import copy
import os

import pandas as pd

from tabrepo import load_repository, EvaluationRepository
from tabrepo.scripts_v5.TabPFN_class import CustomTabPFN
from tabrepo.scripts_v5.TabPFNv2_class import CustomTabPFNv2
from tabrepo.scripts_v5.TabForestPFN_class import CustomTabForestPFN
from tabrepo.scripts_v5.AutoGluon_class import CustomAutoGluon
from tabrepo.scripts_v5.LGBM_class import CustomLGBM
from tabrepo.scripts_v5.ag_models.tabforestpfn_model import TabForestPFNModel
from experiment_utils import run_experiments, convert_leaderboard_to_configs
from experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabrepo.utils.cache import SimulationExperiment


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_200"  # 100 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_tabforestpfn_simulator"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo: EvaluationRepository = load_repository(context_name, cache=True)

    # Subset to tasks supported by TabPFNv2
    task_metadata = repo.task_metadata.copy(deep=True)
    task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= 10000]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] >= 2000]
    task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 500]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]

    datasets = list(task_metadata["dataset"])
    # datasets = datasets[:50]  # Capping to 50 because TabPFNv2 runs into daily limit with more

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] >= 2]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 1000]
    datasets = list(task_metadata["dataset"])

    folds = [0, 1, 2]

    # datasets = [
    #     "blood-transfusion-service-center",  # binary
    #     "Australian",  # binary
    #     "balance-scale",  # multiclass
    #     # "MIP-2016-regression",  # regression
    # ]

    # To run everything:
    # datasets = repo.datasets
    # folds = repo.folds

    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]

    methods_dict = {
        # # "LightGBM": {},  # Dummy example model
        # # "TabPFN": {},  # Doesn't support regression
        # "TabPFNv2": {},
        # # "TabForestPFN_N32_E50": {"n_ensembles": 32, "max_epochs": 50},
        # # "TabForestPFN_N1_E10": {"n_ensembles": 1, "max_epochs": 10},
        # "TabForestPFN_N32_E10": {"n_ensembles": 32, "max_epochs": 10},
        # # "TabForestPFN_N1_E0": {"n_ensembles": 1, "max_epochs": 0},
        # # "TabForestPFN_N32_E0": {"n_ensembles": 32, "max_epochs": 0},
        # "TabForestPFN_N1_E0_nosplit": {"n_ensembles": 1, "max_epochs": 0, "split_val": False},
        # "TabForestPFN_N32_E0_nosplit": {"n_ensembles": 32, "max_epochs": 0, "split_val": False},
        # "TabForest_N32_E0_nosplit": {
        #     "n_ensembles": 32, "max_epochs": 0, "split_val": False,
        #     "path_weights": '/home/ubuntu/workspace/tabpfn_weights/tabforest.pt'
        # },
        # "TabPFN_N32_E0_nosplit": {
        #     "n_ensembles": 32, "max_epochs": 0, "split_val": False,
        #     "path_weights": '/home/ubuntu/workspace/tabpfn_weights/tabpfn.pt'
        # },
        # "LightGBM_c1_BAG_L1_V2": {"fit_kwargs": {
        #     "num_bag_folds": 8, "fit_weighted_ensemble": False, "num_bag_sets": 1, "calibrate": False,
        #     "hyperparameters": {"GBM": [{}]},
        # }},
        "TabForestPFN_N4_E10_BAG_L1": {"fit_kwargs": {
            "num_bag_folds": 8, "fit_weighted_ensemble": False, "num_bag_sets": 1, "calibrate": False,
            "hyperparameters": {TabForestPFNModel: [{"n_ensembles": 4, "max_epochs": 10}]},
        }},
    }
    method_cls_dict = {
        "LightGBM": CustomLGBM,
        "TabPFN": CustomTabPFN,
        "TabPFNv2": CustomTabPFNv2,
        "TabForestPFN_N32_E50": CustomTabForestPFN,
        "TabForestPFN_N1_E10": CustomTabForestPFN,
        "TabForestPFN_N32_E10": CustomTabForestPFN,
        "TabForestPFN_N1_E0": CustomTabForestPFN,
        "TabForestPFN_N32_E0": CustomTabForestPFN,
        "TabForestPFN_N1_E0_nosplit": CustomTabForestPFN,
        "TabForestPFN_N32_E0_nosplit": CustomTabForestPFN,
        "TabForest_N32_E0_nosplit": CustomTabForestPFN,
        "TabPFN_N32_E0_nosplit": CustomTabForestPFN,
        "LightGBM_c1_BAG_L1_V2": CustomAutoGluon,
        "TabForestPFN_N4_E10_BAG_L1": CustomAutoGluon,
    }
    methods = list(methods_dict.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        methods_dict=methods_dict,
        method_cls=method_cls_dict,
        experiment_cls=OOFExperimentRunner,
        cache_cls=SimulationExperiment,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_lst_simulation_artifacts = [result["simulation_artifacts"] for result in results_lst]
    results_lst_df = [result["df_results"] for result in results_lst]

    results_df = pd.concat(results_lst_df, ignore_index=True)
    results_df = convert_leaderboard_to_configs(results_df)

    results_df["metric"] = results_df["metric"].map({
        "root_mean_squared_error": "rmse",
    }).fillna(results_df["metric"])

    num_rows_test_dict = {}

    # FIXME: Don't require all results in memory at once
    simulation_artifacts_full = {}
    for simulation_artifacts in results_lst_simulation_artifacts:
        for k in simulation_artifacts.keys():
            if k not in simulation_artifacts_full:
                simulation_artifacts_full[k] = {}
            for f in simulation_artifacts[k]:
                if f not in simulation_artifacts_full[k]:
                    simulation_artifacts_full[k][f] = copy.deepcopy(simulation_artifacts[k][f])
                else:
                    for method in simulation_artifacts[k][f]["pred_proba_dict_val"]:
                        if method in simulation_artifacts_full[k][f]["pred_proba_dict_val"]:
                            raise AssertionError(f"Two results exist for tid {k}, fold {f}, method {method}!")
                        else:
                            simulation_artifacts_full[k][f]["pred_proba_dict_val"][method] = simulation_artifacts[k][f]["pred_proba_dict_val"][method]
                            simulation_artifacts_full[k][f]["pred_proba_dict_test"][method] = simulation_artifacts[k][f]["pred_proba_dict_test"][method]

    for d in simulation_artifacts_full:
        for f in simulation_artifacts_full[d]:
            num_rows_test_dict[(d, f)] = len(list(simulation_artifacts_full[d][f]["pred_proba_dict_test"].values())[0])

    from autogluon.common.utils.simulation_utils import convert_simulation_artifacts_to_tabular_predictions_dict
    zeroshot_pp, zeroshot_gt = convert_simulation_artifacts_to_tabular_predictions_dict(simulation_artifacts=simulation_artifacts_full)

    save_loc = "./tabforestpfn_sim/"
    save_loc = os.path.abspath(save_loc)
    save_loc_data_dir = save_loc + "model_predictions/"

    from tabrepo.predictions import TabularPredictionsInMemory
    from tabrepo.simulation.ground_truth import GroundTruth
    from autogluon.common.savers import save_pd
    from tabrepo.contexts.context import BenchmarkContext, construct_context
    from tabrepo.contexts.subcontext import BenchmarkSubcontext

    predictions = TabularPredictionsInMemory.from_dict(zeroshot_pp)
    ground_truth = GroundTruth.from_dict(zeroshot_gt)
    predictions.to_data_dir(data_dir=save_loc_data_dir)
    ground_truth.to_data_dir(data_dir=save_loc_data_dir)

    df_configs = convert_leaderboard_to_configs(leaderboard=results_df)

    # FIXME: Hack to get time_infer_s correct, instead we should keep time_infer_s to the original and transform it internally to be per row
    # FIXME: Keep track of number of rows of train/test per task internally in Repository
    tmp = df_configs[["dataset", "fold"]].apply(tuple, axis=1)
    df_configs["time_infer_s"] = df_configs["time_infer_s"] / tmp.map(num_rows_test_dict)

    save_pd.save(path=f"{save_loc}configs.parquet", df=df_configs)

    context: BenchmarkContext = construct_context(
        name="tabforestpfn_sim",
        datasets=datasets,
        folds=folds,
        local_prefix=save_loc,
        local_prefix_is_relative=False,
        has_baselines=False)
    subcontext = BenchmarkSubcontext(parent=context)
    repo_2: EvaluationRepository = subcontext.load_from_parent()
    # FIXME: infer_time is incorrect for TabForestPFN
    # FIXME: infer_time is incorrect for ALL and ALL_PLUS_TabForestPFN during eval

    from tabrepo import EvaluationRepositoryCollection
    repo_combined = EvaluationRepositoryCollection(repos=[repo, repo_2], config_fallback="ExtraTrees_c1_BAG_L1")

    repo_combined = repo_combined.subset(datasets=repo_2.datasets())

    # FIXME: repo_combined._zeroshot_context.df_metrics contains 200 datasets when it should contain only 110

    configs = repo_combined.configs()

    configs_og = repo.configs()

    result_ens_og, result_ens_og_weights = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs_og, ensemble_size=25, rank=False)

    weights_og = result_ens_og_weights.mean(axis=0).sort_values(ascending=False)
    print("weights_og")
    print(weights_og)

    result_ens, result_ens_weights = repo_combined.evaluate_ensembles(datasets=repo_combined.datasets(), configs=configs, ensemble_size=25, rank=False)
    weights = result_ens_weights.mean(axis=0).sort_values(ascending=False)
    print("weights")
    print(weights)

    result_ens_og = result_ens_og.reset_index()
    result_ens_og["framework"] = "ALL"
    result_ens = result_ens.reset_index()
    result_ens["framework"] = "ALL_PLUS_TabForestPFN"

    results_df_2 = pd.concat([result_ens, result_ens_og], ignore_index=True)

    # print(f"AVG OG: {result_ens_og[0].mean()}")
    # print(f"AVG: {result_ens[0].mean()}")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df.head(100))

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "TabPFN_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
        "TabForestPFN_N4_E10_BAG_L1",
    ]

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
        "H2OAutoML_4h8c_2023_11_14",
        "flaml_4h8c_2023_11_14",
        "lightautoml_4h8c_2023_11_14",
        "autosklearn_4h8c_2023_11_14",
    ]

    metrics = repo_combined.compare_metrics(
        # results_df,
        results_df_2,
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")
    evaluator_kwargs = {
        # "frameworks_compare_vs_all": ["TabPFNv2"],
    }
    evaluator_output = repo_combined.plot_overall_rank_comparison(
        results_df=metrics,
        evaluator_kwargs=evaluator_kwargs,
        save_dir=expname,
    )
