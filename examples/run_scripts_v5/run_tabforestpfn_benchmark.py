from __future__ import annotations

import pandas as pd

from tabrepo import load_repository, EvaluationRepository
from tabrepo.scripts_v5.TabPFN_class import CustomTabPFN
from tabrepo.scripts_v5.TabPFNv2_class import CustomTabPFNv2
from tabrepo.scripts_v5.TabForestPFN_class import CustomTabForestPFN
from tabrepo.scripts_v5.AutoGluon_class import AGWrapper
from tabrepo.scripts_v5.LGBM_class import CustomLGBM
from tabrepo.scripts_v5.ag_models.tabforestpfn_model import TabForestPFNModel
from experiment_utils import run_experiments, convert_leaderboard_to_configs

if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_200"  # 100 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./initial_experiment_tabpfn_v2"  # folder location of all experiment artifacts
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
    # datasets = datasets[:50]

    task_metadata = task_metadata[task_metadata["dataset"].isin(datasets)]
    task_metadata = task_metadata[task_metadata["NumberOfClasses"] >= 2]
    # task_metadata = task_metadata[task_metadata["NumberOfFeatures"] <= 100]
    # task_metadata = task_metadata[task_metadata["NumberOfInstances"] > 1000]
    datasets = list(task_metadata["dataset"])

    folds = [0]

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
        "TabForestPFN_N32_E10": {"n_ensembles": 32, "max_epochs": 10},
        # # "TabForestPFN_N1_E0": {"n_ensembles": 1, "max_epochs": 0},
        # # "TabForestPFN_N32_E0": {"n_ensembles": 32, "max_epochs": 0},
        # "TabForestPFN_N1_E0_nosplit": {"n_ensembles": 1, "max_epochs": 0, "split_val": False},
        "TabForestPFN_N32_E0_nosplit": {"n_ensembles": 32, "max_epochs": 0, "split_val": False},
        "TabForest_N32_E0_nosplit": {
            "n_ensembles": 32, "max_epochs": 0, "split_val": False,
            "path_weights": '/home/ubuntu/workspace/tabpfn_weights/tabforest.pt'
        },
        "TabPFN_N32_E0_nosplit": {
            "n_ensembles": 32, "max_epochs": 0, "split_val": False,
            "path_weights": '/home/ubuntu/workspace/tabpfn_weights/tabpfn.pt'
        },
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
        "LightGBM_c1_BAG_L1_V2": AGWrapper,
        "TabForestPFN_N4_E10_BAG_L1": AGWrapper,
    }
    methods = list(methods_dict.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        methods_dict=methods_dict,
        method_cls=method_cls_dict,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_df = pd.concat(results_lst, ignore_index=True)
    results_df = convert_leaderboard_to_configs(results_df)

    results_df["metric"] = results_df["metric"].map({
        "root_mean_squared_error": "rmse",
    }).fillna(results_df["metric"])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

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
    ]

    metrics = repo.compare_metrics(
        results_df,
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")
    evaluator_kwargs = {
        # "frameworks_compare_vs_all": ["TabPFNv2"],
    }
    evaluator_output = repo.plot_overall_rank_comparison(
        results_df=metrics,
        evaluator_kwargs=evaluator_kwargs,
        save_dir=expname,
    )
