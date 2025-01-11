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

"""
TabPFNv2 VS all
                       framework  Winrate   >   <   =  % Loss Reduction  % Loss Reduction (median)  Avg Fit Speed Diff  Avg Inf Speed Diff      rank  loss_rescaled  time_train_s  time_infer_s
0                       TabPFNv2  0.50000   0   0  50          0.000000                   0.000000            0.000000            0.000000  4.291667       0.186162      0.934110      0.097907
1   AutoGluon_bq_4h8c_2023_11_14  0.57000  28  21   1         14.110429                   1.917624        -1960.237690            9.251874  5.135417       0.247669   2063.745833      0.028471
2    lightautoml_4h8c_2023_11_14  0.68000  34  16   0         17.359928                   4.928494        -4298.520399            8.563472  5.385417       0.255190   4213.825000      0.014373
3      H2OAutoML_4h8c_2023_11_14  0.70000  35  15   0         19.085679                  13.324732       -14432.318875          106.100702  6.802083       0.339628  13044.322917      0.002040
4             CatBoost_c1_BAG_L1  0.74000  37  13   0         25.706583                  10.328370          -69.269554          887.651422  6.947917       0.338563     93.875805      0.000217
5    autosklearn_4h8c_2023_11_14  0.73000  36  13   1         19.228122                  12.791765       -16020.402483          150.558859  7.135417       0.370768  14413.591667      0.002555
6               TabPFN_c1_BAG_L1  0.71875  34  13   1         25.071025                   9.138129           -1.089452            3.965683  7.604167       0.393965      1.869860      0.020259
7          flaml_4h8c_2023_11_14  0.76000  38  12   0         27.639940                  15.156316       -15604.101477          589.218612  7.916667       0.383986  14040.468750      0.000247
8       NeuralNetTorch_c1_BAG_L1  0.72000  36  14   0         26.903051                  13.106830           -5.495663          146.402236  8.250000       0.484219      6.060828      0.000863
9      NeuralNetFastAI_c1_BAG_L1  0.77000  38  11   1         28.385360                  14.775004           -3.315155          113.758083  8.375000       0.476137      3.930714      0.001109
10            LightGBM_c1_BAG_L1  0.79000  39  10   1         28.758478                  17.424386            0.003896          425.042064  8.593750       0.422179      1.316792      0.000423
11             XGBoost_c1_BAG_L1  0.84000  42   8   0         29.972264                  17.330113           -0.003084          117.374998  8.989583       0.448378      1.164495      0.001515
12        RandomForest_c1_BAG_L1  0.79000  39  10   1         33.749791                  25.824886            1.482995          180.453868  9.677083       0.553550      0.385799      0.000600
13          ExtraTrees_c1_BAG_L1  0.80000  40  10   0         34.411778                  19.729834            1.597654          176.087238  9.895833       0.566863      0.367525      0.000624
"""
