from __future__ import annotations

import pandas as pd

from autogluon.tabular import TabularDataset
from tabrepo.tabarena.tabarena import TabArena


frameworks_rename = {
    "AG_bq_M1910B_DSL_4h8c_2023_10_12_purucker": "AutoGluon 1.0 Preview (Best, 4h8c)",
    "AutoGluon_benchmark_4h8c_gp3_amlb_2023": "AutoGluon 0.8 (Best, 4h8c)",
    "AutoGluon_benchmark_1h8c_gp3_amlb_2023": "AutoGluon 0.8 (Best, 1h8c)",
    "AutoGluon_hq_4h8c_gp3_amlb_2023": "AutoGluon 0.8 (High, 4h8c)",
    "AutoGluon_hq_1h8c_gp3_amlb_2023": "AutoGluon 0.8 (High, 1h8c)",
    "AutoGluon_bq_4h8c_2022_03_25": "AutoGluon 0.4 (Best, 4h8c)",
    "AutoGluon_bq_1h8c_2022_03_25": "AutoGluon 0.4 (Best, 1h8c)",
    "AutoGluon_bq_ds_N100_4h8c_2023_11_08_tabrepo": "AutoGluon 1.0 Preview (Best+ZS, 4h8c)",
    "AutoGluon_bq_ds_N100_1h8c_2023_11_08_tabrepo": "AutoGluon 1.0 Preview (Best+ZS, 1h8c)",
    "AutoGluon_bestquality_1h8c_2021_02_06_v0_1_0": "AutoGluon 0.1 (Best, 1h8c)",
    "AutoGluon_benchmark_1h8c_gp3_amlb_2022": "AutoGluon 0.3 (Best, 1h8c)",
    # "AutoGluon_bq_1h8c_2023_11_03": "AutoGluon v1.0 Preview (Best, 1h8c)",
    "AutoGluon_bq_DSL_1h8c_2023_11_03": "AutoGluon 1.0 Preview (Best, 1h8c)",

    "flaml_1h8c_gp3_amlb_2023": "FLAML (2023, 1h8c)",
    "mljarsupervised_benchmark_1h8c_gp3_amlb_2023": "MLJAR (2023, 1h8c)",
    "H2OAutoML_1h8c_gp3_amlb_2023": "H2OAutoML (2023, 1h8c)",
    "lightautoml_1h8c_gp3_amlb_2023": "lightautoml (2023, 1h8c)",
    "autosklearn_1h8c_gp3_amlb_2023": "autosklearn (2023, 1h8c)",
    "GAMA_benchmark_1h8c_gp3_amlb_2023": "GAMA (2023, 1h8c)",
    "TPOT_1h8c_gp3_amlb_2023": "TPOT (2023, 1h8c)",
    "TunedRandomForest_1h8c_gp3_amlb_2023": "TunedRandomForest (2023, 1h8c)",
    "RandomForest_1h8c_gp3_amlb_2023": "RandomForest (2023, 1h8c)",
    "constantpredictor_1h8c_gp3_amlb_2023": "constantpredictor (2023, 1h8c)",

    "flaml_4h8c_gp3_amlb_2023": "FLAML (2023, 4h8c)",
    "mljarsupervised_benchmark_4h8c_gp3_amlb_2023": "MLJAR (2023, 4h8c)",
    "H2OAutoML_4h8c_gp3_amlb_2023": "H2OAutoML (2023, 4h8c)",
    "lightautoml_4h8c_gp3_amlb_2023": "lightautoml (2023, 4h8c)",
    "autosklearn_4h8c_gp3_amlb_2023": "autosklearn (2023, 4h8c)",
    "GAMA_benchmark_4h8c_gp3_amlb_2023": "GAMA (2023, 4h8c)",
    "TPOT_4h8c_gp3_amlb_2023": "TPOT (2023, 4h8c)",
    "TunedRandomForest_4h8c_gp3_amlb_2023": "TunedRandomForest (2023, 4h8c)",
    "RandomForest_4h8c_gp3_amlb_2023": "RandomForest (2023, 4h8c)",
    "constantpredictor_4h8c_gp3_amlb_2023": "constantpredictor (2023, 4h8c)",

    "AutoGluon_bq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (Best, 4h8c)",
    "AutoGluon_gq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (Good, 4h8c)",
    "AutoGluon_hq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c)",
    "AutoGluon_hq_v1_il0001_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.001)",
    "AutoGluon_hq_v1_il00005_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0005)",
    "AutoGluon_hq_v1_il00001_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0001)",
    "AutoGluon_hq_v1_il000005_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.00005)",
    "AutoGluon_benchmark_4h8c_gp3_amlb_2022": "AutoGluon 0.3.1 (Best, 4h8c)",

    "AutoGluon_CatBoost_4h8c_2023_12_07": "CatBoost (2023, 4h8c)",
    "AutoGluon_LightGBM_4h8c_2023_12_07": "LightGBM (2023, 4h8c)",
    "AutoGluon_XGBoost_4h8c_2023_12_07": "XGBoost (2023, 4h8c)",

    "AutoGluon_bq_1h8c_2023_11_29": "AutoGluon 1.0 (Best, 1h8c)",
    "AutoGluon_bq_30m8c_2023_11_29": "AutoGluon 1.0 (Best, 30m8c)",
    "AutoGluon_bq_10m8c_2023_11_29": "AutoGluon 1.0 (Best, 10m8c)",
    "AutoGluon_bq_5m8c_2023_11_29": "AutoGluon 1.0 (Best, 5m8c)",

    "AutoGluon_HQIL_60min": "AutoGluon 1.2 (Fast, 1h8c)",
    "AutoGluon_HQ_60min": "AutoGluon 1.2 (High, 1h8c)",
    "AutoGluon_benchmark_60min": "AutoGluon 1.2 (Best, 1h8c)",

}


if __name__ == '__main__':
    amlb2025_prefix = "s3://neerick-autogluon/tabarena/benchmarks/amlb2025/"

    results_files = [
        "amlb_all.csv"
    ]

    results_files = [f"{amlb2025_prefix}{result_file}" for result_file in results_files]

    results_dfs = [
        TabularDataset(result_file) for result_file in results_files
    ]

    results_files_old = [
        "s3://automl-benchmark-ag/aggregated/ec2/2021_02_06_v0_1_0/results_preprocessed.csv",  # v0.1.0
        # "s3://automl-benchmark-ag/aggregated/ec2/2022_03_25/results_preprocessed.csv",  # v0.4.0
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_26/results_preprocessed_min.parquet",  # v1.0.0
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_29/results_preprocessed_min.parquet",  # v1.0.0
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2023_preprocessed.parquet",
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2022_preprocessed.csv",
    ]

    df_results = pd.concat(results_dfs, ignore_index=True)
    df_results = df_results[~df_results["result"].isnull()]

    df_results["fold"] = df_results["fold"].astype(int)
    df_results["metric_error"] = -df_results["result"]
    optimum_result = df_results["metric"].map({
        "auc": 1.0,
        "neg_rmse": 0.0,
        "neg_logloss": 0.0,
    })
    df_results["metric_error"] = optimum_result - df_results["result"]

    df_results_old = pd.concat([TabularDataset(result_file) for result_file in results_files_old], ignore_index=True)
    df_results_old["result"] = -df_results_old["metric_error"]
    df_results_old["training_duration"] = df_results_old["time_train_s"]
    df_results_old["predict_duration"] = df_results_old["time_infer_s"]
    df_results_old["fold"] = df_results_old["fold"].astype(int)
    df_results_old["task"] = df_results_old["dataset"]

    df_results = pd.concat([
        df_results,
        df_results_old,
    ], ignore_index=True)

    df_results["metric"] = df_results["metric"].map({
        "auc": "roc_auc",
        "neg_rmse": "rmse",
        "neg_logloss": "logloss",
    })
    df_results["training_duration"] = df_results["training_duration"].fillna(df_results["duration"])
    df_results = df_results[~df_results["predict_duration"].isnull()]

    fillna_method = f"constantpredictor_60min"
    baseline_method = "RandomForest_60min"

    df_results["framework"] = df_results["framework"].map(frameworks_rename).fillna(df_results["framework"])

    default_methods = [
        "AutoGluon 0.1 (Best, 1h8c)",
        "AutoGluon 0.3 (Best, 1h8c)",
        # "AutoGluon 0.4 (Best, 1h8c)",
        "AutoGluon 0.8 (Best, 1h8c)",
        # "AutoGluon 1.0 (Best, 1h8c)",
        "AutoGluon 1.2 (Fast, 1h8c)",
        "AutoGluon 1.2 (High, 1h8c)",

        "AutoGluon 1.2 (Best, 1h8c)",
        # "AutoGluon_benchmark_10min",

        "constantpredictor_60min",
        "RandomForest_60min",
        "TunedRandomForest_60min",
    ]

    method_suffix = "_60min"

    banned_methods = [
        f"FEDOT{method_suffix}",
        f"NaiveAutoML{method_suffix}",
        f"autosklearn2{method_suffix}",
        f"GAMA{method_suffix}",
        f"TPOT{method_suffix}",
    ]

    df_results = df_results[(df_results["framework"].str.contains(method_suffix)) | (df_results["framework"].isin(default_methods))]
    df_results = df_results[~df_results["framework"].isin(banned_methods)]
    df_results = df_results[df_results["framework"] != fillna_method]

    arena = TabArena(
        method_col="framework",
        task_col="task",
        error_col="metric_error",
        columns_to_agg_extra=[
            "training_duration",
            "predict_duration",
        ],
        seed_column="fold",
    )

    # df_fillna = df_results[df_results[arena.method_col] == fillna_method]
    # df_fillna = df_fillna.drop(columns=[arena.method_col])
    df_fillna = None
    df_results_fillna = arena.fillna_data(data=df_results, df_fillna=df_fillna, fillna_method="worst")
    # df_results_fillna = df_results
    # df_results_fillna = df_results_fillna[df_results_fillna["framework"] != fillna_method]

    print(len(df_results_fillna[arena.task_col].unique()))

    results_agg = arena.leaderboard(
        data=df_results_fillna,
        # include_error=True,
        include_elo=True,
        include_failure_counts=True,
        include_mrr=True,
        include_rank_counts=True,
        include_winrate=True,
        elo_kwargs={
            "calibration_framework": baseline_method,
            "calibration_elo": 1000,
            "BOOTSTRAP_ROUNDS": 100,
        },
        baseline_relative_error=baseline_method,
        relative_error_kwargs={"agg": "gmean"},
    )
    print(results_agg)

    results_per_task = arena.compute_results_per_task(data=df_results_fillna)
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/autogluon2025/critical-diagram.png")
    arena.plot_critical_diagrams(results_per_task=results_per_task, save_path=f"./figures/autogluon2025/critical-diagram.pdf")

    results_per_task_rename = results_per_task.rename(columns={
        arena.method_col: "framework",
        arena.task_col: "dataset",
        "training_duration": "time_train_s",
        "predict_duration": "time_infer_s",
        arena.error_col: "metric_error",
    })

    from autogluon_benchmark.plotting.plotter import Plotter
    plotter = Plotter(
        results_ranked_df=results_per_task_rename,
        results_ranked_fillna_df=results_per_task_rename,
        save_dir=f"./figures/autogluon2025"
    )

    plotter.plot_all(
        calibration_framework=baseline_method,
        calibration_elo=1000,
    )
