from __future__ import annotations

import pandas as pd

from autogluon.common.loaders import load_pd
from tabrepo.loaders._results import preprocess_configs


def convert_to_dataset_fold(df: pd.DataFrame, column_name="dataset", column_type=str) -> pd.DataFrame:
    """
    Converts "{dataset}_{fold}" dataset column to (dataset, fold) columns.

    Parameters
    ----------
    df: pandas DataFrame of benchmark results
        The DataFrame must have a column named `dataset` in the form "{dataset}_{fold}".
        Example: "adult_5"
    column_name: str, default = "dataset"
        The name of the column to create alongside "fold".
    column_type: type, default = str
        The dtype of the column to create alongside "fold".

    Returns
    -------
    Pandas DataFrame of benchmark results with the dataset column split into (dataset, fold) columns

    """
    df = df.copy(deep=True)
    df["fold"] = df["dataset"].apply(lambda x: x.rsplit("_", 1)[1]).astype(int)
    df[column_name] = df["dataset"].apply(lambda x: x.rsplit("_", 1)[0]).astype(column_type)
    df_columns = list(df.columns)
    df_columns = [column_name, "fold"] + [c for c in df_columns if c not in [column_name, "dataset", "fold"]]
    return df[df_columns]  # Reorder so dataset and fold are the first two columns.


"""
Updates the original input files to the new input files format.
This should not be necessary to run in future, but is present for reference on how the conversion was done.
"""
if __name__ == '__main__':
    # Download repository from S3 and cache it locally for re-use in future calls
    # context: BenchmarkContext = get_context("D244_F3_C1416")
    # benchmark_paths = context.benchmark_paths

    raw = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21/leaderboard_preprocessed_configs.csv"
    results_by_dataset = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21/evaluation/configs/results_ranked_by_dataset_all.csv"
    comparison = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21/evaluation/compare/results_ranked_by_dataset_valid.csv"
    # raw = benchmark_paths.configs
    # results_by_dataset = benchmark_paths.results_by_dataset
    comparison_raw = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21_dummy/baselines_raw.csv"

    df_comparison_raw = load_pd.load(comparison_raw)
    df_comparison_raw["tid"] = df_comparison_raw["tid"].astype(int)

    df_raw = load_pd.load(raw)
    df_raw["tid"] = df_raw["tid"].astype(int)
    df_results_by_dataset = load_pd.load(results_by_dataset)
    df_comparison = load_pd.load(comparison)
    # FIXME: HACK
    if "fold" not in df_results_by_dataset.columns:
        df_results_by_dataset_v2 = convert_to_dataset_fold(df_results_by_dataset, column_name="tid", column_type=int)
        # df_results_by_dataset["dataset"] = df_results_by_dataset["dataset"].astype(int)
    if "fold" not in df_comparison.columns:
        df_comparison_v2 = convert_to_dataset_fold(df_comparison, column_name="tid", column_type=int)

    df_raw_v2 = df_raw.copy(deep=True)
    df_raw_v2 = preprocess_configs(df_raw_v2)

    tid_to_dataset_dict = df_raw_v2[["tid", "dataset"]].drop_duplicates().set_index("tid").squeeze().to_dict()
    tid_to_metric_dict = df_raw_v2[["tid", "metric"]].drop_duplicates().set_index("tid").squeeze().to_dict()

    df_results_by_dataset_v2["dataset"] = df_results_by_dataset_v2["tid"].map(tid_to_dataset_dict)
    df_comparison_v2["dataset"] = df_comparison_v2["tid"].map(tid_to_dataset_dict)
    df_results_by_dataset_v2["metric"] = df_results_by_dataset_v2["tid"].map(tid_to_metric_dict)
    df_comparison_v2["metric"] = df_comparison_v2["tid"].map(tid_to_metric_dict)

    df_comparison_v3 = df_comparison_v2[[
        "dataset",
        "fold",
        "framework",
        "metric_error",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]

    df_results_by_dataset_v3 = df_results_by_dataset_v2[[
        "dataset",
        "fold",
        "framework",
        "metric_error",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]

    df_raw_v3 = df_raw_v2[[
        "dataset",
        "tid",
        "fold",
        "framework",
        "metric_error",
        "metric_error_val",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]

    df_raw_v3 = df_raw_v3.sort_values(by=["dataset", "fold", "framework"])
    df_results_by_dataset_v3 = df_results_by_dataset_v3.sort_values(by=["dataset", "fold", "framework"])
    df_comparison_v3 = df_comparison_v3.sort_values(by=["dataset", "fold", "framework"])

    df_raw_v4 = df_raw_v3.drop(columns=["time_infer_s"]).merge(df_results_by_dataset_v3[["dataset", "fold", "framework", "time_infer_s"]], on=["dataset", "fold", "framework"])

    df_results_by_dataset_v3 = df_results_by_dataset_v3.drop(columns=["metric_error"]).merge(df_raw_v4[["dataset", "fold", "framework", "metric_error"]], on=["dataset", "fold", "framework"])

    df_raw_v4 = df_raw_v4.reset_index(drop=True)
    df_results_by_dataset_v3 = df_results_by_dataset_v3.reset_index(drop=True)
    df_comparison_v3 = df_comparison_v3.reset_index(drop=True)
    
    print(df_raw_v4.drop(columns=["metric_error_val"]).equals(df_results_by_dataset_v3))

    df_comparison_v4 = df_comparison_v3.copy(deep=True)
    df_comparison_v4 = df_comparison_v4.drop(columns=["metric_error"]).merge(df_comparison_raw[["dataset", "fold", "framework", "metric_error"]],
                                                                             on=["dataset", "fold", "framework"])

    df_comparison_v4 = df_comparison_v4[[
        "dataset",
        "fold",
        "framework",
        "metric_error",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]

    assert len(df_comparison_v4) == len(df_comparison_v3)

    from autogluon.common.savers import save_pd
    configs_suffix = "configs.parquet"
    baselines_suffix = "baselines.parquet"

    save_pd.save(path=configs_suffix, df=df_raw_v4, compression="snappy")
    save_pd.save(path=baselines_suffix, df=df_comparison_v4, compression="snappy")

    # s3_prefix = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21/"
    # save_pd.save(path=f"{s3_prefix}{configs_suffix}", df=df_raw_v4, compression="snappy")
    # save_pd.save(path=f"{s3_prefix}{baselines_suffix}", df=df_comparison_v4, compression="snappy")
