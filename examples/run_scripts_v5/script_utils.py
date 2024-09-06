import copy

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon_benchmark.preprocessing.amlb_preprocessor import AMLBPreprocessor
from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample
from tabrepo.repository.abstract_repository import AbstractRepository


def load_ag11_bq_baseline(datasets: list[str], folds: list[int], repo: AbstractRepository) -> pd.DataFrame:
    ag12_raw = load_pd.load(f"s3://automl-benchmark-ag/aggregated/ec2/2024_10_25/results.csv")

    df_processed_ag12_2024: pd.DataFrame = AMLBPreprocessor(framework_suffix="2024_10_25").transform(df=ag12_raw)
    df_processed_ag12_2024 = df_processed_ag12_2024[df_processed_ag12_2024["framework"] == "AutoGluon_bq_mainline_4h8c_2024_10_25"]
    df_processed_ag12_2024 = df_processed_ag12_2024[df_processed_ag12_2024["dataset"].isin(datasets)]
    df_processed_ag12_2024 = df_processed_ag12_2024[df_processed_ag12_2024["fold"].isin(folds)]

    df_processed_ag12_2024["metric"] = df_processed_ag12_2024["metric"].map({
        "auc": "roc_auc",
        "neg_logloss": "log_loss",
    })

    baseline_fillna = "AutoGluon_bq_4h8c_2023_11_14"
    baseline_df = copy.deepcopy(repo._zeroshot_context.df_baselines)
    baseline_df = baseline_df.drop(columns=["task"])
    baseline_df = baseline_df[baseline_df["framework"] == baseline_fillna]

    df_processed_ag12_2024 = convert_time_infer_s_from_batch_to_sample(df_processed_ag12_2024, repo=repo)
    df_processed_ag12_2024_ref = df_processed_ag12_2024.set_index(["dataset", "fold"])

    fillna_rows = []
    for dataset in datasets:
        for fold in folds:
            if (dataset, fold) not in df_processed_ag12_2024_ref.index:
                print(dataset, fold)
                fillna_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["fold"] == fold)]
                print(fillna_row)
                assert len(fillna_row) == 1
                fillna_rows.append(fillna_row)

    if fillna_rows:
        fillna_rows = pd.concat(fillna_rows, axis=0, ignore_index=True)
        fillna_rows["framework"] = "AutoGluon_bq_mainline_4h8c_2024_10_25"
        df_processed_ag12_2024 = pd.concat([df_processed_ag12_2024, fillna_rows], ignore_index=True)
    df_processed_ag12_2024 = df_processed_ag12_2024[baseline_df.columns]

    df_processed_ag12_2024["framework"] = "AutoGluon_bq_4h8c_2024_10_25"

    return df_processed_ag12_2024
