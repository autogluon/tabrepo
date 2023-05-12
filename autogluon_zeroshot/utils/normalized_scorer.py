from typing import List
import numpy as np
import pandas as pd


class NormalizedScorer:
    def __init__(self,
                 df_results_by_dataset: pd.DataFrame,
                 datasets: List[str],
                 baseline: str = None,
                 metric_error_col: str = 'metric_error',
                 dataset_col: str = 'dataset',
                 framework_col: str = 'framework',
                 ):
        """
        :param df_results_by_dataset: Dataframe of method performance containing columns `metric_error_col`,
        `dataset_col` and `framework_col`.
        :param datasets: datasets to consider
        """
        assert all(col in df_results_by_dataset for col in [metric_error_col, dataset_col, framework_col])
        all_datasets = set(df_results_by_dataset[dataset_col].unique())
        for dataset in datasets:
            assert dataset in all_datasets, f"dataset {dataset} not present in passed evaluations"
        self.topline_dict = df_results_by_dataset.groupby("dataset").min()[metric_error_col].to_dict()
        if baseline is not None:
            assert baseline in df_results_by_dataset[framework_col].unique()
            self.baseline_dict = df_results_by_dataset[
                df_results_by_dataset[framework_col] == baseline
            ].groupby("dataset").min()[metric_error_col].to_dict()
        else:
            self.baseline_dict = df_results_by_dataset.groupby("dataset").median(numeric_only=True)[
                metric_error_col].to_dict()

    # TODO rename to score, create parent class
    def rank(self, dataset: str, error: float) -> float:
        baseline = self.baseline_dict[dataset]
        topline = self.topline_dict[dataset]
        res = (error - topline) / np.clip(baseline - topline, a_min=1e-5, a_max=None)
        return np.clip(res, 0, 1)
