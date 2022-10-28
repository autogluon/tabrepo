from typing import List

import pandas as pd


def get_rank(error: float, error_lst: List[float], higher_is_better: bool = False) -> float:
    rank = 1
    for e in error_lst:
        if error == e:
            rank += 0.5
        elif higher_is_better:
            if error < e:
                rank += 1
        else:
            if error > e:
                rank += 1
    return rank


class RankScorer:
    def __init__(self,
                 df_results_by_dataset: pd.DataFrame,
                 datasets: List[str],
                 metric_error_col: str = 'metric_error',
                 dataset_col: str = 'dataset'):
        self.error_dict  = {}
        for i, dataset in enumerate(datasets):
            automl_error_list = sorted(list(df_results_by_dataset[df_results_by_dataset[dataset_col] == dataset][metric_error_col]))
            self.error_dict[dataset] = automl_error_list

    def rank(self, dataset: str, error: float) -> float:
        rank = get_rank(error, self.error_dict[dataset])
        return rank
