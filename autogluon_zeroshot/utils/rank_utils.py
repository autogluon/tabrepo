from typing import List

import pandas as pd


def get_rank(error: float, error_lst: List[float], higher_is_better: bool = False, ties_win=False, pct=False) -> float:
    """
    If ties_win=True, rank will equal a win if tied with an error in error_lst. (ex: rank 1.0)
    If ties_win=False, rank will equal a tie if tied with an error in error_lst. (ex: rank 1.5)

    If pct=True, rescales output to be between 0 and 1, with 0 = best, 1 = worst.
    """
    rank = 0
    for e in error_lst:
        if error == e:
            # tie
            if ties_win:
                pass  # count as a win
            else:
                rank += 0.5  # count as a tie
        elif higher_is_better:
            if error < e:
                rank += 1
        else:
            if error > e:
                rank += 1

    if pct:
        rank /= len(error_lst)
    else:
        rank += 1
    return rank


class RankScorer:
    def __init__(self,
                 df_results_by_dataset: pd.DataFrame,
                 datasets: List[str],
                 metric_error_col: str = 'metric_error',
                 dataset_col: str = 'dataset',
                 framework_col: str = 'framework',
                 ties_win: bool = False,
                 pct: bool = False,
                 ):
        """
        :param df_results_by_dataset: Dataframe of method performance containing columns `metric_error_col`,
        `dataset_col` and `framework_col`.
        :param datasets: datasets to consider
        :param ties_win: whether ties count as a win (True) or a tie (False). Set False to ensure symmetric equivalence.
        :param pct: whether to display the returned rankings in percentile form.
        """
        assert all(col in df_results_by_dataset for col in [metric_error_col, dataset_col, framework_col])
        all_datasets = set(df_results_by_dataset[dataset_col].unique())
        for dataset in datasets:
            assert dataset in all_datasets, f"dataset {dataset} not present in passed evaluations"
        self.ties_win = ties_win
        self.pct = pct
        df_pivot = df_results_by_dataset.pivot_table(values=metric_error_col, index=dataset_col, columns=framework_col)
        df_pivot.values.sort(axis=1)
        self.error_dict = {dataset: df_pivot.loc[dataset] for dataset in datasets}

    def rank(self, dataset: str, error: float) -> float:
        """
        Get the rank of a result on a dataset given an error.
        """
        return get_rank(error=error, error_lst=self.error_dict[dataset], ties_win=self.ties_win, pct=self.pct)
