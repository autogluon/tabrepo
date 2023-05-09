from typing import List

import pandas as pd

from .configuration_list_scorer import ConfigurationListScorer
from .simulation_context import ZeroshotSimulatorContext


class SingleBestConfigScorer(ConfigurationListScorer):
    def __init__(self,
                 df_results_by_dataset: pd.DataFrame,
                 datasets: List[str] = None,
                 score_col: str = 'rank',
                 score_val_col: str = 'score_val',
                 model_col: str = 'framework',
                 dataset_col: str = 'dataset'):
        """
        Enables to score the best configuration from a given list of configuration.
        The configuration selected is the one with the lowest validation score is selected and its test-score
        is returned.
        :param df_results_by_dataset: dataframe with results on base models on each dataset/fold.
        :param datasets: list of datasets/folds formatted as `['359987_8', '359933_3', ...]`
        :param score_col:
        :param score_val_col:
        :param model_col:
        :param dataset_col:
        """
        super(SingleBestConfigScorer, self).__init__(datasets=datasets)

        assert all(col in df_results_by_dataset for col in [score_col, score_val_col, model_col, dataset_col])
        self.score_col = score_col
        self.score_val_col = score_val_col
        self.model_col = model_col
        self.dataset_col = dataset_col
        if datasets is not None:
            df_results_by_dataset = df_results_by_dataset[
                df_results_by_dataset[dataset_col].isin(datasets)]
        self.df_results_by_dataset = df_results_by_dataset
        self.datasets = list(self.df_results_by_dataset[dataset_col].unique())
        self.df_pivot_val = self.df_results_by_dataset.pivot_table(index=self.model_col, columns=self.dataset_col, values=self.score_val_col)

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        return cls(
            df_results_by_dataset=zeroshot_simulator_context.df_results_by_dataset_vs_automl,
            **kwargs,
        )

    def get_best_validation_configs_df(self, configs: list) -> pd.DataFrame:
        best_val_model_series = self.df_pivot_val.loc[configs].idxmax(axis=0).to_frame(name=self.model_col)
        best_val_model_by_dataset_df = self.df_results_by_dataset.merge(best_val_model_series, on=[self.dataset_col, self.model_col])
        return best_val_model_by_dataset_df

    def score_per_dataset(self, configs: List[str], score_col=None) -> dict:
        if score_col is None:
            score_col = self.score_col
        best_val_model_by_dataset_df = self.get_best_validation_configs_df(configs=configs)
        return best_val_model_by_dataset_df[[self.dataset_col, score_col]].set_index(self.dataset_col).squeeze().to_dict()

    def score(self, configs: List[str]) -> float:
        """
        :param configs: list of configuration to select from. The test score of the configuration with the best
        validation score is returned.
        :return: the test-error selected with validation scores making sure that the test scores of each model are not
        used for the selection.
        """
        best_val_model_by_dataset_df = self.get_best_validation_configs_df(configs=configs)
        # this is the error without knowing the test score of each model and oracle picking the best,
        # instead using validation score to pick best
        avg_error_real = best_val_model_by_dataset_df[self.score_col].mean()
        return avg_error_real

    def compute_errors(self, configs: list):
        errors = self.score_per_dataset(score_col='metric_error', configs=configs)
        return errors

    def subset(self, datasets: List[str]) -> "SingleBestConfigScorer":
        """
        :param datasets:
        :return: a scorer only considering the datasets passed as argument which can be used to evaluate performance
        on hold-out datasets.
        """
        return self.__class__(
            df_results_by_dataset=self.df_results_by_dataset,
            datasets=datasets,
            score_col=self.score_col,
            score_val_col=self.score_val_col,
            model_col=self.model_col,
            dataset_col=self.dataset_col,
        )

