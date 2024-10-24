from __future__ import annotations

from typing import Tuple, List, Union
from autogluon_benchmark import OpenMLTaskWrapper
from tabrepo import EvaluationRepository

class ContextDataLoader(OpenMLTaskWrapper):
    """
    Class to Fetch Train Test Splits of context dataset
    """
    def get_context_train_test_split(self, repo: EvaluationRepository, task_id: Union[int, List[int]], repeat: int = 0,
                                     fold: int = 0, sample: int = 0):
        if repo.tid_to_dataset(task_id) in repo.datasets():
            train_indices, test_indices = self.task.get_train_test_split_indices(repeat=repeat, fold=fold,
                                                                                 sample=sample)
            X_train = self.X.loc[train_indices]
            y_train = self.y[train_indices]
            X_test = self.X.loc[test_indices]
            y_test = self.y[test_indices]
            return X_train, y_train, X_test, y_test
        else:
            raise KeyError(f"Dataset for task_id {task_id} not found.")

    # Add Another function to just get the X and y for random state