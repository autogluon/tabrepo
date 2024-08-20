from __future__ import annotations
import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation_repository import EvaluationRepository
from ..predictions.tabular_predictions import TabularModelPredictions
from ..simulation.configuration_list_scorer import ConfigurationListScorer
from ..simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.single_best_config_scorer import SingleBestConfigScorer


# TODO: WIP. This is not a fully functional class yet.
class EvaluationRepositoryCollection:
    """
    Simple Repository class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.
    """
    def __init__(
            self,
            repos: list[EvaluationRepository],
            config_fallback: str = None,
    ):
        """
        :param config_fallback: if specified, used to replace the result of a configuration that is missing, if not
        specified an error is thrown when querying a config that does not exist. A cheap baseline such as the result
        of a mean predictor can be used for the fallback.
        """
        self.repos: list[EvaluationRepository] = repos
        self.set_config_fallback(config_fallback)

        # TODO: Create AbstractRepository to avoid code-dupe
        # TODO: raise exception if overlap in (dataset, fold, config)
        # TODO: implement config_fallback
        # TODO: Merge ground_truth -> Easy
        # TODO: Merge zeroshot_context -> Hard
        # TODO: Merge tabular_predictions -> Hard
        #  Mostly done, need to implement _construct_ensemble_selection_config_scorer

    def configs(self):
        raise NotImplementedError

    # TODO: Optimize
    def goes_where(self, dataset: str, fold: int, config: str) -> int | None:
        """
        Returns the repo idx containing the specified config in a given dataset fold. Returns None if no such repo exists.
        """
        for i, repo in enumerate(self.repos):
            if dataset in repo.datasets():
                task = repo.task_name(dataset=dataset, fold=fold)
                configs = repo.configs(tasks=[task])
                if config in configs:
                    return i
        return None

    def set_config_fallback(self, config_fallback: str = None):
        if config_fallback:
            assert config_fallback in self.configs()
        self._config_fallback = config_fallback

    def predict_test_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        return self._predict_multi(predict_func="predict_test_multi", dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=binary_as_multiclass)

    def predict_val_multi(self, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        return self._predict_multi(predict_func="predict_val_multi", dataset=dataset, fold=fold, configs=configs, binary_as_multiclass=binary_as_multiclass)

    def _predict_multi(self, predict_func: str, dataset: str, fold: int, configs: List[str] = None, binary_as_multiclass: bool = False) -> np.ndarray:
        if configs is None:
            configs = self.configs()

        repo_map = defaultdict(list)
        repo_idx_map = defaultdict(list)
        predict_map = dict()
        config_idx_lst = []
        for i, config in enumerate(configs):
            repo_idx = self.goes_where(dataset=dataset, fold=fold, config=config)
            repo_map[repo_idx].append(config)
            repo_idx_map[repo_idx].append(i)
            config_idx_lst.append(repo_idx)

        for repo_idx, config_lst in repo_map.items():
            f = getattr(self.repos[repo_idx], predict_func)
            predict_map[repo_idx] = f(dataset=dataset, fold=fold, configs=config_lst, binary_as_multiclass=binary_as_multiclass)

        shape = predict_map[repo_idx].shape[1:]
        predict_multi = np.zeros(shape=(len(configs),) + shape, dtype=predict_map[repo_idx].dtype)

        for repo_idx, predict in predict_map.items():
            predict_multi[repo_idx_map[repo_idx], :] = predict

        return predict_multi

    def _construct_ensemble_selection_config_scorer(self,
                                                    ensemble_size: int = 10,
                                                    backend='ray',
                                                    **kwargs) -> EnsembleSelectionConfigScorer:
        raise NotImplementedError

    def _construct_single_best_config_scorer(self, **kwargs) -> SingleBestConfigScorer:
        raise NotImplementedError

    def force_to_dense(self, inplace: bool = False, verbose: bool = True):
        raise NotImplementedError
