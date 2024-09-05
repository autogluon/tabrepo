from __future__ import annotations

import copy
from collections import defaultdict
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Self

from .abstract_repository import AbstractRepository
from .ensemble_mixin import EnsembleMixin
from .ground_truth_mixin import GroundTruthMixin
from .evaluation_repository import EvaluationRepository
from ..contexts.utils import prune_zeroshot_gt
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext


# TODO: Improve error message for overlap
class EvaluationRepositoryCollection(AbstractRepository, EnsembleMixin, GroundTruthMixin):
    """
    Simple Repository class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.
    """
    def __init__(
        self,
        repos: list[EvaluationRepository | "EvaluationRepositoryCollection"],
        config_fallback: str = None,
        overlap: Literal["raise", "first", "last"] = "raise",
    ):
        self.overlap = overlap
        self.repos: list[EvaluationRepository | "EvaluationRepositoryCollection"] = repos

        zeroshot_context = merge_zeroshot([repo._zeroshot_context for repo in self.repos])
        self._ground_truth: GroundTruth = merge_ground_truth([repo._ground_truth for repo in self.repos])
        self._mapping = self._update_mapping()
        super().__init__(zeroshot_context=zeroshot_context, config_fallback=config_fallback)

    def _update_mapping(self):
        combinations = self._generate_dataset_fold_config_combinations(repos=self.repos)
        return self._combination_mapping_to_repo_index(combinations=combinations, overlap=self.overlap)

    def goes_where(self, dataset: str, fold: int, config: str) -> int | None:
        """
        Returns the repo idx containing the specified config in a given dataset fold. Returns None if no such repo exists.
        """
        return self._mapping.get((dataset, fold, config), None)

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
            if repo_idx is None:
                if self._config_fallback is None:
                    raise ValueError(f"The following combination is not present in this repository: (dataset='{dataset}', fold={fold}, config='{config}')")
                # get fallback
                repo_idx = self.goes_where(dataset=dataset, fold=fold, config=self._config_fallback)
                if repo_idx is None:
                    raise ValueError(
                        f"The following combination is not present in this repository: (dataset='{dataset}', fold={fold}, config='{config}')"
                        f"\nAdditionally, the fallback config is not present in (dataset='{dataset}', fold={fold}, config='{self._config_fallback}')"
                    )
                repo_map[repo_idx].append(self._config_fallback)
            else:
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

    # TODO: Make a better docstring, confusing what this `exactly` does
    def force_to_dense(self, inplace: bool = False, verbose: bool = True) -> Self:
        """
        Method to force the repository to a dense representation inplace.

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        :param inplace: If True, will perform logic inplace.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return dense repo if inplace=False or self after inplace updates in this call.
        """
        if not inplace:
            return copy.deepcopy(self).force_to_dense(inplace=True, verbose=verbose)

        datasets = self.datasets(union=True)
        df_dataset_folds = self._zeroshot_context.df_dataset_folds()

        n_folds = self.n_folds()
        n_folds_per_dataset = df_dataset_folds["dataset"].value_counts()
        datasets_dense = list(n_folds_per_dataset[n_folds_per_dataset == n_folds].index)

        # subset to only datasets that contain at least one result for all folds
        if set(datasets) != set(datasets_dense):
            self.subset(datasets=datasets_dense, inplace=inplace, force_to_dense=False)

        # subset to only configs that have results in all tasks
        configs = self.configs(union=True)
        configs_dense = self.configs(union=False)
        if set(configs) != set(configs_dense):
            self.subset(configs=configs_dense, inplace=inplace, force_to_dense=False)

        # subset all repos in collection so they do not contain invalid results
        for repo in self.repos:
            repo.subset(
                datasets=self.datasets(),
                configs=self.configs(),
                folds=self.folds,
                inplace=inplace,
                force_to_dense=False,
            )

        self._mapping = self._update_mapping()
        self._ground_truth = prune_zeroshot_gt(
            zeroshot_pred_proba=None,
            zeroshot_gt=self._ground_truth,
            dataset_to_tid_dict=self._dataset_to_tid_dict,
            verbose=verbose,
        )
        return self

    @staticmethod
    def _generate_dataset_fold_config_combinations(repos: list[EvaluationRepository]) -> list[list[Tuple[str, int, str]]]:
        """
        Returns the combinations (dataset, fold, config) for each repository
        """
        return [repo.dataset_fold_config_pairs() for repo in repos]

    @staticmethod
    def _combination_mapping_to_repo_index(
        combinations: list[list[Tuple[str, int, str]]],
        overlap: Literal["raise", "first", "last"] = "raise",
    ) -> dict[Tuple[str, int, str], int]:
        """
        Returns a dictionary mapping each (dataset, fold, config) to the repository index
        """
        if overlap == "first":
            len_combinations = len(combinations)
            mapping = {
                (dataset, fold, config): repo_index
                for repo_index in range(len_combinations-1, -1, -1)
                for (dataset, fold, config) in combinations[repo_index]
            }
        elif overlap in ["last", "raise"]:
            mapping = {
                (dataset, fold, config): repo_index
                for repo_index, repo_combinations in enumerate(combinations)
                for (dataset, fold, config) in repo_combinations
            }
            if overlap == "raise":
                len_combinations_total = 0
                for c in combinations:
                    len_combinations_total += len(c)
                if len_combinations_total != len(mapping):
                    # TODO: Improve error message
                    raise AssertionError(f"Overlap detected in provided repositories! (overlap='{overlap}')")
        else:
            raise ValueError(f"Unknown overlap value: '{overlap}'")
        return mapping


def merge_zeroshot(zeroshot_contexts: list[ZeroshotSimulatorContext], require_matching_flags: bool = False) -> ZeroshotSimulatorContext:
    assert isinstance(zeroshot_contexts, list)

    if any([z.df_baselines is not None for z in zeroshot_contexts]):
        df_baselines = pd.concat([z.df_baselines for z in zeroshot_contexts], ignore_index=True)
        df_baselines = df_baselines.drop_duplicates(ignore_index=True)
    else:
        df_baselines = None

    df_configs = pd.concat([z.df_configs for z in zeroshot_contexts], ignore_index=True)
    df_configs = df_configs.drop_duplicates(ignore_index=True)
    df_metadata = pd.concat([z.df_metadata for z in zeroshot_contexts], ignore_index=True)
    df_metadata = df_metadata.drop_duplicates(ignore_index=True)

    configs_hyperparameters = {}
    for z in zeroshot_contexts:
        if z.configs_hyperparameters is not None:
            intersection = set(configs_hyperparameters.keys()).intersection(set(z.configs_hyperparameters.keys()))
            for k in intersection:
                assert configs_hyperparameters[k] == z.configs_hyperparameters[k], (f"Inconsistent hyperparameters for config={k} found!\n"
                                                                                    f"Hyperparameters 1:\n"
                                                                                    f"\t{configs_hyperparameters[k]}\n"
                                                                                    f"Hyperparameters 2:\n"
                                                                                    f"\t{z.configs_hyperparameters[k]}")
            configs_hyperparameters.update(z.configs_hyperparameters)
    if len(configs_hyperparameters) == 0:
        configs_hyperparameters = None

    pct = zeroshot_contexts[0].pct
    score_against_only_baselines = zeroshot_contexts[0].score_against_only_baselines

    if require_matching_flags:
        for z in zeroshot_contexts[1:]:
            assert pct == z.pct, f"Inconsistent `pct` value found! ({pct}, {z.pct})"
        for z in zeroshot_contexts[1:]:
            assert score_against_only_baselines == z.score_against_only_baselines, f"Inconsistent `score_against_only_baselines` value found! ({score_against_only_baselines}, {z.score_against_only_baselines})"

    folds = None
    if any(z.folds is None for z in zeroshot_contexts):
        folds = None
    else:
        folds = []
        for z in zeroshot_contexts:
            folds += [f for f in z.folds if f not in folds]

    zeroshot_context_merged = ZeroshotSimulatorContext(
        df_configs=df_configs,
        df_baselines=df_baselines,
        df_metadata=df_metadata,
        configs_hyperparameters=configs_hyperparameters,
        folds=folds,
        pct=pct,
        score_against_only_baselines=score_against_only_baselines,
    )

    return zeroshot_context_merged


# TODO: Does not yet verify equivalence
def merge_ground_truth(ground_truths: list[GroundTruth]) -> GroundTruth:
    assert isinstance(ground_truths, list)

    label_test_dict = copy.copy(ground_truths[0]._label_test_dict)
    label_val_dict = copy.copy(ground_truths[0]._label_val_dict)
    for gt in ground_truths[1:]:
        datasets_gt = gt.datasets
        for d in datasets_gt:
            if d not in label_test_dict:
                label_test_dict[d] = {}
                label_val_dict[d] = {}
            label_test_dict[d].update(gt._label_test_dict[d])
            label_val_dict[d].update(gt._label_val_dict[d])
    ground_truth_merged = GroundTruth(label_test_dict=label_test_dict, label_val_dict=label_val_dict)
    return ground_truth_merged
