from collections import defaultdict
import json
from pathlib import Path
from typing import Optional, List, Union, Tuple

import pandas as pd

from .ground_truth import GroundTruth

from .sim_utils import get_dataset_to_tid_dict, get_task_to_dataset_dict, filter_datasets, get_dataset_to_metric_problem_type
from ..predictions import TabularModelPredictions, TabularPredictionsMemmap, TabularPredictionsInMemory, TabularPredictionsInMemoryOpt
from ..utils.rank_utils import RankScorer


def _default_dict():
    # this free function is added as `defaultdict(lambda: defaultdict(dict))` cant be pickled
    return defaultdict(dict)


class ZeroshotSimulatorContext:
    def __init__(
            self,
            df_raw: pd.DataFrame,
            df_results_by_dataset: pd.DataFrame,
            folds: List[int],
            df_results_by_dataset_automl: pd.DataFrame = None,
            df_metadata: pd.DataFrame = None,
            pct: bool = False,
            score_against_only_automl: bool = True,
    ):
        """
        Encapsulates results evaluated on multiple base models/datasets/folds.
        :param df_results_by_dataset: results of base models on multiple datasets/folds
        :param df_results_by_dataset_automl: results of automl systems by multiple datasets/folds
        :param df_raw: 
        :param folds: List of folds to be considered in a list of integers
        :param pct: whether to use percentage rather than rank numbers
        :param score_against_only_automl: if `True`, the scores are ranks (or percentage if `pct` is True) over automl
        baselines. If False, the scores are computed against both automl baselines and random model configurations
        for all base models (random-forest, knns etc).
        """
        self.folds = folds
        self.score_against_only_automl = score_against_only_automl
        self.pct = pct
        self.df_metadata = df_metadata
        # TODO align_valid_folds returns 8 values and does many different things, it would help to break down to more
        #  modular functions
        self.df_results_by_dataset_automl, \
        self.df_results_by_dataset_vs_automl, \
        self.df_raw, \
        self.df_metrics, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self._align_valid_folds(
            df_results_by_dataset=df_results_by_dataset,
            df_results_by_dataset_automl=df_results_by_dataset_automl,
            df_raw=df_raw,
            folds=folds,
            score_against_only_automl=self.score_against_only_automl,
            pct=self.pct,
        )
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

        self.dataset_to_problem_type_dict = self.df_results_by_dataset_vs_automl[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()

    def _compute_dataset_to_tasks(self) -> dict:
        """
        Returns the mapping of dataset parent to dataset fold names.
        For example:
        {
            'DATASET_NAME': ['DATASET_NAME_1', 'DATASET_NAME_2', ..., 'DATASET_NAME_10'],
            ...,
        }

        """
        dataset_to_tasks_dict = dict()
        for task in self.unique_tasks:
            dataset = self.task_to_dataset_dict[task]
            if dataset in dataset_to_tasks_dict:
                dataset_to_tasks_dict[dataset].append(task)
            else:
                dataset_to_tasks_dict[dataset] = [task]
        for task in dataset_to_tasks_dict:
            dataset_to_tasks_dict[task] = sorted(dataset_to_tasks_dict[task])
        return dataset_to_tasks_dict

    def _update_all(self, folds=None):
        if folds is None:
            folds = self.folds
        self.folds = folds
        self.df_results_by_dataset_automl, \
        self.df_results_by_dataset_vs_automl, \
        self.df_raw, \
        self.df_metrics, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self._align_valid_folds(
            df_results_by_dataset=self.df_results_by_dataset_vs_automl,
            df_results_by_dataset_automl=self.df_results_by_dataset_automl,
            df_raw=self.df_raw,
            folds=folds,
            score_against_only_automl=self.score_against_only_automl,
            pct=self.pct,
        )
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

        self.dataset_to_problem_type_dict = self.df_results_by_dataset_vs_automl[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()

    @staticmethod
    def _align_valid_folds(*,
                           df_raw: pd.DataFrame,
                           df_results_by_dataset: pd.DataFrame,
                           df_results_by_dataset_automl: pd.DataFrame,
                           folds: List[int],
                           score_against_only_automl: bool,
                           pct: bool):
        # assert that each dataset contains only one problem type
        dataset_problem_types = df_raw[["tid", "problem_type"]].drop_duplicates()
        assert len(dataset_problem_types) == len(dataset_problem_types["tid"].unique())

        # assert that each dataset-tid combination is exclusive
        dataset_tid = df_raw[["dataset", "tid"]].drop_duplicates()
        assert len(dataset_tid) == len(dataset_tid["dataset"].unique())
        assert len(dataset_tid) == len(dataset_tid["tid"].unique())

        df_results_by_dataset = df_results_by_dataset.drop(columns=["dataset"], errors="ignore").merge(dataset_tid, on=["tid"])
        if df_results_by_dataset_automl is not None:
            df_results_by_dataset_automl = df_results_by_dataset_automl.drop(columns=["dataset"], errors="ignore").merge(dataset_tid, on=["tid"])

        df_results_by_dataset = df_results_by_dataset.drop(columns=["problem_type"], errors="ignore").merge(dataset_problem_types, on=["tid"])
        if df_results_by_dataset_automl is not None:
            df_results_by_dataset_automl = df_results_by_dataset_automl.drop(columns=["problem_type"], errors="ignore").merge(dataset_problem_types, on=["tid"])

        df_results_by_dataset = df_results_by_dataset[df_results_by_dataset['fold'].isin(folds)]
        unique_dataset_folds_set = df_results_by_dataset[['dataset', 'fold']].drop_duplicates()

        config_task_counts_raw = df_raw[['framework', 'fold', 'tid']].value_counts()
        config_task_counts_results_by_dataset = df_results_by_dataset[['framework', 'dataset', 'fold']].value_counts()

        sources_to_check = [
            ("df_raw", config_task_counts_raw),
            ("df_results_by_dataset", config_task_counts_results_by_dataset),
        ]

        if df_results_by_dataset_automl is not None:
            df_results_by_dataset_automl = df_results_by_dataset_automl.merge(unique_dataset_folds_set, on=["dataset", "fold"])
            unique_dataset_folds_set = df_results_by_dataset_automl[['dataset', 'fold']].drop_duplicates()
            config_task_counts_results_by_dataset_automl = df_results_by_dataset_automl[['framework', 'dataset', 'fold']].value_counts()
            sources_to_check.append(("df_results_by_dataset_automl", config_task_counts_results_by_dataset_automl))

        for source, config_task_counts in sources_to_check:
            if config_task_counts.max() > 1:
                raise AssertionError(f'Multiple rows in `{source}` exist for a config task pair! '
                                     f'There should only ever be one row per config task pair. '
                                     f'You might have multiple results from re-runs or other bugs that have not been de-duplicated.\n'
                                     f'Config Task Counts:\n'
                                     f'{config_task_counts}')

        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        a = df_results_by_dataset[['dataset', 'fold']].drop_duplicates()
        a = a[a['fold'].isin(folds)]
        b = a['dataset'].value_counts()
        b = b[b == len(folds)]
        unique_datasets = list(b.index)
        unique_datasets = sorted(unique_datasets)

        unique_datasets_set = set(unique_datasets)
        unique_dataset_folds_set = unique_dataset_folds_set[unique_dataset_folds_set["dataset"].isin(unique_datasets_set)]

        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        df_raw["task"] = df_raw["tid"].astype(str) + '_' + df_raw["fold"].astype(str)
        df_results_by_dataset["task"] = df_results_by_dataset["tid"].astype(str) + '_' + df_results_by_dataset["fold"].astype(str)

        if df_results_by_dataset_automl is not None:
            df_results_by_dataset_automl["task"] = df_results_by_dataset_automl["tid"].astype(str) + '_' + df_results_by_dataset_automl["fold"].astype(str)

        unique_tasks = sorted(list(df_raw['task'].unique()))

        if df_results_by_dataset_automl is None:
            df_results_baselines = df_results_by_dataset
        elif score_against_only_automl:
            df_results_baselines = df_results_by_dataset_automl
        else:
            df_results_baselines = pd.concat([df_results_by_dataset, df_results_by_dataset_automl], ignore_index=True)
        rank_scorer_vs_automl = RankScorer(
            df_results_by_dataset=df_results_baselines,
            datasets=unique_tasks,
            pct=pct,
        )
        df_results_by_dataset_vs_automl = df_results_by_dataset.copy()
        df_results_by_dataset_vs_automl['rank'] = df_results_by_dataset_vs_automl.apply(
            lambda row: rank_scorer_vs_automl.rank(row["task"], row["metric_error"]), axis=1
        )

        task_to_dataset_dict = get_task_to_dataset_dict(df_raw=df_raw)
        dataset_to_tid_dict = get_dataset_to_tid_dict(df_raw=df_raw)
        assert len(unique_datasets) == len(dataset_to_tid_dict.keys())

        df_metrics = get_dataset_to_metric_problem_type(df_raw=df_raw)

        return (
            df_results_by_dataset_automl,
            df_results_by_dataset_vs_automl,
            df_raw,
            df_metrics,
            task_to_dataset_dict,
            dataset_to_tid_dict,
            unique_tasks,
            unique_datasets,
            rank_scorer_vs_automl,
        )

    def print_info(self):
        out = '====== Zeroshot Simulator Context Info ======\n'
        out += f'# Configs: {len(self.get_configs())}\n'
        out += f'# Datasets: {len(self.unique_datasets)}\n'
        out += f'# Folds: {len(self.folds)}\n'
        out += f'Folds: {self.folds}\n'
        out += f'# Folds*Datasets: {len(self.unique_tasks)}\n'
        out += '=============================================\n'
        print(out)

    def get_datasets(self, problem_type=None) -> List[str]:
        datasets = self.unique_datasets
        if problem_type is not None:
            if isinstance(problem_type, list):
                datasets = [dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] in problem_type]
            else:
                datasets = [dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] == problem_type]
        return datasets

    def get_tids(self, problem_type=None) -> List[int]:
        datasets = self.get_datasets(problem_type=problem_type)
        tids = [self.dataset_to_tid_dict[dataset] for dataset in datasets]
        return tids

    def get_tasks(self,
                  datasets: Optional[List[str]] = None,
                  problem_type: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        :param datasets: a list of dataset parent names, only return folds that have a parent in this list
        :param problem_type: a problem type from AutoGluon in "multiclass", "binary", ... or list of problem types
        :return: List of datasets-folds formatted as `['359987_8', '359933_3', ...]` where the dataset is encoded before
        the "_" and the fold after.
        """
        if datasets is not None:
            tasks = self._get_tasks_from_datasets(datasets=datasets)
        else:
            tasks = self.unique_tasks
        if problem_type is not None:
            if not isinstance(problem_type, list):
                problem_type = [problem_type]
            tasks = [task for task in tasks if self.dataset_to_problem_type_dict[self.task_to_dataset_dict[task]] in problem_type]
        return tasks

    def _get_tasks_from_datasets(self, datasets: List[str]):
        dataset_folds = []
        for d in datasets:
            dataset_folds += self.dataset_to_tasks_dict[d]
        return dataset_folds

    @property
    def tid_to_dataset_dict(self):
        return {v: k for k, v in self.dataset_to_tid_dict.items()}

    def get_configs(self) -> list:
        """Return all valid configs"""
        return list(self.df_results_by_dataset_vs_automl['framework'].unique())

    def load_groundtruth(self, paths_gt: List[str]) -> Tuple[dict, dict]:
        gt_val = defaultdict(_default_dict)
        gt_test = defaultdict(_default_dict)
        for p in paths_gt:
            with open(Path(p).parent / "metadata.json", "r") as f:
                metadata = json.load(f)
            if metadata["dataset"] in self.dataset_to_tid_dict:
                tid = self.dataset_to_tid_dict[metadata["dataset"]]
                fold = metadata["fold"]
                if Path(p).stem.startswith("label-test"):
                    gt_test[tid][fold] = pd.read_csv(p, index_col=0)
                else:
                    gt_val[tid][fold] = pd.read_csv(p, index_col=0)
        return GroundTruth(gt_val, gt_test)

    def load_pred(self, path_pred_proba: Union[Path, str], datasets: List[str], prediction_format: str = "memmap") -> TabularModelPredictions:
        """
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        """
        assert prediction_format in ["memmap", "memopt", "mem"]

        class_map = {
            "memmap": TabularPredictionsMemmap,
            "memopt": TabularPredictionsInMemoryOpt,
            "mem": TabularPredictionsInMemory
        }

        path_pred_proba = Path(path_pred_proba)
        zeroshot_pred_proba = class_map[prediction_format].from_data_dir(data_dir=path_pred_proba, datasets=datasets)
        all_datasets = self.get_datasets()
        valid_datasets = [d for d in zeroshot_pred_proba.datasets if d in all_datasets]
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        return zeroshot_pred_proba

    def subset_tids(self, tids: List[int]):
        tid_to_dataset_dict = self.tid_to_dataset_dict
        datasets = [tid_to_dataset_dict[tid] for tid in tids]
        self.subset_datasets(datasets=datasets)

    def subset_datasets(self, datasets: List[str]):
        """
        Only keep the provided datasets, drop all others
        """
        unique_datasets_subset = []
        for d in self.unique_datasets:
            if d in datasets:
                unique_datasets_subset.append(d)
        for d in datasets:
            if d not in self.unique_datasets:
                raise ValueError(f'Missing expected dataset {d} in ZeroshotSimulatorContext!')
        self._update_unique_datasets(unique_datasets_subset)

        # Remove datasets from internal dataframes
        self.df_raw = self.df_raw[self.df_raw["dataset"].isin(datasets)]
        self.df_results_by_dataset_vs_automl = self.df_results_by_dataset_vs_automl[self.df_results_by_dataset_vs_automl["dataset"].isin(datasets)]
        if self.df_results_by_dataset_automl is not None:
            self.df_results_by_dataset_automl = self.df_results_by_dataset_automl[self.df_results_by_dataset_automl["dataset"].isin(datasets)]
        if self.df_metadata is not None:
            self.df_metadata = self.df_metadata[self.df_metadata["name"].isin(datasets)]
        self.dataset_to_tid_dict = {d: t for d, t in self.dataset_to_tid_dict.items() if d in datasets}

    def subset_problem_types(self, problem_types: List[str]):
        """
        Only keep the provided problem_types, drop all others
        """
        datasets = self.get_tids(problem_type=problem_types)
        self.subset_datasets(datasets=datasets)

    def subset_models(self, models):
        """
        Only keep the provided models, drop all others
        """
        self.df_results_by_dataset_vs_automl = self.df_results_by_dataset_vs_automl[
            self.df_results_by_dataset_vs_automl['framework'].isin(models)
        ]

    def subset_folds(self, folds: List[int]):
        """
        Only keep the provided folds, drop all others
        """
        self._update_all(folds=folds)

    def _update_unique_datasets(self, unique_datasets):
        for d in unique_datasets:
            assert d in self.unique_datasets
        unique_tasks = []
        dataset_to_tasks_dict = dict()
        for d in unique_datasets:
            dataset_to_tasks_dict[d] = self.dataset_to_tasks_dict[d]
            unique_tasks += dataset_to_tasks_dict[d]
        self.unique_datasets = unique_datasets
        self.unique_tasks = unique_tasks
        self.dataset_to_tasks_dict = dataset_to_tasks_dict
