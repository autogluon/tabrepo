from __future__ import annotations

import copy
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
            df_results_by_dataset_automl: pd.DataFrame = None,
            df_metadata: pd.DataFrame = None,
            folds: List[int] | None = None,
            pct: bool = False,
            score_against_only_automl: bool = True,
    ):
        """
        Encapsulates results evaluated on multiple base models/datasets/folds.
        :param df_raw:
        :param df_results_by_dataset_automl: results of automl systems by multiple datasets/folds
        :param folds: List of folds to be considered in a list of integers. If None, will not filter by fold.
        :param pct: whether to use percentage rather than rank numbers
        :param score_against_only_automl: if `True`, the scores are ranks (or percentage if `pct` is True) over automl
        baselines. If False, the scores are computed against both automl baselines and random model configurations
        for all base models (random-forest, knns etc).
        """
        self.folds = folds
        self.score_against_only_automl = score_against_only_automl
        self.pct = pct

        # TODO align_valid_folds returns 8 values and does many different things, it would help to break down to more
        #  modular functions
        self.df_raw, \
        self.df_results_by_dataset_automl, \
        self.df_results_by_dataset_vs_automl, \
        self.df_metrics, \
        self.df_metadata, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self._align_valid_folds(
            df_raw=df_raw,
            df_comparison=df_results_by_dataset_automl,
            df_metadata=df_metadata,
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
        self.df_raw, \
        self.df_results_by_dataset_automl, \
        self.df_results_by_dataset_vs_automl, \
        self.df_metrics, \
        self.df_metadata, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self._align_valid_folds(
            df_raw=self.df_raw,
            df_comparison=self.df_results_by_dataset_automl,
            df_metadata=self.df_metadata,
            folds=folds,
            score_against_only_automl=self.score_against_only_automl,
            pct=self.pct,
        )
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

        self.dataset_to_problem_type_dict = self.df_results_by_dataset_vs_automl[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()

    @classmethod
    def _align_valid_folds(cls,
                           *,
                           df_raw: pd.DataFrame,
                           df_comparison: pd.DataFrame,
                           df_metadata: pd.DataFrame,
                           folds: List[int] | None,
                           score_against_only_automl: bool,
                           pct: bool):
        cls._validate_df_raw(df_raw=df_raw)
        if df_comparison is not None:
            cls._validate_df_comparison(df_comparison=df_comparison)
        # assert that each dataset contains only one problem type
        dataset_problem_types = df_raw[["dataset", "problem_type"]].drop_duplicates()
        assert len(dataset_problem_types) == len(dataset_problem_types["dataset"].unique()), \
            "Error: datasets exist in `df_raw` that contain multiple problem_types!"

        if df_comparison is not None:
            dataset_problem_types_comparison = df_comparison[["dataset", "problem_type"]].drop_duplicates()
            assert len(dataset_problem_types_comparison) == len(dataset_problem_types_comparison["dataset"].unique()), \
                "Error: datasets exist in `df_comparison` that contain multiple problem_types!"
            dataset_problem_types_map_raw = dataset_problem_types.set_index("dataset").squeeze().to_dict()
            dataset_problem_types_map_comparison = dataset_problem_types_comparison.set_index("dataset").squeeze().to_dict()
            for d in dataset_problem_types_map_raw.keys():
                problem_type_raw = dataset_problem_types_map_raw[d]
                if d in dataset_problem_types_map_comparison:
                    problem_type_comparison = dataset_problem_types_map_comparison[d]
                    assert problem_type_raw == problem_type_comparison, \
                        (f"Error: Dataset `{d}` has a different `problem_type` between `df_raw` and `df_comparison`. They must match:\n"
                         f"\tdf_raw       : {problem_type_raw}\n"
                         f"\tdf_comparison: {problem_type_comparison}")

        if "tid" not in df_raw.columns:
            print(f"Note: `tid` is missing from `df_raw` columns. `tid` will be created by mapping the sorted unique `dataset` values to [0, n-1]. "
                  f"These values will be unrelated to OpenML task IDs.")
            df_raw = df_raw.copy(deep=True)
            datasets = sorted(list(df_raw["dataset"].unique()))
            dataset_to_tid_map = {d: i for i, d in enumerate(datasets)}
            df_raw["tid"] = df_raw["dataset"].map(dataset_to_tid_map).astype(int)

        # assert that each dataset-tid combination is exclusive
        dataset_tid = df_raw[["dataset", "tid"]].drop_duplicates()
        assert len(dataset_tid) == len(dataset_tid["dataset"].unique())
        assert len(dataset_tid) == len(dataset_tid["tid"].unique())

        if df_comparison is not None:
            df_comparison = df_comparison.drop(columns=["tid"], errors="ignore").merge(dataset_tid, on=["dataset"])
            df_comparison = df_comparison.drop(columns=["problem_type"], errors="ignore").merge(dataset_problem_types, on=["dataset"])

        unique_dataset_folds_set = df_raw[['dataset', 'fold']].drop_duplicates()

        sources_to_check = []
        if df_comparison is not None:
            df_comparison = filter_datasets(df=df_comparison, datasets=unique_dataset_folds_set)
            unique_dataset_folds_set = df_comparison[['dataset', 'fold']].drop_duplicates()
            df_raw = filter_datasets(df=df_raw, datasets=unique_dataset_folds_set)
            sources_to_check.append(("df_comparison", df_comparison))
        sources_to_check.append(("df_raw", df_raw))

        for source, df_source in sources_to_check:
            config_task_counts = df_source[['framework', 'dataset', 'fold']].value_counts()
            if config_task_counts.max() > 1:
                raise AssertionError(f'Multiple rows in `{source}` exist for a config task pair! '
                                     f'There should only ever be one row per config task pair. '
                                     f'You might have multiple results from re-runs or other bugs that have not been de-duplicated.\n'
                                     f'Config Task Counts:\n'
                                     f'{config_task_counts}')

        if df_comparison is not None:
            df_comparison = filter_datasets(df=df_comparison, datasets=unique_dataset_folds_set)

        if folds is not None:
            tasks = df_raw[['dataset', 'fold']].drop_duplicates()
            tasks_in_valid_folds = tasks[tasks['fold'].isin(folds)]
            dataset_fold_counts = tasks_in_valid_folds['dataset'].value_counts()
            datasets_with_all_folds = dataset_fold_counts[dataset_fold_counts == len(folds)]
            unique_datasets = sorted(list(datasets_with_all_folds.index))
            unique_datasets_set = set(unique_datasets)
            unique_dataset_folds_set = unique_dataset_folds_set[unique_dataset_folds_set["dataset"].isin(unique_datasets_set)]
            df_raw = filter_datasets(df=df_raw, datasets=unique_dataset_folds_set)
            if df_comparison is not None:
                df_comparison = filter_datasets(df=df_comparison, datasets=unique_dataset_folds_set)

        df_raw["task"] = df_raw["tid"].astype(str) + '_' + df_raw["fold"].astype(str)
        if df_comparison is not None:
            df_comparison["task"] = df_comparison["tid"].astype(str) + '_' + df_comparison["fold"].astype(str)

        unique_tasks = sorted(list(df_raw["task"].unique()))
        unique_datasets = sorted(list(df_raw["dataset"].unique()))

        if df_comparison is None:
            df_results_baselines = df_raw.copy(deep=True)
        elif score_against_only_automl:
            df_results_baselines = df_comparison.copy(deep=True)
        else:
            df_results_baselines = pd.concat([df_raw, df_comparison], ignore_index=True)
        rank_scorer_vs_automl = RankScorer(
            df_results_by_dataset=df_results_baselines,
            datasets=unique_tasks,
            pct=pct,
        )
        df_configs_ranked = df_raw.copy()
        df_configs_ranked['rank'] = df_configs_ranked.apply(
            lambda row: rank_scorer_vs_automl.rank(row["task"], row["metric_error"]), axis=1
        )

        task_to_dataset_dict = get_task_to_dataset_dict(df_raw=df_raw)
        dataset_to_tid_dict = get_dataset_to_tid_dict(df_raw=df_raw)
        assert len(unique_datasets) == len(dataset_to_tid_dict.keys())

        df_metrics = get_dataset_to_metric_problem_type(df_raw=df_raw)

        if df_metadata is not None:
            assert "dataset" in df_metadata, (f"Missing required `dataset` column in metadata.\n"
                                              f"Columns: {list(df_metadata.columns)}")
            df_metadata = copy.deepcopy(df_metadata)
            df_metadata = df_metadata[df_metadata["dataset"].isin(unique_datasets)]
            assert sorted(list(df_metadata["dataset"].unique())) == sorted(list(unique_datasets))
            assert len(df_metadata) == len(unique_datasets)

        return (
            df_raw,
            df_comparison,
            df_configs_ranked,
            df_metrics,
            df_metadata,
            task_to_dataset_dict,
            dataset_to_tid_dict,
            unique_tasks,
            unique_datasets,
            rank_scorer_vs_automl,
        )

    @staticmethod
    def _validate_df(df: pd.DataFrame, name: str, required_columns: List[str]):
        df_columns = list(df.columns)
        missing_required_columns = [c for c in required_columns if c not in df_columns]
        if missing_required_columns:
            present_required_columns = [c for c in required_columns if c in df_columns]
            present_extra_columns = [c for c in df_columns if c not in required_columns]
            raise AssertionError(f"Missing required columns in `{name}`:\n"
                                 f"\tMissing Required: {missing_required_columns}\n"
                                 f"\tPresent Required: {present_required_columns}\n"
                                 f"\tPresent    Extra: {present_extra_columns}\n"
                                 f"{df}")

    @classmethod
    def _validate_df_raw(cls, df_raw: pd.DataFrame):
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric_error_val",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        cls._validate_df(df=df_raw, name="df_raw", required_columns=required_columns)

    @classmethod
    def _validate_df_comparison(cls, df_comparison: pd.DataFrame):
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        cls._validate_df(df=df_comparison, name="df_comparison", required_columns=required_columns)

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

    @staticmethod
    def task_name_from_tid(tid: int, fold: int) -> str:
        return f"{tid}_{fold}"

    def get_configs(self, *, datasets: List[str] = None, tasks: List[str] = None, union: bool = True) -> List[str]:
        """
        Return all valid configs.
        By default, will return all configs that appear in any task at least once.

        Parameters
        ----------
        datasets : List[str], default = None
            If specified, will only consider the configs present in the given datasets
        tasks: List[str], default = None
            If specified, will only consider the configs present in the given tasks
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns
        -------
        A list of config names satisfying the above conditions.
        """
        df = self.df_results_by_dataset_vs_automl
        if datasets is not None:
            datasets_all = set(self.get_datasets())
            datasets_invalid = set(datasets).difference(datasets_all)
            if len(datasets_invalid) != 0:
                raise ValueError(f"Invalid datasets specified: {sorted(list(datasets_invalid))}")
            df = df[df["dataset"].isin(datasets)]
        if tasks is not None:
            tasks_all = set(self.get_tasks())
            tasks_invalid = set(tasks).difference(tasks_all)
            if len(tasks_invalid) != 0:
                raise ValueError(f"Invalid tasks specified: {sorted(list(tasks_invalid))}")
            df = df[df["task"].isin(tasks)]

        if len(df) == 0:
            raise AssertionError(f"No valid results for tasks={tasks} | datasets={datasets}")

        return self._get_configs_from_df(df=df, union=union)

    def load_groundtruth(self, paths_gt: List[str]) -> Tuple[dict, dict]:
        gt_val = defaultdict(_default_dict)
        gt_test = defaultdict(_default_dict)
        for p in paths_gt:
            with open(Path(p).parent / "metadata.json", "r") as f:
                metadata = json.load(f)
            dataset = metadata["dataset"]
            if dataset in self.dataset_to_tid_dict:
                fold = metadata["fold"]
                if Path(p).stem.startswith("label-test"):
                    gt_test[dataset][fold] = pd.read_csv(p, index_col=0)
                else:
                    gt_val[dataset][fold] = pd.read_csv(p, index_col=0)
        return GroundTruth(label_val_dict=gt_val, label_test_dict=gt_test)

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
            self.df_metadata = self.df_metadata[self.df_metadata["dataset"].isin(datasets)]
        self.dataset_to_tid_dict = {d: t for d, t in self.dataset_to_tid_dict.items() if d in datasets}

    def subset_problem_types(self, problem_types: List[str]):
        """
        Only keep the provided problem_types, drop all others
        """
        datasets = self.get_datasets(problem_type=problem_types)
        self.subset_datasets(datasets=datasets)

    def subset_configs(self, configs: List[str]):
        """
        Only keep the provided configs, drop all others
        """
        self.df_results_by_dataset_vs_automl = self.df_results_by_dataset_vs_automl[
            self.df_results_by_dataset_vs_automl['framework'].isin(configs)
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

    @staticmethod
    def _get_configs_from_df(df: pd.DataFrame, union: bool = True) -> List[str]:
        """
        Parameters
        ----------
        df: pd.DataFrame
            A DataFrame containing the columns "task" and "framework".
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns
        -------
        The list of "framework" values present in `df` that satisfy the `union` value logic.
        """
        if union:
            res = df["framework"].unique()
        else:
            tasks = list(df["task"].unique())
            res = None
            for task in tasks:
                methods = set(df.loc[df["task"] == task, "framework"].unique())
                if res is None:
                    res = methods
                else:
                    res = res.intersection(methods)
        return sorted(list(res))
