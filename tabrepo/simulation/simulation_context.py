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
            df_configs: pd.DataFrame,
            df_baselines: pd.DataFrame = None,
            df_metadata: pd.DataFrame = None,
            configs_hyperparameters: dict = None,
            folds: List[int] | None = None,
            pct: bool = False,
            score_against_only_baselines: bool = True,
    ):
        """
        Encapsulates results evaluated on multiple base models/datasets/folds.
        :param df_configs: results of configs by multiple datasets/folds
        :param df_baselines: results of baseline systems by multiple datasets/folds
        :param folds: List of folds to be considered in a list of integers. If None, will not filter by fold.
        :param pct: whether to use percentage rather than rank numbers
        :param score_against_only_baselines: if `True`, the scores are ranks (or percentage if `pct` is True) over the baselines only
        baselines. If False, the scores are computed against both baselines and the configs.
        """
        self.folds = folds
        self.score_against_only_baselines = score_against_only_baselines
        self.pct = pct

        # TODO align_valid_folds returns 8 values and does many different things, it would help to break down to more
        #  modular functions
        self.df_configs, \
        self.df_baselines, \
        self.df_configs_ranked, \
        self.df_metrics, \
        self.df_metadata, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.configs_hyperparameters, \
        self.rank_scorer = self._align_valid_folds(
            df_configs=df_configs,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
            configs_hyperparameters=configs_hyperparameters,
            folds=folds,
            score_against_only_automl=self.score_against_only_baselines,
            pct=self.pct,
        )
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

        self.dataset_to_problem_type_dict = self.df_configs_ranked[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze(axis=1).to_dict()
        self.task_to_fold_dict = self.df_configs_ranked[["task", "fold"]].drop_duplicates().set_index("task").squeeze(axis=1).to_dict()

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
        self.df_configs, \
        self.df_baselines, \
        self.df_configs_ranked, \
        self.df_metrics, \
        self.df_metadata, \
        self.task_to_dataset_dict, \
        self.dataset_to_tid_dict, \
        self.unique_tasks, \
        self.unique_datasets, \
        self.configs_hyperparameters, \
        self.rank_scorer = self._align_valid_folds(
            df_configs=self.df_configs,
            df_baselines=self.df_baselines,
            df_metadata=self.df_metadata,
            configs_hyperparameters=self.configs_hyperparameters,
            folds=folds,
            score_against_only_automl=self.score_against_only_baselines,
            pct=self.pct,
        )
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

        self.dataset_to_problem_type_dict = self.df_configs_ranked[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze(axis=1).to_dict()
        self.task_to_fold_dict = self.df_configs_ranked[["task", "fold"]].drop_duplicates().set_index("task").squeeze(axis=1).to_dict()

    @classmethod
    def _align_valid_folds(cls,
                           *,
                           df_configs: pd.DataFrame,
                           df_baselines: pd.DataFrame,
                           df_metadata: pd.DataFrame,
                           configs_hyperparameters: dict | None = None,
                           folds: List[int] | None,
                           score_against_only_automl: bool,
                           pct: bool):
        cls._validate_df_configs(df_configs=df_configs)
        if df_baselines is not None:
            cls._validate_df_baselines(df_baselines=df_baselines)
        # assert that each dataset contains only one problem type
        dataset_problem_types = df_configs[["dataset", "problem_type"]].drop_duplicates()
        assert len(dataset_problem_types) == len(dataset_problem_types["dataset"].unique()), \
            "Error: datasets exist in `df_configs` that contain multiple problem_types!"

        if df_baselines is not None:
            dataset_problem_types_comparison = df_baselines[["dataset", "problem_type"]].drop_duplicates()
            assert len(dataset_problem_types_comparison) == len(dataset_problem_types_comparison["dataset"].unique()), \
                "Error: datasets exist in `df_comparison` that contain multiple problem_types!"
            dataset_problem_types_map_configs = dataset_problem_types.set_index("dataset").squeeze(axis=1).to_dict()
            dataset_problem_types_map_baselines = dataset_problem_types_comparison.set_index("dataset").squeeze(axis=1).to_dict()
            for d in dataset_problem_types_map_configs.keys():
                problem_type_configs = dataset_problem_types_map_configs[d]
                if d in dataset_problem_types_map_baselines:
                    problem_type_baselines = dataset_problem_types_map_baselines[d]
                    assert problem_type_configs == problem_type_baselines, \
                        (f"Error: Dataset `{d}` has a different `problem_type` between `df_configs` and `df_baselines`. They must match:\n"
                         f"\tdf_configs  : {problem_type_configs}\n"
                         f"\tdf_baselines: {problem_type_baselines}")

        if "tid" not in df_configs.columns:
            print(f"Note: `tid` is missing from `df_configs` columns. `tid` will be created by mapping the sorted unique `dataset` values to [0, n-1]. "
                  f"These values will be unrelated to OpenML task IDs.")
            df_configs = df_configs.copy(deep=True)
            datasets = sorted(list(df_configs["dataset"].unique()))
            dataset_to_tid_map = {d: i for i, d in enumerate(datasets)}
            df_configs["tid"] = df_configs["dataset"].map(dataset_to_tid_map).astype(int)

        # assert that each dataset-tid combination is exclusive
        dataset_tid = df_configs[["dataset", "tid"]].drop_duplicates()
        assert len(dataset_tid) == len(dataset_tid["dataset"].unique())
        assert len(dataset_tid) == len(dataset_tid["tid"].unique())

        if df_baselines is not None:
            df_baselines = df_baselines.drop(columns=["tid"], errors="ignore").merge(dataset_tid, on=["dataset"])
            df_baselines = df_baselines.drop(columns=["problem_type"], errors="ignore").merge(dataset_problem_types, on=["dataset"])

        unique_dataset_folds_set = df_configs[['dataset', 'fold']].drop_duplicates()

        sources_to_check = []
        if df_baselines is not None:
            df_baselines = filter_datasets(df=df_baselines, datasets=unique_dataset_folds_set)
            unique_dataset_folds_set = df_baselines[['dataset', 'fold']].drop_duplicates()
            df_configs = filter_datasets(df=df_configs, datasets=unique_dataset_folds_set)
            sources_to_check.append(("df_baselines", df_baselines))
        sources_to_check.append(("df_configs", df_configs))

        for source, df_source in sources_to_check:
            config_task_counts = df_source[['framework', 'dataset', 'fold']].value_counts()
            if config_task_counts.max() > 1:
                raise AssertionError(f'Multiple rows in `{source}` exist for a config task pair! '
                                     f'There should only ever be one row per config task pair. '
                                     f'You might have multiple results from re-runs or other bugs that have not been de-duplicated.\n'
                                     f'Config Task Counts:\n'
                                     f'{config_task_counts}')

        if df_baselines is not None:
            df_baselines = filter_datasets(df=df_baselines, datasets=unique_dataset_folds_set)

        if folds is not None:
            tasks = df_configs[['dataset', 'fold']].drop_duplicates()
            tasks_in_valid_folds = tasks[tasks['fold'].isin(folds)]
            dataset_fold_counts = tasks_in_valid_folds['dataset'].value_counts()
            datasets_with_all_folds = dataset_fold_counts[dataset_fold_counts == len(folds)]
            unique_datasets = sorted(list(datasets_with_all_folds.index))
            unique_datasets_set = set(unique_datasets)
            unique_dataset_folds_set = unique_dataset_folds_set[unique_dataset_folds_set["dataset"].isin(unique_datasets_set)]
            unique_dataset_folds_set = unique_dataset_folds_set[unique_dataset_folds_set["fold"].isin(folds)]
            df_configs = filter_datasets(df=df_configs, datasets=unique_dataset_folds_set)
            if df_baselines is not None:
                df_baselines = filter_datasets(df=df_baselines, datasets=unique_dataset_folds_set)

        df_configs["task"] = df_configs["tid"].astype(str) + '_' + df_configs["fold"].astype(str)
        if df_baselines is not None:
            df_baselines["task"] = df_baselines["tid"].astype(str) + '_' + df_baselines["fold"].astype(str)

        unique_tasks = sorted(list(df_configs["task"].unique()))
        unique_datasets = sorted(list(df_configs["dataset"].unique()))

        if score_against_only_automl:
            assert df_baselines is not None, (f"`score_against_only_automl=True`, but `df_baselines` is None. "
                                              f"Either specify `df_baselines` or set `score_against_only_automl=False`.")
            df_comparison = df_baselines.copy(deep=True)
        elif df_baselines is None:
            df_comparison = df_configs.copy(deep=True)
        else:
            df_comparison = pd.concat([df_configs, df_baselines], ignore_index=True)
        rank_scorer = RankScorer(
            df_results=df_comparison,
            tasks=unique_tasks,
            pct=pct,
        )
        df_configs_ranked = df_configs.copy()
        df_configs_ranked['rank'] = df_configs_ranked.apply(
            lambda row: rank_scorer.rank(row["task"], row["metric_error"]), axis=1
        )

        task_to_dataset_dict = get_task_to_dataset_dict(df=df_configs)
        dataset_to_tid_dict = get_dataset_to_tid_dict(df=df_configs)
        assert len(unique_datasets) == len(dataset_to_tid_dict.keys())

        df_metrics = get_dataset_to_metric_problem_type(df=df_configs)

        if df_metadata is not None:
            assert "dataset" in df_metadata, (f"Missing required `dataset` column in metadata.\n"
                                              f"Columns: {list(df_metadata.columns)}")
            df_metadata = copy.deepcopy(df_metadata)
            df_metadata = df_metadata[df_metadata["dataset"].isin(unique_datasets)]
            assert sorted(list(df_metadata["dataset"].unique())) == sorted(list(unique_datasets))
            assert len(df_metadata) == len(unique_datasets)

        if configs_hyperparameters is not None:
            cls._verify_configs_hyperparameters(configs_hyperparameters=configs_hyperparameters)
            configs_hyperparameters = copy.deepcopy(configs_hyperparameters)
            unique_configs = sorted(list(df_configs["framework"].unique()))
            # TODO: Avoid needing to do configs_base, make the names match to begin with
            configs_base = set([cls._config_name_to_config_base(config) for config in unique_configs])
            configs_hyperparameters_keys = list(configs_hyperparameters.keys())
            for c in configs_hyperparameters_keys:
                if c not in configs_base:
                    configs_hyperparameters.pop(c)

        return (
            df_configs,
            df_baselines,
            df_configs_ranked,
            df_metrics,
            df_metadata,
            task_to_dataset_dict,
            dataset_to_tid_dict,
            unique_tasks,
            unique_datasets,
            configs_hyperparameters,
            rank_scorer,
        )

    def df_dataset_folds(self) -> pd.DataFrame:
        df_dataset_folds = self.df_configs[["dataset", "fold"]].drop_duplicates().reset_index(drop=True)
        return df_dataset_folds

    def dataset_folds(self) -> list[tuple]:
        dataset_folds = self.df_dataset_folds().values.tolist()
        dataset_folds = [tuple(dataset_fold) for dataset_fold in dataset_folds]
        return dataset_folds

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
    def _validate_df_configs(cls, df_configs: pd.DataFrame):
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
        cls._validate_df(df=df_configs, name="df_configs", required_columns=required_columns)

    @classmethod
    def _validate_df_baselines(cls, df_baselines: pd.DataFrame):
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
        cls._validate_df(df=df_baselines, name="df_baselines", required_columns=required_columns)

    def print_info(self):
        out = '====== Zeroshot Simulator Context Info ======\n'
        out += f'# Configs: {len(self.get_configs())}\n'
        out += f'# Datasets: {len(self.unique_datasets)}\n'
        if self.folds is not None:
            out += f'# Folds: {len(self.folds)}\n'
            out += f'Folds: {self.folds}\n'
        out += f'# Folds*Datasets: {len(self.unique_tasks)}\n'
        out += '=============================================\n'
        print(out)

    def get_datasets(self, problem_type=None, union=True) -> List[str]:
        datasets = self.unique_datasets
        if problem_type is not None:
            if isinstance(problem_type, list):
                datasets = [dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] in problem_type]
            else:
                datasets = [dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] == problem_type]
        if not union:
            configs = self.get_configs(union=True)
            n_configs = len(configs)
            df_configs_filtered = self.df_configs[self.df_configs["framework"].isin(configs)]
            value_counts = df_configs_filtered.value_counts(["dataset", "fold"])
            value_counts_valid = value_counts[value_counts == n_configs]
            dataset_value_counts = value_counts_valid.reset_index(drop=False)[["dataset", "count"]].groupby("dataset")["count"].sum().to_dict()
            dataset_task_counts = {d: len(tasks) for d, tasks in self.dataset_to_tasks_dict.items()}

            # filter to only datasets that contain all configs
            datasets = [d for d in datasets if dataset_value_counts.get(d, 0) == (dataset_task_counts[d] * n_configs)]
        return datasets

    def task_to_fold(self, task) -> int:
        return self.task_to_fold_dict[task]

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
        df = self.df_configs_ranked
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

    def load_groundtruth(self, paths_gt: List[str]) -> GroundTruth:
        gt_val = defaultdict(_default_dict)
        gt_test = defaultdict(_default_dict)
        unique_datasets = set(self.unique_datasets)
        for p in paths_gt:
            with open(Path(p).parent / "metadata.json", "r") as f:
                metadata = json.load(f)
            dataset = metadata["dataset"]
            if dataset in unique_datasets:
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
        zeroshot_pred_proba: TabularModelPredictions = class_map[prediction_format].from_data_dir(data_dir=path_pred_proba, datasets=datasets)
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
        self.df_configs = self.df_configs[self.df_configs["dataset"].isin(datasets)]
        self.df_configs_ranked = self.df_configs_ranked[self.df_configs_ranked["dataset"].isin(datasets)]
        if self.df_baselines is not None:
            self.df_baselines = self.df_baselines[self.df_baselines["dataset"].isin(datasets)]
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
        self.df_configs = self.df_configs[
            self.df_configs['framework'].isin(configs)
        ]
        self.df_configs_ranked = self.df_configs_ranked[
            self.df_configs_ranked['framework'].isin(configs)
        ]
        self._update_all()

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

    def get_configs_hyperparameters(self, configs: List[str] | None = None, include_ag_args: bool = True) -> dict[str, dict | None]:
        if configs is None:
            configs = self.get_configs()
        return {c: self.get_config_hyperparameters(config=c, include_ag_args=include_ag_args) for c in configs}

    def get_config_hyperparameters(self, config: str, include_ag_args: bool = True) -> dict | None:
        if config not in self.get_configs():
            raise ValueError(f"User-specified config does not exist: '{config}'")
        if self.configs_hyperparameters is None:
            return None
        # FIXME: (TabRepo v2) Hardcoding _BAG name, avoid this
        config_base = self._config_name_to_config_base(config=config)
        if config_base not in self.configs_hyperparameters:
            return None
        config_hyperparameters = self.configs_hyperparameters[config_base]
        if not include_ag_args:
            config_hyperparameters = copy.deepcopy(config_hyperparameters)
            config_hyperparameters["hyperparameters"].pop("ag_args", None)
        return config_hyperparameters

    # FIXME: This should be removed eventually
    @classmethod
    def _config_name_to_config_base(cls, config: str) -> str:
        return config.rsplit("_BAG_", 1)[0]

    @property
    def configs_type(self) -> dict[str, str | None]:
        """
        Returns a dict of config name -> config type.
        If model type is unknown, the value will be `None`.

        For example:
        "CatBoost_c1_BAG_L1": "CAT"
        "ExtraTrees_r13_BAG_L1": "XT"
        """
        configs = self.get_configs()
        configs_type = {}
        for config in configs:
            config_hyperparameters = self.get_config_hyperparameters(config=config)
            if config_hyperparameters is None:
                configs_type[config] = None
            else:
                configs_type[config] = config_hyperparameters.get("model_type", None)
        return configs_type

    def df_configs_task(self, dataset: str, fold: int, configs: list[str] = None) -> pd.DataFrame:
        df_configs_task = self.df_configs[(self.df_configs["dataset"] == dataset) & (self.df_configs["fold"] == fold)]
        if configs is not None:
            configs = set(configs)
            df_configs_task = df_configs_task[df_configs_task["framework"].isin(configs)]
        return df_configs_task

    # TODO: Support max_models and max_models_per_type for simulation
    def get_top_configs(self, dataset: str, fold: int, configs: list[str] = None, max_models: int = None, max_models_per_type: int = None) -> List[str]:
        df_configs_task = self.df_configs_task(dataset=dataset, fold=fold, configs=configs)
        df_configs_task_sorted = df_configs_task.sort_values(by=["metric_error_val", "framework"])
        df_configs_task_sorted["type"] = df_configs_task_sorted["framework"].map(self.configs_type).fillna("nan")

        if max_models_per_type is not None:
            df_configs_task_sorted["rank_by_type"] = df_configs_task_sorted.groupby(["type", "task"])["metric_error_val"].rank("first").astype(int)
            df_configs_task_sorted = df_configs_task_sorted[df_configs_task_sorted["rank_by_type"] <= max_models_per_type]
        if max_models is not None:
            df_configs_task_sorted["rank"] = df_configs_task_sorted.groupby("task")["metric_error_val"].rank("first").astype(int)
            df_configs_task_sorted = df_configs_task_sorted[df_configs_task_sorted["rank"] <= max_models]
        return list(df_configs_task_sorted["framework"])

    # TODO: Potentially change the format in future
    # TODO: Check for `model_type` key?
    # TODO: What about `name_prefix` and `name_suffix`?
    @classmethod
    def _verify_configs_hyperparameters(cls, configs_hyperparameters: dict[str, dict[str, str | dict]]):
        for config, v in configs_hyperparameters.items():
            assert isinstance(v, dict), f"configs_hyperparameters value for key {config} must be of type dict, found: {type(v)}"
            assert "hyperparameters" in v, f"configs_hyperparameters value for key {config} must include a `hyperparameters` key"
            assert isinstance(v["hyperparameters"], dict), (f"configs_hyperparameters['{config}']['hyperparameters'] "
                                                            f"must be of type dict, found: {type(v['hyperparameters'])}")
