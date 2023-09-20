import pickle
import sys
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
from autogluon.common.loaders import load_pkl

from ..loaders import Paths

from .sim_utils import get_dataset_to_tid_dict, get_dataset_name_to_tid_dict, filter_datasets
from .tabular_predictions import TabularPicklePredictions, TabularPicklePerTaskPredictions, TabularModelPredictions
from ..utils.rank_utils import RankScorer


class ZeroshotSimulatorContext:
    def __init__(
            self, 
            df_results_by_dataset: pd.DataFrame,
            df_results_by_dataset_automl: pd.DataFrame,
            df_raw: pd.DataFrame, 
            folds: List[int],
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
        self.df_results_by_dataset_automl = df_results_by_dataset_automl
        self.df_metadata = df_metadata
        self.df_results_by_dataset_vs_automl, \
        self.df_raw, \
        self.dataset_name_to_tid_dict, \
        self.dataset_to_tid_dict, \
        self.dataset_name_to_fold_dict, \
        self.unique_dataset_folds, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self.align_valid_folds(
            df_results_by_dataset=df_results_by_dataset,
            df_results_by_dataset_automl=df_results_by_dataset_automl,
            df_raw=df_raw,
            folds=folds,
            score_against_only_automl=self.score_against_only_automl,
            pct=self.pct,
        )
        self.dataset_parent_to_fold_map = self._compute_dataset_parent_to_fold_map()

        tmp = self.df_results_by_dataset_vs_automl[['dataset', 'tid', 'problem_type']]
        self.dataset_to_problem_type_dict = tmp[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()
        self.tid_to_problem_type_dict = tmp[['tid', 'problem_type']].drop_duplicates().set_index(
            'tid').squeeze().to_dict()

    def _compute_dataset_parent_to_fold_map(self) -> dict:
        """
        Returns the mapping of dataset parent to dataset fold names.
        For example:
        {
            'DATASET_NAME': ['DATASET_NAME_1', 'DATASET_NAME_2', ..., 'DATASET_NAME_10'],
            ...,
        }

        """
        dataset_parent_to_fold_map = dict()
        for d in self.unique_dataset_folds:
            dataset_parent = self.dataset_name_to_tid_dict[d]
            if dataset_parent in dataset_parent_to_fold_map:
                dataset_parent_to_fold_map[dataset_parent].append(d)
            else:
                dataset_parent_to_fold_map[dataset_parent] = [d]
        for d in dataset_parent_to_fold_map:
            dataset_parent_to_fold_map[d] = sorted(dataset_parent_to_fold_map[d])
        return dataset_parent_to_fold_map

    def _update_all(self, folds=None):
        if folds is None:
            folds = self.folds
        self.folds = folds
        self.df_results_by_dataset_vs_automl, \
        self.df_raw, \
        self.dataset_name_to_tid_dict, \
        self.dataset_to_tid_dict, \
        self.dataset_name_to_fold_dict, \
        self.unique_dataset_folds, \
        self.unique_datasets, \
        self.rank_scorer_vs_automl = self.align_valid_folds(
            df_results_by_dataset=self.df_results_by_dataset_vs_automl,
            df_results_by_dataset_automl=self.df_results_by_dataset_automl,
            df_raw=self.df_raw,
            folds=folds,
            score_against_only_automl=self.score_against_only_automl,
            pct=self.pct,
        )
        self.dataset_parent_to_fold_map = self._compute_dataset_parent_to_fold_map()

        tmp = self.df_results_by_dataset_vs_automl[['dataset', 'tid', 'problem_type']]
        self.dataset_to_problem_type_dict = tmp[['dataset', 'problem_type']].drop_duplicates().set_index(
            'dataset').squeeze().to_dict()
        self.tid_to_problem_type_dict = tmp[['tid', 'problem_type']].drop_duplicates().set_index(
            'tid').squeeze().to_dict()

    @staticmethod
    def align_valid_folds(*,
                          df_results_by_dataset,
                          df_results_by_dataset_automl,
                          df_raw,
                          folds,
                          score_against_only_automl,
                          pct):
        df_results_by_dataset = df_results_by_dataset[df_results_by_dataset['fold'].isin(folds)]
        unique_dataset_folds_set = set(list(df_results_by_dataset['dataset'].unique()))
        df_results_by_dataset_automl = df_results_by_dataset_automl[
            df_results_by_dataset_automl['dataset'].isin(unique_dataset_folds_set)]

        unique_dataset_folds_set = set(list(df_results_by_dataset_automl['dataset'].unique()))

        config_task_counts_raw = df_raw[['model', 'fold', 'tid']].value_counts()
        config_task_counts_results_by_dataset = df_results_by_dataset[['framework', 'dataset']].value_counts()
        config_task_counts_results_by_dataset_automl = df_results_by_dataset_automl[['framework', 'dataset']].value_counts()

        for source, config_task_counts in [
            ("df_raw", config_task_counts_raw),
            ("df_results_by_dataset", config_task_counts_results_by_dataset),
            ("df_results_by_dataset_automl", config_task_counts_results_by_dataset_automl),
        ]:
            if config_task_counts.max() > 1:
                raise AssertionError(f'Multiple rows in `{source}` exist for a config task pair! '
                                     f'There should only ever be one row per config task pair. '
                                     f'You might have multiple results from re-runs or other bugs that have not been de-duplicated.\n'
                                     f'Config Task Counts:\n'
                                     f'{config_task_counts}')

        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        a = df_results_by_dataset[['tid', 'fold']].drop_duplicates()
        a = a[a['fold'].isin(folds)]
        b = a['tid'].value_counts()
        b = b[b == len(folds)]
        unique_datasets = list(b.index)
        unique_datasets = sorted(unique_datasets)

        dataset_name_to_fold_dict = df_results_by_dataset[['dataset', 'fold']].drop_duplicates().set_index('dataset')[
            'fold'].to_dict()

        dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
        unique_dataset_folds = []
        unique_datasets_set = set(unique_datasets)
        for dataset in unique_dataset_folds_set:
            if dataset_name_to_tid_dict[dataset] in unique_datasets_set:
                unique_dataset_folds.append(dataset)
        unique_dataset_folds = sorted(unique_dataset_folds)
        unique_dataset_folds_set = set(unique_dataset_folds)

        df_results_by_dataset, df_raw = filter_datasets(df_results_by_dataset=df_results_by_dataset,
                                                        df_raw=df_raw,
                                                        datasets=unique_dataset_folds_set)

        dataset_name_to_tid_dict = get_dataset_name_to_tid_dict(df_raw=df_raw)
        dataset_to_tid_dict = get_dataset_to_tid_dict(df_raw=df_raw)
        assert len(unique_datasets) == len(dataset_to_tid_dict.keys())

        if score_against_only_automl:
            df_results_baselines = df_results_by_dataset_automl
        else:
            df_results_baselines = pd.concat([df_results_by_dataset, df_results_by_dataset_automl], ignore_index=True)
        rank_scorer_vs_automl = RankScorer(
            df_results_by_dataset=df_results_baselines,
            datasets=unique_dataset_folds,
            pct=pct,
        )
        df_results_by_dataset_vs_automl = df_results_by_dataset.copy()
        df_results_by_dataset_vs_automl['rank'] = df_results_by_dataset_vs_automl.apply(
            lambda row: rank_scorer_vs_automl.rank(row["dataset"], row["metric_error"]), axis=1
        )

        return (
            df_results_by_dataset_vs_automl,
            df_raw,
            dataset_name_to_tid_dict,
            dataset_to_tid_dict,
            dataset_name_to_fold_dict,
            unique_dataset_folds,
            unique_datasets,
            rank_scorer_vs_automl,
        )

    def print_info(self):
        out = '====== Zeroshot Simulator Context Info ======\n'
        out += f'# Configs: {len(self.get_configs())}\n'
        out += f'# Datasets: {len(self.unique_datasets)}\n'
        out += f'# Folds: {len(self.folds)}\n'
        out += f'Folds: {self.folds}\n'
        out += f'# Folds*Datasets: {len(self.unique_dataset_folds)}\n'
        out += '=============================================\n'
        print(out)

    def get_datasets(self, problem_type=None) -> list:
        datasets = self.unique_datasets
        if problem_type is not None:
            if isinstance(problem_type, list):
                datasets = [dataset for dataset in datasets if self.tid_to_problem_type_dict[dataset] in problem_type]
            else:
                datasets = [dataset for dataset in datasets if self.tid_to_problem_type_dict[dataset] == problem_type]
        return datasets

    def get_dataset_folds(self,
                          datasets: Optional[List[str]] = None,
                          problem_type: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        :param datasets: a list of dataset parent names, only return folds that have a parent in this list
        :param problem_type: a problem type from AutoGluon in "multiclass", "binary", ... or list of problem types
        :return: List of datasets-folds formatted as `['359987_8', '359933_3', ...]` where the dataset is encoded before
        the "_" and the fold after.
        # Todo/Note it might be clearer to add a column fold in the dataframe and return List[Tuple[str, int]] with
        tuples of dataset/fold.
        """
        if datasets is not None:
            dataset_folds = self._get_dataset_folds_from_datasets(datasets=datasets)
        else:
            dataset_folds = self.unique_dataset_folds
        if problem_type is not None:
            if not isinstance(problem_type, list):
                problem_type = [problem_type]
            dataset_folds = [dataset for dataset in dataset_folds if self.dataset_to_problem_type_dict[dataset] in problem_type]
        return dataset_folds

    def _get_dataset_folds_from_datasets(self, datasets: List[str]):
        dataset_folds = []
        for d in datasets:
            dataset_folds += self.dataset_parent_to_fold_map[d]
        return dataset_folds

    @property
    def tid_to_dataset_dict(self):
        return {v: k for k, v in self.dataset_to_tid_dict.items()}

    def get_configs(self) -> list:
        """Return all valid configs"""
        return list(self.df_results_by_dataset_vs_automl['framework'].unique())

    def load_groundtruth(self, path_gt: List[str]) -> dict:
        zeroshot_gt = dict()
        for p in path_gt:
            zeroshot_gt_cur = load_pkl.load(p)
            for dataset in zeroshot_gt_cur:
                if dataset not in zeroshot_gt:
                    zeroshot_gt[dataset] = dict()
                for fold in zeroshot_gt_cur[dataset]:
                    if fold not in zeroshot_gt[dataset]:
                        zeroshot_gt[dataset][fold] = zeroshot_gt_cur[dataset][fold]
                    else:
                        raise AssertionError(f'Duplicate zs_gt results exist for: dataset="{dataset}", fold={fold}')
        zeroshot_gt = {k: v for k, v in zeroshot_gt.items() if k in self.dataset_to_tid_dict}
        zeroshot_gt = {self.dataset_to_tid_dict[k]: v for k, v in zeroshot_gt.items()}
        return zeroshot_gt

    def load_pred(self, pred_pkl_paths: List[Union[Path, str]], output_dir: str, lazy_format: bool = False) -> TabularModelPredictions:
        pred_pkl_paths = [Path(p) for p in pred_pkl_paths]
        for p in pred_pkl_paths:
            assert p.exists()
        cls = TabularPicklePerTaskPredictions if lazy_format else TabularPicklePredictions
        if lazy_format:
            # convert to lazy format if format not already available
            pred_path = self.convert_lazy_format(pred_pkl_paths=pred_pkl_paths, output_dir=output_dir)
            zeroshot_pred_proba = cls.load(pred_path)
        else:
            zeroshot_pred_proba = cls.from_paths(pred_pkl_paths)

        valid_datasets = [d for d in zeroshot_pred_proba.datasets if d in self.dataset_to_tid_dict]
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        # rename dataset to dataset-ids, eg. 'abalone' is mapped to 359944.0
        zeroshot_pred_proba.rename_datasets({
            k: self.dataset_to_tid_dict[k]
            for k in zeroshot_pred_proba.datasets
        })
        return zeroshot_pred_proba

    @staticmethod
    def convert_lazy_format(pred_pkl_paths: List[Path], output_dir: str, override_if_already_exists: bool = False) -> str:
        """
        :param pred_pkl_paths:
        :param output_dir:
        :param override_if_already_exists:
        :return: the path of the generated lazy format
        """
        if not Path(output_dir).exists() or override_if_already_exists:
            print(f"lazy format folder {output_dir} not found or override option set to True, "
                  f"converting to lazy format. It should take less than 3 min.")
            preds_npy = TabularPicklePerTaskPredictions.from_paths(paths=pred_pkl_paths, output_dir=output_dir)
        return output_dir

    @staticmethod
    def minimize_memory_zeroshot_pred_proba(zeroshot_pred_proba: dict, configs: list):
        """
        Minimizes memory usage of zeroshot_pred_proba by popping all model keys not in the input configs list.

        Note: Performs inplace edits.
        """
        if configs is None:
            return zeroshot_pred_proba
        size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
        print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        task_names = list(zeroshot_pred_proba.keys())
        configs = set(configs)
        for t in task_names:
            available_folds = list(zeroshot_pred_proba[t].keys())
            for f in available_folds:
                model_keys = list(zeroshot_pred_proba[t][f]['pred_proba_dict_val'].keys())
                for k in model_keys:
                    if k not in configs:
                        zeroshot_pred_proba[t][f]['pred_proba_dict_val'].pop(k)
                        zeroshot_pred_proba[t][f]['pred_proba_dict_test'].pop(k)
        size_bytes = sys.getsizeof(pickle.dumps(zeroshot_pred_proba, protocol=4))
        print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        return zeroshot_pred_proba

    def subset_datasets(self, datasets):
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
        self.df_raw = self.df_raw[self.df_raw.tid.isin(datasets)]
        self.df_results_by_dataset_vs_automl = self.df_results_by_dataset_vs_automl[self.df_results_by_dataset_vs_automl["tid"].isin(datasets)]
        self.df_results_by_dataset_automl['tid'] = self.df_results_by_dataset_automl.apply(lambda x: int(x["dataset"].split("_")[0]), axis=1)
        self.df_results_by_dataset_automl = self.df_results_by_dataset_automl[self.df_results_by_dataset_automl["tid"].isin(datasets)]
        self.df_results_by_dataset_automl.drop("tid", axis=1, inplace=True)
        self.df_metadata = self.df_metadata[self.df_metadata.tid.isin(datasets)]
        self.dataset_to_tid_dict = {d: t for d, t in self.dataset_to_tid_dict.items() if t in datasets}

    def subset_problem_types(self, problem_types: List[str]):
        """
        Only keep the provided problem_types, drop all others
        """
        datasets = self.get_datasets(problem_type=problem_types)
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
        unique_dataset_folds = []
        dataset_parent_to_fold_map = dict()
        for d in unique_datasets:
            dataset_parent_to_fold_map[d] = self.dataset_parent_to_fold_map[d]
            unique_dataset_folds += dataset_parent_to_fold_map[d]
        self.unique_datasets = unique_datasets
        self.unique_dataset_folds = unique_dataset_folds
        self.dataset_parent_to_fold_map = dataset_parent_to_fold_map
