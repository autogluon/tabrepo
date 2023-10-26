from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
from pathlib import Path

import boto3
from botocore.errorfactory import ClientError
import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import download_s3_files
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from .utils import load_zeroshot_input
from ..loaders import load_configs, load_results, combine_results_with_score_val, Paths
from ..loaders._results import preprocess_comparison
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..predictions.tabular_predictions import TabularModelPredictions
from ..utils import catchtime


@dataclass
class BenchmarkPaths:
    raw: str
    results_by_dataset: str = None
    comparison: str = None
    task_metadata: str = None
    metadata_join_column: str = "dataset"
    path_pred_proba: str = None
    datasets: List[str] = None
    zs_pp: List[str] = None
    zs_gt: List[str] = None
    configs: List[str] = None

    def __post_init__(self):
        if self.zs_pp is not None and isinstance(self.zs_pp, str):
            self.zs_pp = [self.zs_pp]
        if self.zs_gt is not None and isinstance(self.zs_gt, str):
            self.zs_gt = [self.zs_gt]
        if self.configs is None:
            configs_prefix = Paths.data_root / 'configs'
            configs = [
                f'{configs_prefix}/configs_catboost.json',
                f'{configs_prefix}/configs_fastai.json',
                f'{configs_prefix}/configs_lightgbm.json',
                f'{configs_prefix}/configs_nn_torch.json',
                f'{configs_prefix}/configs_xgboost.json',
                f'{configs_prefix}/configs_rf.json',
                f'{configs_prefix}/configs_xt.json',
                f'{configs_prefix}/configs_knn.json',
            ]
            self.configs = configs

    def print_summary(self):
        max_str_len = max(len(key) for key in self.__dict__.keys())
        print(f'BenchmarkPaths Summary:')
        print("\n".join(f'\t{key + " "*(max_str_len - len(key))} = {value}' for key, value in self.__dict__.items()))

    def get_file_paths(self, include_zs: bool = True) -> List[str]:
        file_paths = [
            self.raw,
            self.results_by_dataset,
            self.comparison,
            self.task_metadata,
        ]
        if include_zs:
            file_paths += self.zs_pp
            file_paths += self.zs_gt
        file_paths = [f for f in file_paths if f is not None]
        return file_paths

    def assert_exists_all(self, check_zs=True):
        self._assert_exists(self.raw, 'raw')
        if self.results_by_dataset is not None:
            self._assert_exists(self.results_by_dataset, 'result_by_dataset')
        if self.comparison is not None:
            self._assert_exists(self.comparison, 'comparison')
        if self.task_metadata is not None:
            self._assert_exists(self.task_metadata, 'task_metadata')
        if check_zs:
            if self.zs_pp is not None:
                for f in self.zs_pp:
                    self._assert_exists(f, f'zs_pp | {f}')
            if self.zs_gt is not None:
                for f in self.zs_gt:
                    self._assert_exists(f, f'zs_gt | {f}')

    @staticmethod
    def _assert_exists(filepath: str, name: str = None):
        if filepath is None:
            raise AssertionError(f'Filepath for {name} cannot be None!')

        if is_s3_url(path=filepath):
            s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=filepath)
            s3 = boto3.client('s3')
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    # The key does not exist.
                    raise ValueError(f'Filepath for {name} does not exist in S3! '
                                     f'(filepath="{filepath}")')
                elif e.response['Error']['Code'] == "403":
                    raise ValueError(f'Filepath for {name} does not exist in S3 or you lack permissions to read! '
                                     f'(filepath="{filepath}")')
                else:
                    raise e
        else:
            if not Path(filepath).exists():
                raise ValueError(f'Filepath for {name} does not exist on local filesystem! '
                                 f'(filepath="{filepath}")')

    def exists_all(self, check_zs: bool = True) -> bool:
        required_files = self.get_file_paths(include_zs=check_zs)
        return all(self.exists(f) for f in required_files)

    @staticmethod
    def exists(filepath: str) -> bool:
        if filepath is None:
            raise AssertionError(f'Filepath cannot be None!')
        filepath = str(filepath)

        if is_s3_url(path=filepath):
            s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=filepath)
            s3 = boto3.client('s3')
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            except ClientError as e:
                return False
        else:
            if not Path(filepath).exists():
                return False
        return True

    def load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_results_by_dataset, df_raw, df_metadata = load_results(
            results_by_dataset=self.results_by_dataset,
            raw=self.raw,
            metadata=self.task_metadata,
            metadata_join_column=self.metadata_join_column,
            require_tid_in_metadata=self.task_metadata is not None,
        )
        return df_results_by_dataset, df_raw, df_metadata

    def load_comparison(self) -> pd.DataFrame | None:
        if self.comparison is None:
            return None
        df_results_by_dataset_comparison = load_pd.load(self.comparison)
        df_results_by_dataset_comparison = preprocess_comparison(df_comparison_raw=df_results_by_dataset_comparison, inplace=True)
        return df_results_by_dataset_comparison

    def load_predictions(self,
                         zsc: ZeroshotSimulatorContext,
                         prediction_format: str = "memmap",
                         ) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
        """
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        """
        for f in self.zs_pp:
            self._assert_exists(f, f'zs_pp | {f}')
        for f in self.zs_gt:
            self._assert_exists(f, name=f'zs_gt | {f}')
        zeroshot_pred_proba, zeroshot_gt, zsc = load_zeroshot_input(
            path_pred_proba=self.path_pred_proba,
            paths_gt=self.zs_gt,
            zsc=zsc,
            datasets=self.datasets,
            prediction_format=prediction_format,
        )
        return zeroshot_pred_proba, zeroshot_gt, zsc

    def load_configs(self) -> dict:
        return load_configs(self.configs)


class BenchmarkContext:
    def __init__(self,
                 *,
                 folds: List[int],
                 benchmark_paths: BenchmarkPaths,
                 name: str = None,
                 description: str = None,
                 date: str = None,
                 s3_download_map: Dict[str, str] = None,
                 ):
        self.folds = folds
        self.benchmark_paths = benchmark_paths
        self.name = name
        self.description = description
        self.date = date
        self.s3_download_map = s3_download_map

    @classmethod
    def from_paths(cls,
                   *,
                   folds: List[int],
                   name: str = None,
                   description: str = None,
                   date: str = None,
                   s3_download_map: Dict[str, str] = None,
                   **paths):
        return cls(folds=folds,
                   name=name,
                   description=description,
                   date=date,
                   s3_download_map=s3_download_map,
                   benchmark_paths=BenchmarkPaths(**paths))

    def download(self,
                 include_zs: bool = True,
                 exists: str = 'raise',
                 dry_run: bool = False):
        """
        Downloads all BenchmarkContext required files from s3 to local disk.

        :param include_zs: If True, downloads zpp and gt files if they exist.
        :param exists: This determines the behavior of the file download.
            Options: ['ignore', 'raise', 'overwrite']
            If 'ignore': Will only download missing files and ignore files that already exist locally.
                Note: Does not guarantee local and remote files are identical.
            If 'raise': Will raise an exception if any local files exist. Otherwise, it will download all remote files.
                Guarantees alignment between local and remote files (at the time of download)
            If 'overwrite': Will download all remote files, overwriting any pre-existing local files.
                Guarantees alignment between local and remote files (at the time of download)
        :param dry_run: If True, will not download files, but instead log what would have been downloaded.
        """
        print(f'Downloading files for {self.name} context... '
              f'(include_zs={include_zs}, exists="{exists}", dry_run={dry_run})')
        if dry_run:
            print(f'\tNOTE: `dry_run=True`! Files will not be downloaded.')
        assert exists in ["raise", "ignore", "overwrite"]
        assert self.s3_download_map is not None, \
            f'self.s3_download_map is None: download functionality is disabled'
        file_paths_expected = self.benchmark_paths.get_file_paths(include_zs=include_zs)

        file_paths_to_download = [f for f in file_paths_expected if f in self.s3_download_map]
        if len(file_paths_to_download) == 0:
            print(f'WARNING: Matching file paths to download is 0! '
                  f'`self.s3_download_map` probably has incorrect keys.')
        file_paths_already_exist = [f for f in file_paths_to_download if self.benchmark_paths.exists(f)]
        file_paths_missing = [f for f in file_paths_to_download if not self.benchmark_paths.exists(f)]

        if exists == 'raise':
            if file_paths_already_exist:
                raise AssertionError(f'`exists="{exists}"`, '
                                     f'and found {len(file_paths_already_exist)} files that already exist locally!\n'
                                     f'\tExisting Files: {file_paths_already_exist}\n'
                                     f'\tMissing  Files: {file_paths_missing}\n'
                                     f'Either manually inspect and delete existing files, '
                                     f'set `exists="ignore"` to keep your local files and only download missing files, '
                                     f'or set `exists="overwrite"` to overwrite your existing local files.')
        elif exists == 'ignore':
            file_paths_to_download = file_paths_missing
        elif exists == 'overwrite':
            file_paths_to_download = file_paths_to_download
        else:
            raise ValueError(f'Invalid value for exists (`exists="{exists}"`). '
                             f'Valid values: {["raise", "ignore", "overwrite"]}')

        s3_to_local_tuple_list = [(val, key) for key, val in self.s3_download_map.items()
                                  if key in file_paths_to_download]

        log_extra = ''

        num_exist = len(file_paths_already_exist)
        if exists == 'overwrite':
            if num_exist > 0:
                log_extra += f'\tWill overwrite {num_exist} files that exist locally:\n' \
                            f'\t\t{file_paths_already_exist}'
            else:
                log_extra = f''
        if exists == 'ignore':
            log_extra += f'\tWill skip {num_exist} files that exist locally:\n' \
                            f'\t\t{file_paths_already_exist}'
        if file_paths_missing:
            if log_extra:
                log_extra += '\n'
            log_extra += f'Will download {len(file_paths_missing)} files that are missing locally:\n' \
                         f'\t\t{file_paths_missing}'

        if log_extra:
            print(log_extra)
        print(f'\tDownloading {len(s3_to_local_tuple_list)} files from s3 to local...')
        for s3_path, local_path in s3_to_local_tuple_list:
            print(f'\t\t"{s3_path}" -> "{local_path}"')
        download_s3_files(s3_to_local_tuple_list=s3_to_local_tuple_list, dry_run=dry_run)

    def load(self,
             folds: List[int] = None,
             load_predictions: bool = True,
             download_files: bool = True,
             prediction_format: str = "memmap",
             exists: str = 'ignore') -> Tuple[ZeroshotSimulatorContext, dict, TabularModelPredictions, dict]:
        """
        :param folds: If None, uses self.folds as default.
            If specified, must be a subset of `self.folds`. This will filter the results to only the specified folds.
            Restricting folds can be useful to speed up experiments.
        :param load_predictions: If True, loads zpp and gt files.
        :param download_files: If True, will download required files from s3 if they don't already exist locally.
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        :param exists: If download_files=True, this determines the behavior of the file download.
            Options: ['ignore', 'raise', 'overwrite']
            Refer to `self.download` for details.
        :return: Returns four objects in the following order:
            zsc: ZeroshotSimulatorContext
                The zeroshot simulator context object.
            configs: dict
                # TODO: Consider making a part of zsc.
                The dictionary of config names to hyperparameters.
                This is useful when wanting to understand what hyperparameters a particular model config is using.
            zeroshot_pred_proba: TabularModelPredictions
                # TODO: Consider making a part of zsc.
                The prediction probabilities of all configs for all tasks.
                Will be None if `load_predictions=False`.
            zeroshot_gt : dict
                # TODO: Make its own object instead of a raw dict.
                # TODO: Consider making a part of zsc or zeroshot_pred_proba
                The target ground truth for both validation and test samples for all tasks.
                Will be None if `load_predictions=False`.
        """
        assert prediction_format in ["memmap", "memopt", "mem"]
        if folds is None:
            folds = self.folds
        for f in folds:
            assert f in self.folds, f'Fold {f} does not exist in available folds! self.folds={self.folds}'

        with catchtime("Loading ZS Context"):
            print(f'Loading BenchmarkContext:\n'
                  f'\tname: {self.name}\n'
                  f'\tdescription: {self.description}\n'
                  f'\tdate: {self.date}\n'
                  f'\tfolds: {folds}')
            if download_files and exists == 'ignore':
                if self.benchmark_paths.exists_all(check_zs=load_predictions):
                    print(f'All required files are present...')
                    download_files = False
            if download_files:
                self.benchmark_paths.print_summary()
                print(f'Downloading input files from s3...')
                self.download(include_zs=load_predictions, exists=exists)
            self.benchmark_paths.assert_exists_all(check_zs=load_predictions)

            zsc = self._load_zsc(folds=folds)
            print(f'Loading config hyperparameter definitions... Note: Hyperparameter definitions are only accurate for the latest version.')
            configs_full = self._load_configs()

            if load_predictions:
                zeroshot_pred_proba, zeroshot_gt, zsc = self._load_predictions(zsc=zsc, prediction_format=prediction_format)
            else:
                zeroshot_pred_proba = None
                zeroshot_gt = None

        return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt

    def _load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_results_by_dataset, df_raw, df_metadata = self.benchmark_paths.load_results()
        df_results_by_dataset = combine_results_with_score_val(df_raw, df_results_by_dataset)
        return df_results_by_dataset, df_raw, df_metadata

    def _load_configs(self) -> dict:
        return self.benchmark_paths.load_configs()

    def _load_predictions(self,
                          zsc: ZeroshotSimulatorContext,
                          prediction_format: str) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
        return self.benchmark_paths.load_predictions(zsc=zsc, prediction_format=prediction_format)

    def _load_zsc(self, folds: List[int]) -> ZeroshotSimulatorContext:
        df_results_by_dataset, df_raw, df_metadata = self._load_results()

        # Load in real framework results to score against
        print(f'Loading comparison_frameworks: {self.benchmark_paths.comparison}')
        df_results_by_dataset_automl = self.benchmark_paths.load_comparison()
        zsc = ZeroshotSimulatorContext(
            df_results_by_dataset=df_results_by_dataset,
            df_results_by_dataset_automl=df_results_by_dataset_automl,
            df_raw=df_raw,
            folds=folds,
            df_metadata=df_metadata,
        )
        return zsc


def construct_s3_download_map(
    s3_prefix: str,
    path_context: str,
    split_key: str,
    files_pp: List[str],
    files_gt: List[str]
) -> Dict[str, str]:
    split_value = f"{s3_prefix}model_predictions/"
    s3_download_map = {
        "evaluation/compare/results_ranked_by_dataset_valid.csv": "evaluation/compare/results_ranked_by_dataset_valid.csv",
        "evaluation/configs/results_ranked_by_dataset_all.csv": "evaluation/configs/results_ranked_by_dataset_all.csv",
        "leaderboard_preprocessed_configs.csv": "leaderboard_preprocessed_configs.csv",
    }
    s3_download_map = {f'{path_context}{k}': f'{s3_prefix}{v}' for k, v in s3_download_map.items()}
    _s3_download_map_metadata_pp = {f"{split_key}{f}": f"{split_value}{f}" for f in files_pp}
    _s3_download_map_metadata_gt = {f"{split_key}{f}": f"{split_value}{f}" for f in files_gt}
    s3_download_map.update(_s3_download_map_metadata_pp)
    s3_download_map.update(_s3_download_map_metadata_gt)
    s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in s3_download_map.items()}
    return s3_download_map


def construct_context(
        name: str,
        datasets: List[str],
        folds: List[int],
        local_prefix: str,
        s3_prefix: str = None,
        description: str = None,
        date: str = None,
        task_metadata: str = "task_metadata_244.csv",
        metadata_join_column: str = "dataset",
) -> BenchmarkContext:
    """

    Parameters
    ----------
    name
    description
    date
    datasets
    folds
    local_prefix: str, default = None
        The location for input files to be downloaded to / located.
    s3_prefix: str, default = None
        The s3 location for input files to download from.
        If None, then all files must already exist locally in the `local_prefix` directory.
        Example: "s3://s3_bucket/foo/bar/2023_08_21/"
    task_metadata

    Returns
    -------
    BenchmarkContext object that is able to load the data.
    """
    path_context = str(Paths.results_root / local_prefix) + os.sep

    split_key = str(Path(path_context) / "model_predictions") + os.sep



    files_pred = ["metadata.json", "pred-test.dat", "pred-val.dat"]
    _files_pp = [f"{dataset}/{fold}/{f}" for dataset in datasets for fold in folds for f in files_pred]
    files_label = ["label-test.csv.zip", "label-val.csv.zip"]
    _files_gt = [f"{dataset}/{fold}/{f}" for dataset in datasets for fold in folds for f in files_label]

    if s3_prefix is not None:
        _s3_download_map = construct_s3_download_map(
            s3_prefix=s3_prefix,
            path_context=path_context,
            split_key=split_key,
            files_pp=_files_pp,
            files_gt=_files_gt,
        )
    else:
        _s3_download_map = None

    zs_pp = [f"{split_key}{f}" for f in _files_pp]
    zs_pp = [Paths.rel_to_abs(k, relative_to=Paths.data_root) for k in zs_pp]

    zs_gt = [f"{split_key}{f}" for f in _files_gt]
    zs_gt = [Paths.rel_to_abs(k, relative_to=Paths.data_root) for k in zs_gt]

    _result_paths = dict(
        # results_by_dataset=str(Path(path_context) / "evaluation/configs/results_ranked_by_dataset_all.csv"),
        # comparison=str(Path(path_context) / "evaluation/compare/results_ranked_by_dataset_valid.csv"),
        # raw=str(Path(path_context) / "leaderboard_preprocessed_configs.csv"),
        comparison=str(Path(path_context) / "comparison.parquet"),
        raw=str(Path(path_context) / "raw.parquet"),
    )

    if task_metadata is not None:
        _task_metadata_path = dict(
            task_metadata=str(Paths.data_root / "metadata" / task_metadata),
        )
    else:
        _task_metadata_path = dict()

    _bag_zs_path = dict(
        zs_gt=zs_gt,
        zs_pp=zs_pp,
        path_pred_proba=split_key,
    )

    context: BenchmarkContext = BenchmarkContext.from_paths(
        name=name,
        description=description,
        date=date,
        folds=folds,
        s3_download_map=_s3_download_map,
        datasets=datasets,
        metadata_join_column=metadata_join_column,
        **_result_paths,
        **_bag_zs_path,
        **_task_metadata_path,
    )
    return context

