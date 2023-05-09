from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import boto3
from botocore.errorfactory import ClientError
import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import download_s3_files
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from .utils import load_zeroshot_input
from ..loaders import load_configs, load_results, combine_results_with_score_val, Paths
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..simulation.tabular_predictions import TabularModelPredictions
from ..utils import catchtime


@dataclass
class BenchmarkPaths:
    result: str
    results_by_dataset: str
    raw: str
    comparison: str
    task_metadata: str = None
    zs_pp: str = None
    zs_gt: str = None
    configs: List[str] = None

    def __post_init__(self):
        if self.configs is None:
            configs_prefix_1 = Paths.data_root / 'configs' / 'configs_20221004'
            configs_prefix_2 = Paths.data_root / 'configs'
            configs = [
                f'{configs_prefix_1}/configs_catboost.json',
                f'{configs_prefix_1}/configs_fastai.json',
                f'{configs_prefix_1}/configs_lightgbm.json',
                f'{configs_prefix_1}/configs_nn_torch.json',
                f'{configs_prefix_1}/configs_xgboost.json',
                f'{configs_prefix_2}/configs_rf.json',
                f'{configs_prefix_2}/configs_xt.json',
                f'{configs_prefix_2}/configs_knn.json',
            ]
            self.configs = configs

    def print_summary(self):
        max_str_len = max(len(key) for key in self.__dict__.keys())
        print(f'BenchmarkPaths Summary:')
        print("\n".join(f'\t{key + " "*(max_str_len - len(key))} = {value}' for key, value in self.__dict__.items()))

    def get_file_paths(self, include_zs: bool = True) -> List[str]:
        file_paths = [
            self.result,
            self.results_by_dataset,
            self.raw,
            self.comparison,
            self.task_metadata,
        ]
        if include_zs:
            file_paths += [
                self.zs_pp,
                self.zs_gt,
            ]
        file_paths = [f for f in file_paths if f is not None]
        return file_paths

    def assert_exists_all(self, check_zs=True):
        self._assert_exists(self.result, 'result')
        self._assert_exists(self.results_by_dataset, 'result_by_dataset')
        self._assert_exists(self.raw, 'raw')
        self._assert_exists(self.comparison, 'comparison')
        if self.task_metadata is not None:
            self._assert_exists(self.task_metadata, 'task_metadata')
        if check_zs:
            if self.zs_pp is not None:
                self._assert_exists(self.zs_pp, 'zs_pp')
            if self.zs_gt is not None:
                self._assert_exists(self.zs_gt, 'zs_gt')

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

    def load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_results, df_results_by_dataset, df_raw, df_metadata = load_results(
            results=self.result,
            results_by_dataset=self.results_by_dataset,
            raw=self.raw,
            metadata=self.task_metadata,
            require_tid_in_metadata=self.task_metadata is not None,
        )
        return df_results, df_results_by_dataset, df_raw, df_metadata

    def load_comparison(self) -> pd.DataFrame:
        return load_pd.load(self.comparison)

    def load_predictions(self,
                         zsc: ZeroshotSimulatorContext,
                         lazy_format: bool = False) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
        self._assert_exists(self.zs_pp, name='zs_pp')
        self._assert_exists(self.zs_gt, name='zs_gt')
        zeroshot_pred_proba, zeroshot_gt, zsc = load_zeroshot_input(
            path_pred_proba=self.zs_pp,
            path_gt=self.zs_gt,
            zsc=zsc,
            lazy_format=lazy_format,
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
             lazy_format: bool = False,
             download_files: bool = True,
             exists: str = 'ignore') -> Tuple[ZeroshotSimulatorContext, dict, TabularModelPredictions, dict]:
        """
        :param folds: If None, uses self.folds as default.
            If specified, must be a subset of `self.folds`. This will filter the results to only the specified folds.
            Restricting folds can be useful to speed up experiments.
        :param load_predictions: If True, loads zpp and gt files.
        :param lazy_format: If True, returns model predictions in lazy format, else returns them in memory.
            Ignored if load_predictions=False
        :param download_files: If True, will download required files from s3 if they don't already exist locally.
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
            self.benchmark_paths.print_summary()
            if download_files and exists == 'ignore':
                if self.benchmark_paths.exists_all(check_zs=load_predictions):
                    print(f'All required files are present...')
                    download_files = False
            if download_files:
                print(f'Downloading input files from s3...')
                self.download(include_zs=load_predictions, exists=exists)
            self.benchmark_paths.assert_exists_all(check_zs=load_predictions)

            zsc = self._load_zsc(folds=folds)
            configs_full = self._load_configs()

            if load_predictions:
                zeroshot_pred_proba, zeroshot_gt, zsc = self._load_predictions(zsc=zsc, lazy_format=lazy_format)
            else:
                zeroshot_pred_proba = None
                zeroshot_gt = None

        return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt

    def _load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_results, df_results_by_dataset, df_raw, df_metadata = self.benchmark_paths.load_results()
        df_results_by_dataset = combine_results_with_score_val(df_raw, df_results_by_dataset)
        return df_results_by_dataset, df_raw, df_metadata

    def _load_configs(self) -> dict:
        return self.benchmark_paths.load_configs()

    def _load_predictions(self,
                          zsc: ZeroshotSimulatorContext,
                          lazy_format: bool = False) -> Tuple[TabularModelPredictions, dict, ZeroshotSimulatorContext]:
        return self.benchmark_paths.load_predictions(zsc=zsc, lazy_format=lazy_format)

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
