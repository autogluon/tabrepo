from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

from autogluon.common.utils.s3_utils import upload_file

from tabrepo.loaders import Paths

from .abstract_artifact_uploader import AbstractArtifactUploader
from .method_metadata import MethodMetadata
from . import tabarena_method_metadata_map


# TODO: Make this use MethodUploaderS3
class TabArena51ArtifactUploader(AbstractArtifactUploader):
    def __init__(self):
        self.artifact_name = "tabarena-2025-06-12"
        self.bucket = "tabarena"
        self.s3_cache_root_prefix = "cache"
        self.s3_cache_root = f"s3://{self.bucket}/{self.s3_cache_root_prefix}"
        self.prefix = f"{self.s3_cache_root_prefix}/artifacts/{self.artifact_name}"
        self.local_paths = Paths
        self.upload_as_public = True
        self.method_metadata_map = tabarena_method_metadata_map
        self.methods = [
            "AutoGluon_v130",
            "Portfolio-N200-4h",
            "CatBoost",
            "Dummy",
            "ExplainableBM",
            "ExtraTrees",
            "KNeighbors",
            "LightGBM",
            "LinearModel",
            "ModernNCA",
            "NeuralNetFastAI",
            "NeuralNetTorch",
            "RandomForest",
            "RealMLP",
            "TabM",
            "XGBoost",

            # "Mitra_GPU",
            "ModernNCA_GPU",
            "RealMLP_GPU",
            "TabDPT_GPU",
            "TabICL_GPU",
            "TabM_GPU",
            "TabPFNv2_GPU",
        ]

    def _method_metadata(self, method: str) -> MethodMetadata:
        return self.method_metadata_map[method]

    def upload_raw(self):
        methods = [method for method in self.methods if self._method_metadata(method).has_raw]
        n_methods = len(methods)

        for i, method in enumerate(methods):
            print(f"Starting raw artifact upload of method {method} ({i+1}/{n_methods})")
            ts = time.time()
            self._upload_raw_method(method=method)
            te = time.time()
            time_elapsed = te - ts
            print(f"Uploaded raw artifact of method {method} ({i+1}/{n_methods} complete) |\tCompleted in {time_elapsed:.2f}s")

    # FIXME: Update this
    def _upload_raw_method(self, method: str):
        metadata = self._method_metadata(method=method)
        path_raw = metadata.path_raw

        relative_to_root = metadata.relative_to_root(metadata.path)
        s3_path = str(Path("cache") / "artifacts" / relative_to_root)

        tmp_dir = Path("tmp")
        file_prefix = tmp_dir / method / "raw"
        shutil.make_archive(file_prefix, 'zip', root_dir=path_raw)
        file_name = f"{file_prefix}.zip"

        self._upload_file(file_name=file_name, prefix=s3_path)
        os.remove(file_name)

        # method_uplodaer = MethodUploader(
        #     method=method,
        #     bucket=self.bucket,
        #     prefix=prefix,
        #     local_path=path_data_method,
        #     upload_as_public=self.upload_as_public,
        # )
        #
        # method_uplodaer.upload_raw()

    def _upload_file(self, file_name: str | Path, prefix: str):
        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}
        upload_file(file_name=file_name, bucket=self.bucket, prefix=prefix, **kwargs)

    def upload_processed(self):
        methods = [method for method in self.methods if self._method_metadata(method).has_processed]
        n_methods = len(methods)

        for i, method in enumerate(methods):
            print(f"Starting processed artifact upload of method {method}")
            ts = time.time()
            self._upload_processed_method(method=method)
            te = time.time()
            time_elapsed = te - ts
            print(f"Uploaded processed artifact of method {method} |\t ({i+1}/{n_methods} complete... |\tCompleted in {time_elapsed:.2f}s")

    def _upload_processed_holdout_method(self, method: str):
        metadata = self._method_metadata(method=method)
        path_processed_holdout = metadata.path_processed_holdout

        relative_to_root = metadata.relative_to_root(metadata.path)
        s3_path = str(Path("cache") / "artifacts" / relative_to_root)

        tmp_dir = Path("~/tabarena_tmp")
        file_prefix = tmp_dir / metadata.artifact_name / metadata.method / "processed_holdout"
        shutil.make_archive(file_prefix, 'zip', root_dir=path_processed_holdout)
        file_name = f"{file_prefix}.zip"

        self._upload_file(file_name=file_name, prefix=s3_path)
        metadata.upload_configs_hyperparameters(s3_cache_root=self.s3_cache_root, holdout=True)

        os.remove(file_name)

    def _upload_processed_method(self, method: str):
        metadata = self._method_metadata(method=method)
        path_processed = metadata.path_processed

        relative_to_root = metadata.relative_to_root(metadata.path)
        s3_path = str(Path("cache") / "artifacts" / relative_to_root)

        tmp_dir = Path("~/tabarena_tmp")
        file_prefix = tmp_dir / metadata.artifact_name / metadata.method / "processed"
        shutil.make_archive(file_prefix, 'zip', root_dir=path_processed)
        file_name = f"{file_prefix}.zip"

        self._upload_file(file_name=file_name, prefix=s3_path)
        metadata.upload_configs_hyperparameters(s3_cache_root=self.s3_cache_root, holdout=False)

        os.remove(file_name)

    def upload_results(self):
        methods = [method for method in self.methods if self._method_metadata(method).has_results]
        for method in methods:
            self._upload_results_method(method=method)

    def _upload_results_method(self, method: str, holdout: bool = False):
        metadata = self._method_metadata(method=method)
        if holdout:
            path_results = metadata.path_results_holdout
        else:
            path_results = metadata.path_results

        relative_to_root = metadata.relative_to_root(path_results)
        s3_path = str(Path("cache") / "artifacts" / relative_to_root)

        if metadata.method_type == "portfolio":
            file_names = [
                "portfolio_results.parquet"
            ]
        else:
            file_names = [
                "model_results.parquet"
            ]

        if metadata.method_type == "config":
            file_names.append("hpo_results.parquet")

        for file_name in file_names:
            local_file_path = path_results / file_name
            self._upload_file(file_name=local_file_path, prefix=s3_path)
