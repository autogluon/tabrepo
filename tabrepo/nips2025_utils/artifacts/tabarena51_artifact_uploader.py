from __future__ import annotations

import time

from tabrepo.nips2025_utils.artifacts.method_uploader import MethodUploaderS3

from .abstract_artifact_uploader import AbstractArtifactUploader
from .method_metadata import MethodMetadata
from . import tabarena_method_metadata_map


class TabArena51ArtifactUploader(AbstractArtifactUploader):
    def __init__(self):
        self.bucket = "tabarena"
        self.s3_prefix_root = "cache"
        self.upload_as_public = True
        methods = [
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
        self.method_metadata_map = {k: v for k, v in tabarena_method_metadata_map.items() if k in methods}
        self.method_metadata_lst = [self.method_metadata_map[m] for m in methods]

    @property
    def methods(self) -> list[str]:
        return [method_metadata.method for method_metadata in self.method_metadata_lst]

    def _method_metadata(self, method: str) -> MethodMetadata:
        return self.method_metadata_map[method]

    def _method_uploader(self, method: str) -> MethodUploaderS3:
        method_metadata = self._method_metadata(method=method)
        downloader = MethodUploaderS3(
            method_metadata=method_metadata,
            bucket=self.bucket,
            s3_prefix_root=self.s3_prefix_root,
            upload_as_public=self.upload_as_public,
        )
        return downloader

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

    def _upload_raw_method(self, method: str):
        uploader = self._method_uploader(method=method)
        uploader.upload_raw()

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

    def _upload_processed_method(self, method: str):
        uploader = self._method_uploader(method=method)
        uploader.upload_processed()

    def upload_results(self):
        methods = [method for method in self.methods if self._method_metadata(method).has_results]
        for method in methods:
            self._upload_results_method(method=method)

    def _upload_results_method(self, method: str, holdout: bool = False):
        uploader = self._method_uploader(method=method)
        uploader.upload_results(holdout=holdout)
