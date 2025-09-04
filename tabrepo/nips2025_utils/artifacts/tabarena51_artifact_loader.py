from __future__ import annotations


from .abstract_artifact_loader import AbstractArtifactLoader

import time

from autogluon.common.utils.s3_utils import s3_bucket_prefix_to_path

from tabrepo.loaders import Paths
from tabrepo.nips2025_utils.artifacts.method_downloader import MethodDownloaderS3
from . import tabarena_method_metadata_map
from .method_metadata import MethodMetadata


class TabArena51ArtifactLoader(AbstractArtifactLoader):
    def __init__(self):
        self.bucket = "tabarena"
        self.s3_prefix_root = "cache"
        self.local_paths = Paths
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

    def _method_downloader(self, method: str) -> MethodDownloaderS3:
        method_metadata = self._method_metadata(method=method)
        downloader = MethodDownloaderS3(
            method_metadata=method_metadata,
            bucket=self.bucket,
            s3_prefix_root=self.s3_prefix_root,
        )
        return downloader

    def download_raw(self):
        print(
            f"======================== READ THIS ========================\n"
            f"Note: Starting download of the complete raw artifacts for the tabarena51 benchmark...\n"
            f"Files will be downloaded from {s3_bucket_prefix_to_path(self.bucket, self.s3_prefix_root)}\n"
            f"Files will be saved to {Paths._tabarena_root_cache}\n"
            f"This will require ~833 GB of disk space and 64+ GB of memory.\n"
            f"With a fast internet connection, this will take about 7 hours.\n"
            f"Only do this if you are interested in indepth analysis of the results or want to exactly reproduce the results from the original raw files.\n"
            f"The raw files are stored in pickle files for convenience. Pickle files are capable of executing arbitrary code on launch. Only load data that you trust.\n"
            f"If you are interested in portfolio building, you do not need the raw results, "
            f"instead you can get the processed results, which are only 100 GB. (Instructions TBD)\n"
            f"==========================================================="
        )
        methods = [method for method in self.methods if self._method_metadata(method).has_raw]
        n_methods = len(self.methods)
        for i, method in enumerate(methods):
            print(f"Starting raw artifact download of method {method} ({i+1}/{n_methods})")
            ts = time.time()
            self._download_raw_method(method=method)
            te = time.time()
            time_elapsed = te - ts
            print(
                f"Downloaded raw artifact of method {method} ({i + 1}/{n_methods} complete)"
                f" |\tCompleted in {time_elapsed:.2f}s"
            )

    def _download_raw_method(self, method: str):
        downloader = self._method_downloader(method=method)
        downloader.download_raw()

    def download_processed(self):
        methods = [method for method in self.methods if self._method_metadata(method).has_processed]
        n_methods = len(methods)
        for i, method in enumerate(methods):
            print(f"Starting processed artifact download of method {method} ({i + 1}/{n_methods})")
            ts = time.time()
            self._download_processed_method(method=method)
            te = time.time()
            time_elapsed = te - ts
            print(
                f"Downloaded processed artifact of method {method} ({i + 1}/{n_methods} complete)"
                f" |\tCompleted in {time_elapsed:.2f}s"
            )

    def _download_processed_method(self, method: str):
        downloader = self._method_downloader(method=method)
        downloader.download_processed()

    def download_results(self, holdout: bool = False):
        methods = [method for method in self.methods if self._method_metadata(method).has_results]
        n_methods = len(methods)
        for i, method in enumerate(methods):
            print(f"Starting results artifact download of method {method} ({i + 1}/{n_methods})")
            self._download_results_method(method=method, holdout=holdout)
            print(
                f"\tDownloaded results artifact of method {method} ({i + 1}/{n_methods} complete)"
            )

    def _download_results_method(self, method: str, holdout: bool = False):
        metadata = self._method_metadata(method=method)
        if holdout and not metadata.is_bag:
            return
        downloader = self._method_downloader(method=method)
        downloader.download_results(holdout=holdout)
