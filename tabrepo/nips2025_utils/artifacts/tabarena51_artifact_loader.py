from __future__ import annotations


from .abstract_artifact_loader import AbstractArtifactLoader

import requests
import time
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tabrepo.loaders import Paths
from . import tabarena_method_metadata_map
from .method_metadata import MethodMetadata


class TabArena51ArtifactLoader(AbstractArtifactLoader):
    def __init__(self):
        self.bucket = "tabarena"
        self.prefix = Path("cache") / "artifacts"
        self.url_prefix = "https://tabarena.s3.us-west-2.amazonaws.com"
        self.artifact_name = "tabarena-2025-06-12"
        self.local_paths = Paths
        self.local_artifact_cache_path = Paths.artifacts_root_cache_tabarena
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

    def download_raw(self):
        print(
            f"======================== READ THIS ========================\n"
            f"Note: Starting download of the complete raw artifacts for the tabarena51 benchmark...\n"
            f"Files will be downloaded from {self.url_prefix}/{self.prefix}/{self.artifact_name}\n"
            f"Files will be saved to {self.local_artifact_cache_path}/{self.artifact_name}\n"
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
        self._download_method(method=method, data_type="raw")

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
        self._download_method(method=method, data_type="processed")

    def download_results(self, holdout: bool = False):
        methods = [method for method in self.methods if self._method_metadata(method).has_results]
        n_methods = len(methods)
        for i, method in enumerate(methods):
            print(f"Starting results artifact download of method {method} ({i + 1}/{n_methods})")
            self._download_results_method(method=method, holdout=holdout)
            print(
                f"\tDownloaded results artifact of method {method} ({i + 1}/{n_methods} complete)"
            )

    # TODO: Add the download logic to the method metadata and call that instead
    def _download_results_method(self, method: str, holdout: bool = False):
        metadata = self._method_metadata(method=method)
        if holdout and not metadata.is_bag:
            return
        url_prefix = f"{self.url_prefix}/{self.prefix.as_posix()}/{self.artifact_name}/methods/{method}"
        if metadata.method_type == "config":
            path_hpo = metadata.path_results_hpo(holdout=holdout)
            url_prefix_full = f"{url_prefix}/{metadata.relative_to_method(path_hpo).as_posix()}"
            _download_file(url=url_prefix_full, local_path=path_hpo)
        if metadata.method_type == "portfolio":
            path_portfolio = metadata.path_results_portfolio(holdout=holdout)
            url_prefix_full = f"{url_prefix}/{metadata.relative_to_method(path_portfolio).as_posix()}"
            _download_file(url=url_prefix_full, local_path=path_portfolio)
        else:
            path_model = metadata.path_results_model(holdout=holdout)
            url_prefix_full = f"{url_prefix}/{metadata.relative_to_method(path_model).as_posix()}"
            _download_file(url=url_prefix_full, local_path=path_model)

    def _download_method(self, method: str, data_type: str):
        url_prefix = f"{self.url_prefix}/{self.prefix.as_posix()}/{self.artifact_name}/methods"
        local_method_dir = Paths.artifacts_root_cache_tabarena / self.artifact_name / "methods"

        filename = f"{data_type}.zip"
        url_file = f"{url_prefix}/{method}/{filename}"
        local_dir = local_method_dir / method / data_type
        _download_and_extract_zip(url=url_file, local_dir=local_dir)


def _download_and_extract_zip(url: str, local_dir: str | Path):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Beginning download of '{url}', extracting into '{local_dir}'...")

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP request errors

    # Use BytesIO to handle the downloaded content as a file-like object
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(local_dir)  # Extract to the specified directory
        print(f"Extraction complete: '{url}' -> '{local_dir}'")


def _download_file(url: str, local_path: str | Path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP request errors

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive chunks
                f.write(chunk)
