from __future__ import annotations


from .abstract_artifact_loader import AbstractArtifactLoader

import requests
import time
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tabrepo.loaders import Paths


class TabArena51ArtifactLoader(AbstractArtifactLoader):
    def __init__(self):
        self.bucket = "tabarena"
        self.prefix = "cache/artifacts"
        self.url_prefix = "https://tabarena.s3.us-west-2.amazonaws.com"
        self.artifact_name = "tabarena-2025-06-12"
        self.local_paths = Paths
        self.local_artifact_cache_path = Paths.artifacts_root_cache_tabarena
        self.methods = [
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

        n_methods = len(self.methods)
        for i, method in enumerate(self.methods):
            print(f"Starting raw artifact download of method {method} ({i+1}/{n_methods})")
            ts = time.time()
            self._download_raw_method(method=method)
            te = time.time()
            time_elapsed = te - ts
            print(
                f"Downloaded raw artifact of method {method} |\t ({i + 1}/{n_methods} complete... "
                f"|\tCompleted in {time_elapsed:.2f}s"
            )

    def _download_raw_method(self, method: str):
        self._download_method(method=method, data_type="raw")

    def download_processed(self):
        download_tabarena51_v2(methods=self.methods, data_type="processed")

    def download_results(self):
        raise NotImplementedError

    def _download_method(self, method: str, data_type: str):
        url_prefix = f"{self.url_prefix}/{self.prefix}/{self.artifact_name}/methods"
        local_method_dir = Paths.artifacts_root_cache_tabarena / self.artifact_name / "methods"

        filename = f"{data_type}.zip"
        url_file = f"{url_prefix}/{method}/{filename}"
        local_dir = local_method_dir / method / data_type
        _download_and_extract_zip(url=url_file, local_dir=local_dir)


def _download_and_extract_zip(url: str, local_dir: str | Path):
    print(f"Beginning download of '{url}', extracting into '{local_dir}'...")

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP request errors

    # Use BytesIO to handle the downloaded content as a file-like object
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(local_dir)  # Extract to the specified directory
        print(f"Extraction complete: '{url}' -> '{local_dir}'")


# TODO: Add check if files already exist
def download_tabarena51_v2(methods, data_type: str = "raw"):
    name = "tabarena-2025-06-12"
    url_prefix = f"https://tabarena.s3.us-west-2.amazonaws.com/artifacts/{name}/methods"
    local_dir = Paths.artifacts_root_cache_tabarena / name / "methods"

    urls = [
        f"{url_prefix}/{method}/{data_type}.zip" for method in methods
    ]

    local_dirs = [local_dir / method / data_type for method in methods]

    print(
        f"======================== READ THIS ========================\n"
        f"Note: Starting download of the complete raw artifacts for the tabarena51 benchmark...\n"
        f"Files will be downloaded from {url_prefix}\n"
        f"Files will be saved to {local_dir}\n"
        f"This will require ~833 GB of disk space and 64+ GB of memory.\n"
        f"With a fast internet connection, this will take about 12 hours.\n"
        f"Only do this if you are interested in indepth analysis of the results or want to exactly reproduce the results from the original raw files.\n"
        f"The raw files are stored in pickle files for convenience. Pickle files are capable of executing arbitrary code on launch. Only load data that you trust.\n"
        f"If you are interested in portfolio building, you do not need the raw results, "
        f"instead you can get the processed results, which are only 100 GB. (Instructions TBD)\n"
        f"==========================================================="
    )

    print(f"Files to download and extract: {urls}")
    for url, cur_local_dir in zip(urls, local_dirs):
        ts = time.time()
        _download_and_extract_zip(url=url, local_dir=cur_local_dir)
        te = time.time()
        time_taken = te - ts
        print(f"Downloaded {url} to {cur_local_dir} |\tCompleted in {time_taken:.2f}s")

    print("Done")
