from __future__ import annotations

import json
from pathlib import Path
import requests
from typing import Literal

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
import pandas as pd

from tabrepo.loaders import Paths


class MethodMetadata:
    def __init__(
        self,
        method: str,
        artifact_name: str,
        *,
        date: str,
        method_type: Literal["config", "baseline", "portfolio"] = "config",
        name_suffix: str | None = None,
        ag_key: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] = "cpu",
        is_bag: bool = False,
        has_raw: bool = False,
        has_processed: bool = False,
        has_results: bool = False,
        upload_as_public: bool = False,
    ):
        self.method = method
        self.artifact_name = artifact_name
        self.date = date
        self.method_type = method_type
        self.ag_key = ag_key
        self.name_suffix = name_suffix
        self.config_default = config_default
        self.compute = compute
        self.is_bag = is_bag
        self.has_raw = has_raw
        self.has_processed = has_processed
        self.has_results = has_results
        self.upload_as_public = upload_as_public

    @property
    def can_hpo(self) -> bool:
        return self.method_type == "config"

    @property
    def has_configs_hyperparameters(self) -> bool:
        return self.method_type == "config"

    @property
    def _path_root(self) -> Path:
        return Paths.artifacts_root_cache_tabarena

    @property
    def path_cache_root(self) -> Path:
        return Paths._tabarena_root_cache

    @property
    def path(self) -> Path:
        return self._path_root / self.artifact_name / "methods" / self.method

    @property
    def path_raw(self) -> Path:
        return self.path / "raw"

    @property
    def path_processed(self) -> Path:
        return self.path / "processed"

    @property
    def path_processed_holdout(self) -> Path:
        return self.path / "processed_holdout"

    @property
    def path_results(self) -> Path:
        return self.path / "results"

    @property
    def path_results_holdout(self) -> Path:
        return self.path_results / "holdout"

    def path_results_hpo(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "hpo_results.parquet"

    def path_results_model(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "model_results.parquet"

    def path_results_portfolio(self, holdout: bool = False) -> Path:
        path_prefix = self.path_results_holdout if holdout else self.path_results
        return path_prefix / "portfolio_results.parquet"

    def relative_to_cache_root(self, path: Path) -> Path:
        return path.relative_to(self.path_cache_root)

    def relative_to_root(self, path: Path) -> Path:
        return path.relative_to(self._path_root)

    def relative_to_method(self, path: Path) -> Path:
        return path.relative_to(self.path)

    def to_s3_cache_loc(self, path: Path, s3_cache_root: str) -> str:
        path_suffix = self.relative_to_cache_root(path=path)
        s3_cache_path = f"{s3_cache_root}/{path_suffix}"
        return s3_cache_path

    def load_model_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_model(holdout=holdout))

    def load_hpo_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_hpo(holdout=holdout))

    def load_portfolio_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_portfolio(holdout=holdout))

    def path_configs_hyperparameters(self, holdout: bool = False) -> Path:
        if holdout:
            path_processed = self.path_processed_holdout
        else:
            path_processed = self.path_processed
        path_configs_hyperparameters = path_processed / "configs_hyperparameters.json"
        return path_configs_hyperparameters

    def load_configs_hyperparameters(self, holdout: bool = False) -> dict[str, dict]:
        with open(self.path_configs_hyperparameters(holdout=holdout), "r") as f:
            out = json.load(f)
        return out

    def download_configs_hyperparameters(self, s3_cache_root: str, holdout: bool = False):
        path_local = self.path_configs_hyperparameters(holdout=holdout)
        s3_path_loc = self.to_s3_cache_loc(path=path_local, s3_cache_root=s3_cache_root)
        self._download_file(url=s3_path_loc, local_path=path_local)

    def upload_configs_hyperparameters(self, s3_cache_root: str, holdout: bool = False):
        path_local = self.path_configs_hyperparameters(holdout=holdout)
        s3_path_loc = self.to_s3_cache_loc(path=path_local, s3_cache_root=s3_cache_root)
        self._upload_file(file_name=path_local, s3_path=s3_path_loc)

    def _upload_file(self, file_name: str | Path, s3_path: str):
        import boto3

        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}
        bucket, prefix = s3_path_to_bucket_prefix(s3_path)

        # Upload the file
        s3_client = boto3.client("s3")
        s3_client.upload_file(Filename=file_name, Bucket=bucket, Key=prefix, **kwargs)

    def _download_file(self, url: str, local_path: str | Path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
