from __future__ import annotations

from pathlib import Path
from typing import Literal

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

    @property
    def can_hpo(self) -> bool:
        return self.method_type == "config"

    @property
    def _path_root(self) -> Path:
        return Paths.artifacts_root_cache_tabarena

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

    def relative_to_root(self, path: Path) -> Path:
        return path.relative_to(self._path_root)

    def relative_to_method(self, path: Path) -> Path:
        return path.relative_to(self.path)

    def load_model_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_model(holdout=holdout))

    def load_hpo_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_hpo(holdout=holdout))

    def load_portfolio_results(self, holdout: bool = False) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_portfolio(holdout=holdout))
