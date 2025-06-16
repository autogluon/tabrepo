from __future__ import annotations

from pathlib import Path
from typing import Literal

from tabrepo.loaders import Paths


class MethodMetadata:
    def __init__(
        self,
        method: str,
        artifact_name: str,
        *,
        date: str,
        method_type: Literal["config", "baseline"] = "config",
        name_suffix: str | None = None,
        ag_key: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] = "cpu",
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
    def path_results(self) -> Path:
        return self.path / "results"

    @property
    def path_results_hpo(self) -> Path:
        return self.path_results / "hpo_results.parquet"

    @property
    def path_results_model(self) -> Path:
        return self.path_results / "model_results.parquet"

    def relative_to_root(self, path: Path) -> Path:
        return path.relative_to(self._path_root)
