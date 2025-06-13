from __future__ import annotations

from typing import Literal


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
