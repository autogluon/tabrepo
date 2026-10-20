"""AutoGluon model wrapper for CompactTabFoundation.

CompactTabFoundation is a parameter-efficient in-context tabular foundation model.
It formulates tabular prediction as zero-shot inference over row and column tokens
using a Structural Causal Model (SCM) transformer architecture.

Codebase: https://github.com/DakshBhalala/TBDML
License: Apache-2.0 (code); OpenMDW-1.1 (open weights)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

# Developer fix: monkey-patch torch.cuda.get_per_process_memory_fraction for PyTorch on Windows
if torch.cuda.is_available() and not hasattr(torch.cuda, "get_per_process_memory_fraction"):
    torch.cuda.get_per_process_memory_fraction = lambda d=None: 1.0  # type: ignore

logger = logging.getLogger(__name__)


class CompactTabFoundationModel(AbstractModel):
    """CompactTabFoundation: In-Context Tabular Foundation Model.

    Evaluates zero-shot predictive inference across diverse tabular schemas in a single
    forward pass without dataset-specific gradient updates.

    Paper: CompactTab: State-of-the-Art Compact Tabular Deep Learning & Foundation Models
    Authors: Daksh Bhalala et al.
    Codebase: https://github.com/DakshBhalala/TBDML
    License: Apache-2.0 (code); OpenMDW-1.1 (open weights)
    """

    ag_key = "TA-COMPACTTAB"
    ag_name = "TA-CompactTabFoundation"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS, REGRESSION]

    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 0

    _default_ag_args_ensemble_extra: ClassVar[dict[str, Any]] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    warmup_modules: ClassVar[tuple[str, ...]] = (
        "torch",
        "sdm",
        "sdm.models",
    )

    size: str = "small"
    default_num_estimators: int = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.x_context = None
        self.y_context = None
        self.classes_ = None
        self._target_col = "__target__"
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []

    def _get_default_auxiliary_params(self) -> dict:
        default_aux = super()._get_default_auxiliary_params()
        default_aux.update(
            {
                "valid_raw_types": ["int", "float", "category", "object"],
                "ignored_type_group_raw": [],
            }
        )
        return default_aux

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _set_default_params(self):
        self._set_default_param_value("size", self.size)
        self._set_default_param_value("num_estimators", self.default_num_estimators)

    def _prepare_table(self, X: pd.DataFrame, y: pd.Series | None = None) -> tuple[Any, Any | None]:
        """Convert pandas inputs to sdm TableTensor with explicit schema mapping."""
        import torch
        from sdm import Stype, TableTensor

        device = torch.device(self.get_device())
        df_x = X.copy().reset_index(drop=True)

        stypes: dict[str, Stype] = {}
        for col in self._num_cols:
            if col in df_x.columns:
                stypes[col] = Stype.numerical
        for col in self._cat_cols:
            if col in df_x.columns:
                stypes[col] = Stype.categorical

        if y is not None:
            if self.problem_type in (BINARY, MULTICLASS):
                if self.classes_ is None:
                    self.classes_ = np.sort(np.unique(np.asarray(y)))
                df_x[self._target_col] = pd.Categorical(np.asarray(y), categories=self.classes_)
                stypes[self._target_col] = Stype.categorical
            else:
                df_x[self._target_col] = np.asarray(y, dtype=np.float32)
                stypes[self._target_col] = Stype.numerical

        table = TableTensor.from_pandas(df_x, stypes=stypes, device=device)

        if y is not None:
            y_table = table.select_columns(self._target_col)
            x_table = table.drop_columns(self._target_col)
            return x_table, y_table

        return table, None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs,
    ) -> None:
        """Fit in-context model by caching the context table representation."""
        import sdm.models
        import torch

        device_str = "cuda" if (num_gpus and num_gpus > 0 and torch.cuda.is_available()) else "cpu"
        self._set_device(device_str)

        self._num_cols = list(X.select_dtypes(include=["number"]).columns)
        self._cat_cols = list(X.select_dtypes(exclude=["number"]).columns)

        if self.problem_type in (BINARY, MULTICLASS):
            self.classes_ = np.sort(np.unique(np.asarray(y)))

        params = self._get_model_params()
        size_name = params.get("size", self.size)
        task_str = "regression" if self.problem_type == REGRESSION else "classification"
        self.model = sdm.models.KumoTabular(
            task=task_str,
            size=size_name,
            pretrained=True,
            device=device_str,
        )

        self.x_context, self.y_context = self._prepare_table(X, y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """In-context probability estimation or regression output."""
        if self.problem_type == REGRESSION:
            return self._predict(X, **kwargs)

        import sdm
        import torch

        if self.model is None or self.x_context is None or self.y_context is None:
            raise RuntimeError("Model has not been fitted.")

        x_query, _ = self._prepare_table(X)
        params = self._get_model_params()
        num_estimators = params.get("num_estimators", self.default_num_estimators)
        device = torch.device(self.get_device())

        with (
            torch.inference_mode(),
            torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
        ):
            probs_table = self.model(
                x_context=self.x_context,
                y_context=self.y_context,
                x_query=x_query,
                num_estimators=num_estimators,
            )

        probs_tensor = probs_table.numerical
        probs = probs_tensor.cpu().detach().float().numpy()

        # Reorder probabilities to match sorted self.classes_ order using output metadata
        if self.classes_ is not None and hasattr(probs_table, "columns") and sdm.Stype.numerical in probs_table.columns:
            out_cols = probs_table.columns[sdm.Stype.numerical]
            class_to_col: dict[Any, int] = {}
            for idx, name in enumerate(out_cols):
                for c in self.classes_:
                    if str(c) == str(name):
                        class_to_col[c] = idx
                        break
            if len(class_to_col) == len(self.classes_):
                col_order = [class_to_col[c] for c in self.classes_]
                probs = probs[:, col_order]

        if self.problem_type == BINARY and probs.shape[1] > 2:
            probs = probs[:, :2]

        row_sums = probs.sum(axis=-1, keepdims=True)
        return np.where(row_sums > 0, probs / np.maximum(row_sums, 1e-12), 1.0 / probs.shape[1])

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Point prediction for regression or argmax label for classification."""
        if self.problem_type in (BINARY, MULTICLASS):
            probs = self._predict_proba(X, **kwargs)
            class_indices = np.argmax(probs, axis=1)
            if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
                return self.classes_[class_indices]
            return class_indices

        import torch

        if self.model is None or self.x_context is None or self.y_context is None:
            raise RuntimeError("Model has not been fitted.")

        x_query, _ = self._prepare_table(X)
        params = self._get_model_params()
        num_estimators = params.get("num_estimators", self.default_num_estimators)
        device = torch.device(self.get_device())

        with (
            torch.inference_mode(),
            torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
        ):
            preds_table = self.model(
                x_context=self.x_context,
                y_context=self.y_context,
                x_query=x_query,
                num_estimators=num_estimators,
            )

        quantiles = preds_table.numerical.cpu().detach().float().numpy()
        return np.mean(quantiles, axis=1)

    def get_device(self) -> str:
        """Return the device string."""
        if self.model is not None:
            param = next(self.model.parameters(), None)
            if param is not None:
                return str(param.device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _set_device(self, device: str):
        """Move network and stored context to target device."""
        import torch

        target_device = torch.device(device)
        if self.model is not None:
            self.model.to(target_device)
        if self.x_context is not None:
            self.x_context = self.x_context.to(target_device)
        if self.y_context is not None:
            self.y_context = self.y_context.to(target_device)


class CompactTabFoundationMediumModel(CompactTabFoundationModel):
    """CompactTabFoundation with medium checkpoint (~62M parameters, 1902 target Elo)."""

    ag_key = "TA-COMPACTTAB-MEDIUM"
    ag_name = "TA-CompactTabFoundation-Medium"
    size = "medium"
    default_num_estimators = 8
