from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class BetaModel(AbstractModel):
    ag_key = "BETA"
    ag_name = "BetaTabPFN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_col_names_ = None
        self.has_num_cols = None
        self.num_prep_ = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        from sklearn.pipeline import Pipeline

        from tabrepo.benchmark.models.ag.beta.deps.talent_utils import (
            get_deep_args,
            set_seeds,
        )
        from tabrepo.benchmark.models.ag.beta.talent_beta_method import BetaMethod
        from tabrepo.benchmark.models.ag.modernnca.modernnca_model import (
            RTDLQuantileTransformer,
        )

        device = "cpu" if num_gpus == 0 else "cuda:0"
        if (device == "cuda:0") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        # Format data for TALENT code
        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        else:
            raise ValueError("Validation data (X_val) must be provided for BetaModel.")

        self.num_prep_ = Pipeline(
            steps=[
                ("qt", RTDLQuantileTransformer()),
                ("imp", SimpleImputer(add_indicator=True)),
            ]
        )
        self.has_num_cols = bool(set(X.columns) - set(self.cat_col_names_))
        ds_parts = {}
        for part, X_data, y_data in [("train", X, y), ("val", X_val, y_val)]:
            tensors = {}

            tensors["x_cat"] = X_data[self.cat_col_names_].to_numpy()
            if self.has_num_cols:
                x_cont_np = X_data.drop(columns=self.cat_col_names_).to_numpy(
                    dtype=np.float32
                )
                if part == "train":
                    self.num_prep_.fit(x_cont_np)
                tensors["x_cont"] = self.num_prep_.transform(x_cont_np)
            else:
                tensors["x_cont"] = np.empty((len(X_data), 0), dtype=np.float32)

            if self.problem_type == "regression":
                tensors["y"] = y_data.to_numpy(np.float32)
            else:
                tensors["y"] = y_data.to_numpy(np.int32)
            ds_parts[part] = tensors

        data = [
            {part: ds_parts[part][tens_name] for part in ["train", "val"]}
            for tens_name in ["x_cont", "x_cat", "y"]
        ]
        info = {
            "task_type": "binclass"
            if self.problem_type == "binary"
            else self.problem_type,
            "n_num_features": ds_parts["train"]["x_cont"].shape[1],
            "n_cat_features": ds_parts["train"]["x_cat"].shape[1],
        }
        if info["n_num_features"] == 0:
            data[0] = None
        if info["n_cat_features"] == 0:
            data[1] = None
        data = tuple(data)

        # Set up model
        hyp = self._get_model_params()
        args, _, _ = get_deep_args()
        set_seeds(hyp["random_state"])
        args.device = device
        args.max_epoch = hyp["max_epoch"]
        args.batch_size = hyp["batch_size"]
        # TODO: come up with solution to set this based on dataset size
        max_context_size = hyp.get("max_context_size", 1000)

        if info["n_num_features"] > 200:
            # Use less K as otherwise exploding memory constraints
            args.config["model"]["k"] = 10

        args.time_to_fit_in_seconds = time_limit
        args.early_stopping_metric = self.stopping_metric

        save_path = self.path + "/tmp_model"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        args.save_path = str(save_path)

        self.model = BetaMethod(args, self.problem_type == "regression", max_context_size=max_context_size)
        self.model.fit(data=data, info=info, train=True, model_name="best-val")
        shutil.rmtree(save_path, ignore_errors=True)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs).copy()

        # TALENT Format
        tensors = {}
        tensors["x_cat"] = X[self.cat_col_names_].to_numpy()
        tensors["x_cont"] = (
            self.num_prep_.transform(
                X.drop(columns=X[self.cat_col_names_]).to_numpy(dtype=np.float32)
            )
            if self.has_num_cols
            else np.empty((len(X), 0), dtype=np.float32)
        )
        if self.problem_type == "regression":
            tensors["y"] = np.zeros(tensors["x_cat"].shape[0])
        else:
            tensors["y"] = np.zeros(tensors["x_cat"].shape[0], dtype=np.int32)
        data = [{"test": tensors[tens_name]} for tens_name in ["x_cont", "x_cat", "y"]]
        for i in range(2):
            if data[i]["test"].size == 0:
                data[i] = None
        data = tuple(data)

        # AG Predict Output
        y_pred = self.model.predict(data=data, info=None, model_name="best-val")
        if self.problem_type == "regression":
            return y_pred.numpy()

        y_pred_proba = torch.softmax(y_pred, dim=-1).numpy()
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        impute_bool: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        # Ordinal Encoding of cat features but keep as cat
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )
            if self.cat_col_names_ is None:
                self.cat_col_names_ = self._feature_generator.features_in[:]
        else:
            self.cat_col_names_ = []

        return X

    def _set_default_params(self):
        default_params = {
            "random_state": 0,
            "max_epoch": 200,
            "batch_size": 1024,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        import torch

        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 1 if torch.cuda.is_available() else 0
        return num_cpus, num_gpus

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
