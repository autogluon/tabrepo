"""Partially adapted from pytabkit's TabM implementation."""

from __future__ import annotations

import logging
import time
from typing import Literal

import pandas as pd
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

TaskType = Literal["regression", "binclass", "multiclass"]

logger = logging.getLogger(__name__)


class TabMModel(AbstractModel):
    ag_key = "TABM"
    ag_name = "TabM"
    ag_priority = 85

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

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
        start_time = time.time()

        # imports various dependencies such as torch
        from tabrepo.benchmark.models.ag.tabm._tabm_internal import TabMImplementation
        from torch.cuda import is_available

        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X_train, X_val, y_train, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = TabMImplementation(
            n_threads=num_cpus,
            device=device,
            problem_type=self.problem_type,
            early_stopping_metric=self.stopping_metric,
            **hyp,
        )

        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=X.select_dtypes(include="category").columns.tolist(),
            time_to_fit_in_seconds=time_limit - (time.time() - start_time) if time_limit is not None else None,
        )

    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X = X.copy(deep=True)
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X

    def _set_default_params(self):
        default_params = dict(
            random_state=0,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = min(ResourceManager.get_gpu_count_torch(), 1)
        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    # FIXME: Find a better estimate for memory usage of TabM. Currently borrowed from FASTAI estimate.
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        return 10 * get_approximate_df_mem_usage(X).sum()

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to force stopping at a specific epoch?
        return {"can_refit_full": False}
