from __future__ import annotations

import copy
import logging
import math
import time
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from autogluon.core.constants import REGRESSION
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.tabular import __version__
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

@contextmanager
def set_logger_level(logger_name: str, level: int):
    _logger = logging.getLogger(logger_name)
    old_level = _logger.level
    _logger.setLevel(level)
    try:
        yield
    finally:
        _logger.setLevel(old_level)


class xRFMImplementation:
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.kwargs = kwargs

    def fit(self, X, y, X_val, y_val):
        import xrfm
        import torch

        # preprocessing

        self.cat_cols_ = X.select_dtypes(include=["category", "string", "object"]).columns.tolist()
        self.num_cols_ = X.select_dtypes(exclude=["category", "string", "object"]).columns.tolist()
        # todo: 'ignore' may be problematic for the automatic categorical handling of xRFM?
        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler_ = StandardScaler()

        x_cat_enc = self.ohe_.fit_transform(X.loc[:, self.cat_cols_])
        x_num_enc = X.loc[:, self.num_cols_].to_numpy().astype(np.float32)

        x_val_cat_enc = self.ohe_.transform(X_val.loc[:, self.cat_cols_])
        x_val_num_enc = X_val.loc[:, self.num_cols_].to_numpy().astype(np.float32)

        if len(self.cat_cols_) == 0:
            X = x_num_enc
            X_val = x_val_num_enc
        elif len(self.num_cols_) == 0:
            X = x_cat_enc
            X_val = x_val_cat_enc
        else:
            X = np.concatenate([x_cat_enc, x_num_enc], axis=1)
            X_val = np.concatenate([x_val_cat_enc, x_val_num_enc], axis=1)
        X = self.scaler_.fit_transform(X)
        X_val = self.scaler_.transform(X_val)

        X = torch.from_numpy(X).float()
        X_val = torch.from_numpy(X_val).float()

        if self.problem_type == REGRESSION:
            # standardize regression target (may not be necessary)
            y = np.asarray(y)
            self.mean_ = np.mean(y, axis=0)
            self.std_ = np.std(y, axis=0) + 1e-30
            y = (y - self.mean_) / self.std_
            if y_val is not None:
                y_val = np.asarray(y_val)
                y_val = (y_val - self.mean_) / self.std_

        y = torch.from_numpy(y).float()
        y_val = torch.from_numpy(y_val).float()

        self.model = xrfm.xRFM(
            # categorical_info=self.categorical_info_,  # set in preprocess()
            **self.kwargs,
        )

        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

    def preprocess_transform(self, X):
        x_cat_enc = self.ohe_.transform(X.loc[:, self.cat_cols_])
        x_num_enc = X.loc[:, self.num_cols_].to_numpy().astype(np.float32)
        X = np.concatenate([x_cat_enc, x_num_enc], axis=1)
        X = self.scaler_.transform(X)
        return X

    def predict(self, X):
        y_pred = np.squeeze(self.model.predict(self.preprocess_transform(X)), axis=1)
        if self.problem_type == REGRESSION:
            y_pred = y_pred * self.std_ + self.mean_
        return y_pred

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocess_transform(X))



# pip install xrfm
class xRFMModel(AbstractModel):

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
        time_limit: float = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()

        import torch

        # FIXME: code assume we only see one GPU in the fit process.
        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()

        metric_map = {
            "roc_auc": "auc",
            "accuracy": "accuracy",
            "balanced_accuracy": "brier",
            "log_loss": "logloss",
            "rmse": "mse",
            "root_mean_squared_error": "mse",
            "r2": "mse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        tuning_metric = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = copy.copy(hyp)

        if tuning_metric is not None:
            init_kwargs["tuning_metric"] = tuning_metric

        bool_to_cat = hyp.pop("bool_to_cat", False)
        impute_bool = hyp.pop("impute_bool", True)

        init_kwargs['rfm_params'] = {
            'model': {
                'kernel': init_kwargs.get('kernel', 'l2'),
                'bandwidth': init_kwargs.get('bandwidth', 10.0),
                'exponent': init_kwargs.get('exponent', 1.0),
                'diag': init_kwargs.get('diag', True),
                'bandwidth_mode': init_kwargs.get('bandwidth_mode', 'constant'),
            },
            'fit': {
                'reg': init_kwargs.get('reg', 1e-5),
                'iters': init_kwargs.get('iters', 5),
                'M_batch_size': init_kwargs.get('M_batch_size', len(X)),  # todo: adjust this dynamically!
                'verbose': False,
                'early_stop_rfm': init_kwargs.get('early_stop_rfm', True),
                'early_stop_multiplier': init_kwargs.get('early_stop_multiplier', 1.1),
            }
        }

        # todo: should we set default_rfm_params as well? -> I guess not?
        # todo: set M_batch_size?
        # todo: do we already need to move stuff to the correct device?

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat, impute_bool=impute_bool)

        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = xRFMImplementation(
            problem_type=self.problem_type,
            n_threads=num_cpus,
            device=device,
            random_state=self.random_seed,
            time_limit_s=time_limit - (time.time() - start_time) if time_limit is not None else None,
            **init_kwargs,
        )

        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        return super()._predict_proba(X=X, kwargs=kwargs)

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, bool_to_cat: bool = False, impute_bool: bool = True, **kwargs) -> np.ndarray:
        """
        Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        # FIXME: is copy needed?
        X = X.copy(deep=True)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
            if impute_bool:  # Technically this should do nothing useful because bools will never have NaN
                self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"])
                self._features_to_keep = self._feature_metadata.get_features(invalid_raw_types=["int", "float"])
            else:
                self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"], invalid_special_types=["bool"])
                self._features_to_keep = [f for f in self._feature_metadata.get_features() if f not in self._features_to_impute]
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [c for c in self._imputer.get_feature_names_out() if c not in self._features_to_impute]
        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            X_impute = pd.DataFrame(X_impute, index=X.index, columns=self._imputer.get_feature_names_out())
            if self._indicator_columns:
                # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
                # TODO: Add to features_bool?
                X_impute[self._indicator_columns] = X_impute[self._indicator_columns].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X

    def _get_random_seed_from_hyperparameters(self, hyperparameters: dict) -> int | None | str:
        return hyperparameters.get("random_state", "N/A")

    def _set_default_params(self):
        default_params = dict(
            # copied from RealMLP
            impute_bool=False,
            name_categories=True,
            bool_to_cat=False,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        **kwargs,
    ) -> int:
        """
        Heuristic memory estimate.
        ```

        """
        if hyperparameters is None:
            hyperparameters = {}

        num_features = len(X.columns)
        num_samples = len(X)

        # taken from pytabkit
        columns_mem_est = 6e4 * num_samples + 2e6 * num_features + 30.0 * num_samples**2 + 30.0 * num_samples * num_features

        dataset_size_mem_est = 5 * get_approximate_df_mem_usage(X).sum()  # roughly 5x DataFrame memory size
        baseline_overhead_mem_est = 2e8  # 200 MB generic overhead

        model_mem_estimate = columns_mem_est + baseline_overhead_mem_est
        model_mem_estimate = min(model_mem_estimate, 3.8e10)  # using the tree strategy caps at <40 GB
        mem_estimate = model_mem_estimate + dataset_size_mem_est

        return mem_estimate

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to mirror RealMLP learning rate scheduler while forcing stopping at a specific epoch?
        tags = {"can_refit_full": False}
        return tags
