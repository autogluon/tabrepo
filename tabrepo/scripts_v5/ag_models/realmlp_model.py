from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel


# TODO: This doesn't actually work for all tasks, need to improve
# pip install pytabkit
class RealMLPModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None

    def get_model_cls(self):
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_S_Regressor

        if self.problem_type in ['binary', 'multiclass']:
            model_cls = RealMLP_TD_Classifier
        else:
            model_cls = RealMLP_TD_S_Regressor
        return model_cls

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        model_cls = self.get_model_cls()

        metric_map = {
            "roc_auc": "cross_entropy",
            "accuracy": "class_error",
            "balanced_accuracy": "1-balanced_accuracy",
            "log_loss": "cross_entropy",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "rmse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        val_metric_name = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = dict()

        if val_metric_name is not None:
            init_kwargs["val_metric_name"] = val_metric_name

        hyp = self._get_model_params()
        if X_val is None:
            hyp["use_early_stopping"] = False
            hyp["val_fraction"] = 0

        # TODO: use_ls toggle!
        self.model = model_cls(
            n_threads=num_cpus,
            device="cpu",
            **init_kwargs,
            **hyp,
        )

        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        # TODO: Categorical indicator?
        print(f'start {self.name}')
        self.model = self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_to_fit_in_seconds=time_limit,
        )

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """
        Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"])
            self._features_to_keep = self._feature_metadata.get_features(invalid_raw_types=["int", "float"])
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [c for c in self._imputer.get_feature_names_out() if c not in self._features_to_impute]
        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            # dtype_dict = {indicator_col: "category" for indicator_col in self._indicator_columns}
            X_impute = pd.DataFrame(X_impute, index=X.index, columns=self._imputer.get_feature_names_out())
            if self._indicator_columns:
                X_impute[self._indicator_columns] = X_impute[self._indicator_columns].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        return X

    def _set_default_params(self):
        default_params = dict(
            random_state=0,
            use_early_stopping=True,
            early_stopping_additive_patience=40,
            early_stopping_multiplicative_patience=3,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "problem_types": [BINARY, MULTICLASS, REGRESSION],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        tags = {"can_refit_full": False}
        return tags
