from __future__ import annotations

import logging
import pandas as pd
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# pip install pytabkit
class RealMLPModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

    def get_model_cls(self, td_s_reg=True):
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_S_Regressor, RealMLP_TD_Regressor

        if self.problem_type in ['binary', 'multiclass']:
            model_cls = RealMLP_TD_Classifier
        else:
            if td_s_reg:
                model_cls = RealMLP_TD_S_Regressor  # FIXME FIXME FIXME
            else:
                model_cls = RealMLP_TD_Regressor
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
        if num_gpus > 0:
            logger.log(30, f"WARNING: GPUs are not yet implemented for RealMLP model, but `num_gpus={num_gpus}` was specified... Ignoring GPU.")

        hyp = self._get_model_params()

        # TODO: Remove eventually?
        td_s_reg = hyp.pop("td_s_reg", True)

        model_cls = self.get_model_cls(td_s_reg=td_s_reg)

        # TODO: Try "roc_auc": "1-auc_ovr_alt"
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

        # FIXME: Temp to test the impact
        use_roc_auc_to_stop = hyp.pop("use_roc_auc_to_stop", False)
        if use_roc_auc_to_stop and self.eval_metric.name == "roc_auc":
            val_metric_name = "1-auc_ovr_alt"
        else:
            val_metric_name = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = dict()

        if val_metric_name is not None:
            init_kwargs["val_metric_name"] = val_metric_name

        # TODO: Make this smarter? Maybe use `eval_metric.needs_pred`
        if hyp["use_ls"] is not None and isinstance(hyp["use_ls"], str) and hyp["use_ls"] == "auto":
            if val_metric_name is None:
                hyp["use_ls"] = False
            elif val_metric_name in ["cross_entropy", "1-auc_ovr_alt"]:
                hyp["use_ls"] = False
            else:
                hyp["use_ls"] = None

        if X_val is None:
            hyp["use_early_stopping"] = False
            hyp["val_fraction"] = 0

        bool_to_cat = hyp.pop("bool_to_cat", False)
        impute_bool = hyp.pop("impute_bool", True)
        name_categories = hyp.pop("name_categories", False)

        # TODO: GPU
        self.model = model_cls(
            n_threads=num_cpus,
            device="cpu",
            **init_kwargs,
            **hyp,
        )

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat, impute_bool=impute_bool)

        # FIXME: In rare cases can cause exceptions if name_categories=False, unknown why
        extra_fit_kwargs = {}
        if name_categories:
            cat_col_names = X.select_dtypes(include='category').columns.tolist()
            extra_fit_kwargs["cat_col_names"] = cat_col_names

        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_to_fit_in_seconds=time_limit,
            **extra_fit_kwargs,
        )

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, bool_to_cat: bool = False, impute_bool: bool = True, **kwargs) -> pd.DataFrame:
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

    def _set_default_params(self):
        default_params = dict(
            random_state=0,
            use_early_stopping=True,
            early_stopping_additive_patience=40,
            early_stopping_multiplicative_patience=3,

            # verdict: use_ls="auto" is much better than None.
            use_ls="auto",

            # verdict: use_roc_auc_to_stop=True is best
            use_roc_auc_to_stop=False,  # TODO: Remove after testing

            # verdict: no impact, but makes more sense to be False.
            impute_bool=True,  # FIXME: Remove after testing

            # verdict: name_categories=True avoids random exceptions being raised in rare cases
            name_categories=False,  # FIXME: Remove after testing

            # verdict: bool_to_cat=True is equivalent to False in terms of quality, but can be slightly faster in training time
            #  and slightly slower in inference time
            bool_to_cat=False,  # FIXME: Remove after testing

            # FIXME: During early testing, accidentally was using TD_S for regression instead of TD.
            #  If False, this uses TD instead of TD_S for regression.
            td_s_reg=True,  # FIXME: Remove after testing
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

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    # FIXME: Find a better estimate for memory usage of RealMLP. Currently borrowed from FASTAI estimate.
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
        #  How to mirror RealMLP learning rate scheduler while forcing stopping at a specific epoch?
        tags = {"can_refit_full": False}
        return tags
