from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# pip install pytabkit
class RealMLPModel(AbstractModel):
    ag_key = "REALMLP"
    ag_name = "RealMLP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

    def get_model_cls(self, default_hyperparameters: Literal["td", "td_s"] = "td"):
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor, RealMLP_TD_S_Classifier, RealMLP_TD_S_Regressor

        assert default_hyperparameters in ["td", "td_s"]
        if self.problem_type in ['binary', 'multiclass']:
            if default_hyperparameters == "td":
                model_cls = RealMLP_TD_Classifier
            else:
                model_cls = RealMLP_TD_S_Classifier
        else:
            if default_hyperparameters == "td":
                model_cls = RealMLP_TD_Regressor
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
        import torch
        device = "cpu" if num_gpus == 0 else "cuda"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()

        default_hyperparameters = hyp.pop("default_hyperparameters")

        model_cls = self.get_model_cls(default_hyperparameters=default_hyperparameters)

        metric_map = {
            "roc_auc": "1-auc_ovr_alt",
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

        bool_to_cat = hyp.pop("bool_to_cat", True)
        impute_bool = hyp.pop("impute_bool", True)
        name_categories = hyp.pop("name_categories", True)

        self.model = model_cls(
            n_threads=num_cpus,
            device="cuda:0",  # FIXME: code assume we only see one GPU in the fit process.
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

            # Don't use early stopping by default, seems to work well without
            use_early_stopping=False,
            early_stopping_additive_patience=40,
            early_stopping_multiplicative_patience=3,

            # verdict: use_ls="auto" is much better than None.
            use_ls="auto",

            # verdict: no impact, but makes more sense to be False.
            impute_bool=False,

            # verdict: name_categories=True avoids random exceptions being raised in rare cases
            name_categories=True,

            # verdict: bool_to_cat=True is equivalent to False in terms of quality, but can be slightly faster in training time
            #  and slightly slower in inference time
            bool_to_cat=True,

            # verdict: use_roc_auc_to_stop=True is best
            # use_roc_auc_to_stop=True,  # Tested and integrated directly, no longer a valid hyperparameter

            # verdict: "td" is better than "td_s"
            default_hyperparameters="td",  # options ["td", "td_s"]
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        import torch
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 1 if torch.cuda.is_available() else 0
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
        hyperparameters: dict = None,
        **kwargs,
    ) -> int:
        """
        Heuristic memory estimate that correlates strongly with RealMLP's more sophisticated method

        More comprehensive memory estimate logic:

        ```python
        from typing import Any

        from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
        from pytabkit.models.data.data import DictDataset, TensorInfo
        from pytabkit.models.sklearn.default_params import DefaultParams

        def estimate_realmlp_cpu_ram_gb(hparams: dict[str, Any], n_numerical: int, cat_sizes: list[int], n_classes: int,
                                        n_samples: int):
            params = copy.copy(DefaultParams.RealMLP_TD_CLASS if n_classes > 0 else DefaultParams.RealMLP_TD_REG)
            params.update(hparams)

            ds = DictDataset(tensors=None, tensor_infos=dict(x_cont=TensorInfo(feat_shape=[n_numerical]),
                                                             x_cat=TensorInfo(cat_sizes=cat_sizes),
                                                             y=TensorInfo(cat_sizes=[n_classes])), device='cpu',
                             n_samples=n_samples)

            alg_interface = NNAlgInterface(**params)
            res = alg_interface.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=1, split_seeds=[0], n_train=n_samples)
            return res.cpu_ram_gb
        ```

        """
        if hyperparameters is None:
            hyperparameters = {}
        plr_hidden_1 = hyperparameters.get("plr_hidden_1", 16)
        plr_hidden_2 = hyperparameters.get("plr_hidden_2", 4)
        hidden_width = hyperparameters.get("hidden_width", 256)

        num_features = len(X.columns)
        columns_mem_est = num_features * 8e5

        hidden_1_weight = 0.13
        hidden_2_weight = 0.42
        width_factor = math.sqrt(hidden_width / 256 + 0.6)

        columns_mem_est_hidden_1 = columns_mem_est * hidden_1_weight * plr_hidden_1 / 16 * width_factor
        columns_mem_est_hidden_2 = columns_mem_est * hidden_2_weight * plr_hidden_2 / 16 * width_factor
        columns_mem_est = columns_mem_est_hidden_1 + columns_mem_est_hidden_2

        dataset_size_mem_est = 5 * get_approximate_df_mem_usage(X).sum()  # roughly 5x DataFrame memory size
        baseline_overhead_mem_est = 3e8  # 300 MB generic overhead

        mem_estimate = dataset_size_mem_est + columns_mem_est + baseline_overhead_mem_est

        return mem_estimate

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to mirror RealMLP learning rate scheduler while forcing stopping at a specific epoch?
        tags = {"can_refit_full": False}
        return tags
