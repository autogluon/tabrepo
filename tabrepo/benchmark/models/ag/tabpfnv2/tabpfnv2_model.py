import os
from contextlib import contextmanager

import pandas as pd

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel


@contextmanager
def override_env_var(key, value):
    original_value = os.getenv(key)
    os.environ[key] = value
    yield
    if original_value is not None:
        os.environ[key] = original_value
    else:
        del os.environ[key]


# FIXME: why so slow?
class TabPFNV2Model(AbstractModel):
    ag_key = "TABPFNV2"
    ag_name = "TabPFNv2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def get_model_cls(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabPFNClassifier
        else:
            model_cls = TabPFNRegressor
        return model_cls

    # FIXME: What is the minimal model artifact?
    #  If zeroshot, maybe we don't save weights for each fold in bag and instead load from a single weights file?
    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, num_cpus=1, **kwargs):
        model_cls = self.get_model_cls()

        metric_map = {
            "roc_auc": "roc",
            "accuracy": "acc",
            "balanced_accuracy": "balanced_acc",
            "log_loss": "log_loss",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "r2",
        }

        eval_metric_tabpfn = metric_map[self.eval_metric.name]

        hyp = self._get_model_params()
        self.model = model_cls(
            # optimize_metric=eval_metric_tabpfn,  # FIXME: How to specify? This existed in the client version
            device="cpu",
            n_jobs=num_cpus,
            **hyp,
        )

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        if self.problem_type in [BINARY, MULTICLASS]:
            if max_classes is not None and self.num_classes > max_classes:
                # TODO: Move to earlier stage when problem_type is checked
                raise AssertionError(f"Max allowed classes for the model is {max_classes}, " f"but found {self.num_classes} classes.")

        X = self.preprocess(X)
        # if X_val is not None:
        #     X_val = self.preprocess(X_val)

        with override_env_var("TABPFN_ALLOW_CPU_LARGE_DATASET", "1"):
            self.model = self.model.fit(
                X=X,
                y=y,
            )

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Converts categorical to label encoded integers
        Keeps missing values, as TabPFN automatically handles missing values internally.
        """
        X = super()._preprocess(X, **kwargs)
        # if self._feature_generator is None:
        #     self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
        #     self._feature_generator.fit(X=X)
        # if self._feature_generator.features_in:
        #     X = X.copy()
        #     X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        # FIXME: Will raise an exception if the model isn't downloaded
        extra_ag_args_ensemble = {
           "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def _ag_params(self) -> set:
        return {"max_classes"}

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
