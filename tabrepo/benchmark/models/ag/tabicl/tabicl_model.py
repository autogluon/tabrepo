import pandas as pd

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel


# TODO: Needs memory usage estimate method
class TabICLModel(AbstractModel):
    ag_key = "TABICL"
    ag_name = "TabICL"

    def get_model_cls(self):
        from tabicl import TabICLClassifier
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabICLClassifier
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        num_cpus: int = 1,
        **kwargs,
    ):
        model_cls = self.get_model_cls()

        hyp = self._get_model_params()
        self.model = model_cls(
            **hyp,
        )

        X = self.preprocess(X)
        # if X_val is not None:
        #     X_val = self.preprocess(X_val)

        self.model = self.model.fit(
            X=X,
            y=y,
            # X_val=X_val,
            # y_val=y_val,
        )

    def _set_default_params(self):
        default_params = {
            "device": "cpu",
            "n_estimators": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        # FIXME: Test if it works with parallel, need to enable n_cpus support
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

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
