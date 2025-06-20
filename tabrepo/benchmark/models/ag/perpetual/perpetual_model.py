from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from autogluon.core.metrics import Scorer


class PerpetualBoostingModel(AbstractModel):
    ag_key = "PB"
    ag_name = "PerpetualBoosting"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features: list[str] = None

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)

        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include="category").columns)

        return X

    # TODO: support sample weights
    # TODO: API does not support a random seed, but rust code does (mhm?)
    # TODO: no GPU support (?), add warning.
    # TODO: no support for passing validation data... problematic
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        # X_val: pd.DataFrame | None = None,
        # y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int | str = "auto",
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        **kwargs,
    ):
        # Preprocess data.
        X = self.preprocess(X, is_train=True)
        paras = self._get_model_params()

        from perpetual import PerpetualBooster

        # FIXME: get correct number within ray job,
        #  here we know it is one of 8 folds now.
        #   -> This is also not correct if not bagged mode is used, or outer
        #   memory limit is not set on SLURM.
        # memory_limit = ResourceManager().get_available_virtual_mem(format="GB") / 8
        # the above does not work as env var is not set in ray job...
        # need to pass this to the model kwargs as well
        # future work... hard coding for now.
        memory_limit = 4

        # safety as memory limit is not strictly enforced by PB
        memory_limit = int(memory_limit * 0.95)

        self.model = PerpetualBooster(
            objective=get_metric_from_ag_metric(
                metric=self.eval_metric, problem_type=self.problem_type
            ),
            num_threads=num_cpus,
            memory_limit=memory_limit,
            categorical_features=self._category_features,
            timeout=time_limit,
            # stopping_rounds - no idea how to set this
            **paras,
        )

        self.model.fit(X=X, y=y, sample_weight=sample_weight)

    def _set_default_params(self):
        default_params = {
            "iteration_limit": 10_000,
            # As we use a timeout this should be the max I guess.
            #   2.0 is used a lot in examples, so I guess it is the max (?).
            "budget": 2.0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = {
            "valid_raw_types": ["int", "float", "category"],
        }
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    # TODO: support refit full
    def _more_tags(self) -> dict:
        return {"can_refit_full": False}


def get_metric_from_ag_metric(*, metric: Scorer, problem_type: str):
    """Map AutoGluon metric to EBM metric for early stopping."""
    if problem_type in [BINARY, MULTICLASS]:
        # Only supports log_loss for classification.
        metric_map = {
            "log_loss": "LogLoss",
        }
        metric_class = metric_map.get(metric.name, "LogLoss")
    elif problem_type == REGRESSION:
        metric_map = {
            "mean_squared_error": "HuberLoss",  # seems to be a better match than RMSE.
            "root_mean_squared_error": "SquaredLoss",
        }
        metric_class = metric_map.get(metric.name, "SquaredLoss")
    else:
        raise AssertionError(f"EBM does not support {problem_type} problem type.")

    return metric_class
