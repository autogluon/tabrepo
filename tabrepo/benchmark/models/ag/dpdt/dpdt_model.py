from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


def _to_cat(X):
    return X


class BoostedDPDTModel(AbstractModel):
    ag_key = "BOOSTEDDPDT"
    ag_name = "boosted_dpdt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessor = None

    def get_model_cls(self):
        from dpdt import AdaBoostDPDT

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = AdaBoostDPDT
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        if self._preprocessor is None:
            import numpy as np
            from sklearn.compose import ColumnTransformer, make_column_selector
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

            categorical_pipeline = Pipeline(
                [
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant", add_indicator=True, fill_value=-1
                        ),
                    ),
                    (
                        "to_category",
                        FunctionTransformer(_to_cat),
                    ),
                ]
            ).set_output(transform="pandas")

            self._preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        SimpleImputer(strategy="mean", add_indicator=True),
                        make_column_selector(dtype_include=np.number),
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        make_column_selector(dtype_include=["object", "category"]),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas")
            self._preprocessor.fit(X)

        return self._preprocessor.transform(X)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        time_limit: float | None = None,
        **kwargs,
    ):
        model_cls = self.get_model_cls()
        hyp = self._get_model_params()

        self.model = model_cls(
            **hyp,
            n_jobs="best" if num_cpus > 1 else num_cpus,
            time_limit=time_limit,
        )
        X = self.preprocess(X)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _set_default_params(self):
        default_params = {
            "random_state": 42,
            "n_estimators": 1000,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
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

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        if hyperparameters is None:
            hyperparameters = {}

        dataset_size_mem_est = (
            10
            * hyperparameters.get("cart_nodes_list", [2.5])[0]
            * get_approximate_df_mem_usage(X).sum()
        )
        baseline_overhead_mem_est = 3e8  # 300 MB generic overhead

        return int(dataset_size_mem_est + baseline_overhead_mem_est)

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        """DPDT does not yet support refit full."""
        return {"can_refit_full": False}
