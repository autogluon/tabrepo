from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


class BoostedDPDTModel(AbstractModel):
    ag_key = "BOOSTEDDPDT"
    ag_name = "boosted_dpdt"

    def get_model_cls(self):
        from dpdt import AdaBoostDPDT

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = AdaBoostDPDT
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, **kwargs):
        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        if num_cpus < 1:
            num_cpus = 'best'
        self.model = model_cls(
            **hyp,
            n_jobs=num_cpus,
        )
        X = self.preprocess(X)
        self.model = self.model.fit(
            X=X,
            y=y,
        )
    

    def _set_default_params(self):
        default_params = {
            "random_state": 42,
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
        num_gpus = 0
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
        if hyperparameters is None:
            hyperparameters = {}
        
        dataset_size_mem_est = 10 * hyperparameters.get('cart_nodes_list')[0] * get_approximate_df_mem_usage(X).sum()
        baseline_overhead_mem_est = 3e8  # 300 MB generic overhead

        mem_estimate = dataset_size_mem_est + baseline_overhead_mem_est

        return mem_estimate

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        """DPDT does not yet support refit full."""
        return {"can_refit_full": False}