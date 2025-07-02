from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


# TODO: Verify if crashes when weights are not yet downloaded and fit in parallel
# TODO: Needs memory usage estimate method
class TabICLModel(AbstractModel):
    ag_key = "TABICL"
    ag_name = "TabICL"
    ag_priority = 65

    def get_model_cls(self):
        from tabicl import TabICLClassifier

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = TabICLClassifier
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    @staticmethod
    def _get_batch_size(n_cells: int):
        if n_cells <= 4_000_000:
            return 8
        elif n_cells <= 6_000_000:
            return 4
        else:
            return 2

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        hyp["batch_size"] = hyp.get("batch_size", self._get_batch_size(X.shape[0] * X.shape[1]))
        self.model = model_cls(
            **hyp,
            device=device,
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

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100000,
                "max_features": 500,
            }
        )
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = min(ResourceManager.get_gpu_count_torch(), 1)
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
