from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


# TODO: Needs memory usage estimate method
class TabICLModel(AbstractModel):
    ag_key = "TABICL"
    ag_name = "TabICL"

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
        else:
            return 4

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

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        from autogluon.common.utils.resource_utils import ResourceManager
        from torch.cuda import is_available

        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = 1 if is_available() else 0
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
