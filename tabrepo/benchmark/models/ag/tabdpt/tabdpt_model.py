from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class TabDPTModel(AbstractModel):
    ag_key = "TABDPT"
    ag_name = "TabDPT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None

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
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        model_cls = TabDPTClassifier if self.problem_type in [BINARY, MULTICLASS] else TabDPTRegressor

        hps = self._get_model_params()
        self._predict_hps = dict(seed=42, context_size=1024)
        X = self.preprocess(X)
        y = y.to_numpy()
        self.model = model_cls(
            device=device,
            use_flash=False, # Does not work for our GPUs, and it is unclear when/which GPUs it works for.
            **hps)
        self.model.fit(X=X, y=y)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X, **self._predict_hps)

        y_pred_proba = self.model.predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabDPT requires numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.to_numpy()

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
