from __future__ import annotations

import pandas as pd

from ..abstract_class import AbstractExecModel
from autogluon.core.models import AbstractModel


class CustomAGModel(AbstractExecModel):
    def __init__(self, model_cls, hyperparameters: dict = None, **kwargs):
        assert issubclass(model_cls, AbstractModel)
        self.model_cls = model_cls
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters
        super().__init__(**kwargs)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.model = self.model_cls(
            path="", name=self.model_cls.__name__, problem_type=self.problem_type, eval_metric=self.eval_metric, hyperparameters=self.hyperparameters,
        )
        self.model.fit(
            X=X,
            y=y,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        y_pred_proba = pd.DataFrame(y_pred_proba, index=X.index)
        return y_pred_proba
