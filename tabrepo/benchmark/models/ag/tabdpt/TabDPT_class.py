from __future__ import annotations

import numpy as np
import pandas as pd

from tabrepo.benchmark.models.wrapper.abstract_class import AbstractExecModel


class CustomTabDPT(AbstractExecModel):
    def __init__(self, n_ensembles=1, max_epochs=0, context_size=2048, device="cpu", path_weights_classification=None, path_weights_regression=None, **kwargs):
        super().__init__(**kwargs)
        self.n_ensembles = n_ensembles
        self.max_epochs = max_epochs
        self.device = device
        self.context_size = context_size
        if path_weights_classification is None:
            path_weights_classification = '/home/ubuntu/workspace/tabdpt_weights/tabdpt_76M.ckpt'
        if path_weights_regression is None:
            path_weights_regression = '/home/ubuntu/workspace/tabdpt_weights/tabdpt_76M.ckpt'
        self.path_weights_classification = path_weights_classification
        self.path_weights_regression = path_weights_regression

    def get_model_cls(self):
        from tabrepo.benchmark.models.ag.tabdpt.internal.tabdpt import TabDPTClassifier, TabDPTRegressor
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabDPTClassifier
        else:
            model_cls = TabDPTRegressor
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        model_cls = self.get_model_cls()

        if self.problem_type == "regression":
            path_weights = self.path_weights_regression
            n_classes = 0
        else:
            path_weights = self.path_weights_classification
            n_classes = len(y.unique())

        X = X.values.astype(np.float64)
        y = y.values
        if X_val is not None:
            X_val = X_val.values.astype(np.float64)
            y_val = y_val.values

        self.model = model_cls(
            device=self.device,
            path=path_weights,
        )
        self.model.fit(
            X=X,
            y=y,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X.values.astype(np.float64), context_size=self.context_size)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X.values.astype(np.float64), context_size=self.context_size)
        return y_pred_proba
