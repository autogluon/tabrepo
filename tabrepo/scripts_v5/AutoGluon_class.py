from __future__ import annotations

import shutil

import pandas as pd

from .abstract_class import AbstractExecModel


class CustomAutoGluon(AbstractExecModel):
    def __init__(self, init_kwargs=None, fit_kwargs=None, preprocess_data=False, preprocess_label=False, **kwargs):
        super().__init__(preprocess_data=preprocess_data, preprocess_label=preprocess_label, **kwargs)
        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.label = "__label__"

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self.label] = y

        fit_kwargs = self.fit_kwargs.copy()

        if X_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.label] = y_val
            fit_kwargs["tuning_data"] = tuning_data

        self.predictor = TabularPredictor(label=self.label, problem_type=self.problem_type, eval_metric=self.eval_metric, **self.init_kwargs)
        self.predictor.fit(train_data=train_data, **fit_kwargs)
        # FIXME: persist
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predictor.predict(X)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.predictor.predict_proba(X)
        return y_pred_proba

    def cleanup(self):
        shutil.rmtree(self.predictor.path, ignore_errors=True)
