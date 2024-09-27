from __future__ import annotations

import pandas as pd
from abstract_class import AbstractExecModel
from lightgbm import LGBMClassifier, LGBMRegressor


class CustomLGBM(AbstractExecModel):
    def get_model_cls(self):
        is_classification = self.problem_type in ['binary', 'multiclass']
        if is_classification:
            model_cls = LGBMClassifier
        elif self.problem_type == 'regression':
            model_cls = LGBMRegressor
        else:
            raise AssertionError(f"LightGBM does not recognize the problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        model_cls = self.get_model_cls()
        self.model = model_cls()
        self.model.fit(
            X=X,
            y=y
        )
        return self

    def _predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)
