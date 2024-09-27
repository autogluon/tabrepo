from __future__ import annotations

import pandas as pd
from abstract_class import AbstractExecModel
from tabpfn import TabPFNClassifier


class CustomTabPFN(AbstractExecModel):
    def get_model_cls(self):
        is_classification = self.problem_type in ['binary', 'multiclass']
        if is_classification:
            model_cls = TabPFNClassifier
        else:
            raise AssertionError(f"TabPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        model_cls = self.get_model_cls()
        self.model = model_cls(device='cpu', N_ensemble_configurations=32)
        self.model.fit(
            X=X,
            y=y,
            overwrite_warning=True,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)
        return y_pred_proba
