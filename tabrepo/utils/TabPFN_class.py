from __future__ import annotations

import pandas as pd
from tabrepo.utils.abstract_class import AbstractExecModel
from tabpfn import TabPFNClassifier


class CustomTabPFN(AbstractExecModel):

    def __init__(self, model: TabPFNClassifier):
        super().__init__()
        self.model = model

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        return self.model.fit(X_train, y_train)

    def _predict_proba(self, X_test: pd.DataFrame):
        return self.model.predict_proba(X_test)
