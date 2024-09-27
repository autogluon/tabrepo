from __future__ import annotations

import pandas as pd
from abstract_class import AbstractExecModel


class CustomTabPFNv2(AbstractExecModel):
    def get_model_cls(self):
        from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabPFNClassifier
        elif self.problem_type == "regression":
            model_cls = TabPFNRegressor
        else:
            raise AssertionError(f"TabPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _get_optimize_metric_tabpfn(self):
        metric_map = {
            "roc_auc": "roc",
            "accuracy": "acc",
            "balanced_accuracy": "balanced_acc",
            "log_loss": "log_loss",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "r2",
        }
        optimize_metric_tabpfn = metric_map[self.eval_metric]
        return optimize_metric_tabpfn

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        model_cls = self.get_model_cls()
        optimize_metric = self._get_optimize_metric_tabpfn()
        self.model = model_cls(
            model="latest_tabpfn_hosted",
            optimize_metric=optimize_metric,
            n_estimators=32,
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
        y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)
        return y_pred_proba
