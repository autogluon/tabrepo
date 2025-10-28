from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import get_metric, Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.utils.time_utils import Timer


class AbstractExecModel:
    can_get_error_val = False
    can_get_oof = False
    can_get_per_child_oof = False
    can_get_per_child_test = False
    can_get_per_child_val_idx = False

    # TODO: Prateek: Find a way to put AutoGluon as default - in the case the user does not want their own class
    def __init__(
        self,
        problem_type: str,
        eval_metric: Scorer,
        preprocess_data: bool = True,
        preprocess_label: bool = True,
    ):
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.preprocess_data = preprocess_data
        self.preprocess_label = preprocess_label
        self.label_cleaner: LabelCleaner = None
        self._feature_generator = None
        self.failure_artifact = None

    def transform_y(self, y: pd.Series) -> pd.Series:
        return self.label_cleaner.transform(y)

    def inverse_transform_y(self, y: pd.Series) -> pd.Series:
        return self.label_cleaner.inverse_transform(y)

    def transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        return self.label_cleaner.transform_proba(y_pred_proba, as_pandas=True)

    def inverse_transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        return self.label_cleaner.inverse_transform_proba(y_pred_proba, as_pandas=True)

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_data:
            return self._feature_generator.transform(X)
        return X

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series):
        if self.preprocess_label:
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = AutoMLPipelineFeatureGenerator()
            X = self._feature_generator.fit_transform(X=X, y=y)
        y = self.transform_y(y)
        return X, y

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        pass

    # TODO: Prateek, Add a toggle here to see if user wants to fit or fit and predict, also add model saving functionality
    # TODO: Nick: Temporary name
    def fit_custom(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        '''
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Returns
        -------
        dict
        Returns predictions, probabilities, fit time and inference time
        '''
        with (Timer() as timer_fit):
            self.fit(X, y)

        self.post_fit(X=X, y=y, X_test=X_test)

        if self.problem_type in ['binary', 'multiclass']:
            with Timer() as timer_predict:
                y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_from_proba(y_pred_proba)
        else:
            with Timer() as timer_predict:
                y_pred = self.predict(X_test)
            y_pred_proba = None

        out = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'time_train_s': timer_fit.duration,
            'time_infer_s': timer_predict.duration,
        }

        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        X, y = self._preprocess_fit_transform(X=X, y=y)
        if X_val is not None:
            X_val = self.transform_X(X_val)
            y_val = self.transform_y(y_val)
        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        if isinstance(y_pred_proba, pd.DataFrame):
            return y_pred_proba.idxmax(axis=1)
        else:
            return np.argmax(y_pred_proba, axis=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self.transform_X(X=X)
        y_pred = self._predict(X)
        return self.inverse_transform_y(y=y_pred)

    def _predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.transform_X(X=X)
        y_pred_proba = self._predict_proba(X=X)
        return self.inverse_transform_y_pred_proba(y_pred_proba=y_pred_proba)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def cleanup(self):
        pass

    def get_metric_error_val(self) -> float:
        raise NotImplementedError
