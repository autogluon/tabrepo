from __future__ import annotations

import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.core.metrics import get_metric, Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon_benchmark.utils.time_utils import Timer


class AbstractExecModel:

    # TODO: Prateek: Find a way to put AutoGluon as default - in the case the user does not want their own class
    def __init__(
        self,
        problem_type: str,
        eval_metric: str,
        preprocess_data: bool = True,
        preprocess_label: bool = True,
    ):
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.preprocess_data = preprocess_data
        self.preprocess_label = preprocess_label
        self._label_cleaner = None

    def _clean_label(self, y, y_test):
        self._label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y, y_uncleaned=y)
        y_clean = self._label_cleaner.transform(y)
        y_test_clean = self._label_cleaner.transform(y_test)
        return y_clean, y_test_clean

    def _clean_data(self, X, y, X_test):
        feature_generator = AutoMLPipelineFeatureGenerator()
        X = feature_generator.fit_transform(X=X, y=y)
        X_test = feature_generator.transform(X=X_test)
        return X, X_test

    # TODO: Nick: Temporary name
    def fit_custom2(self, X, y, X_test, y_test):
        if self.preprocess_label:
            y, y_test = self._clean_label(y=y, y_test=y_test)
        if self.preprocess_data:
            X, X_test = self._clean_data(X=X, y=y, X_test=X_test)
        out = self.fit_custom(X, y, X_test)

        y_pred_test_clean = out["predictions"]
        y_pred_proba_test_clean = out["probabilities"]

        scorer: Scorer = get_metric(metric=self.eval_metric, problem_type=self.problem_type)

        out["test_error"] = self.evaluate(
            y_true=y_test,
            y_pred=y_pred_test_clean,
            y_pred_proba=y_pred_proba_test_clean,
            scorer=scorer,
        )

        if self.preprocess_label:
            out["predictions"] = self._label_cleaner.inverse_transform(out["predictions"])
            if out["probabilities"] is not None:
                out["probabilities"] = self._label_cleaner.inverse_transform_proba(out["probabilities"], as_pandas=True)

        return out

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
            self._fit(X, y)

        if self.problem_type in ['binary', 'multiclass']:
            y_pred, y_pred_proba, timer_predict = self.predict_proba_custom(X=X_test)
        else:
            y_pred, timer_predict = self.predict_custom(X=X_test)
            y_pred_proba = None

        out = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'time_fit': timer_fit.duration,
            'time_predict': timer_predict.duration,
        }

        return out

    def _fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict_custom(self, X: pd.DataFrame):
        '''
        Calls the predict function of the inheriting class and proceeds to perform predictions for regression problems
        Returns
        -------
        predictions and inference time, probabilities will be none
        '''
        with Timer() as timer_predict:
            y_pred = self._predict(X)
            y_pred = pd.Series(y_pred, index=X.index)

        return y_pred, timer_predict

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        return y_pred_proba.idxmax(axis=1)

    def _predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def predict_proba_custom(self, X: pd.DataFrame):
        '''
        Calls the predict function of the inheriting class and proceeds to perform predictions for classification
        problems - binary and multiclass

        Returns
        -------
        predictions and inference time, probabilities will be none
        '''
        with Timer() as timer_predict:
            y_pred_proba = self._predict_proba(X)
        y_pred = self.predict_from_proba(y_pred_proba)

        return y_pred, y_pred_proba, timer_predict

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer):
        if scorer.needs_pred:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred)
        elif self.problem_type == "binary":
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba.iloc[:, 1])
        else:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
        return test_error
