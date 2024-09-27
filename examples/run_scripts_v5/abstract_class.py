from __future__ import annotations

import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
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
        self.label_cleaner: LabelCleaner = None
        self._feature_generator = None

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
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y, y_uncleaned=y)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = AutoMLPipelineFeatureGenerator()
            X = self._feature_generator.fit_transform(X=X, y=y)
        y = self.transform_y(y)
        return X, y

    # TODO: Nick: Temporary name
    def fit_custom2(self, X, y, X_test, y_test):
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
            self.fit(X, y)

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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X, y = self._preprocess_fit_transform(X=X, y=y)
        return self._fit(X=X, y=y)

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
            y_pred = self.predict(X)

        return y_pred, timer_predict

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        return y_pred_proba.idxmax(axis=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self.transform_X(X=X)
        y_pred = self._predict(X)
        return self.inverse_transform_y(y=y_pred)

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
            y_pred_proba = self.predict_proba(X)
        y_pred = self.predict_from_proba(y_pred_proba)

        return y_pred, y_pred_proba, timer_predict

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.transform_X(X=X)
        y_pred_proba = self._predict_proba(X=X)
        return self.inverse_transform_y_pred_proba(y_pred_proba=y_pred_proba)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer):
        y_true = self.transform_y(y_true)
        if scorer.needs_pred:
            y_pred = self.transform_y(y_pred)
            test_error = scorer.error(y_true=y_true, y_pred=y_pred)
        elif self.problem_type == "binary":
            y_pred_proba = self.transform_y_pred_proba(y_pred_proba)
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba.iloc[:, 1])
        else:
            y_pred_proba = self.transform_y_pred_proba(y_pred_proba)
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
        return test_error
