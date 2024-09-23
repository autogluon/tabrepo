from __future__ import annotations

import pandas as pd
from typing import Callable, List
from autogluon.core.data import LabelCleaner
from autogluon.core.metrics import get_metric, Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon_benchmark.frameworks.autogluon.run import ag_eval_metric_map
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper
from autogluon_benchmark.utils.time_utils import Timer
from tabrepo.utils.cache import DummyExperiment, Experiment


class AbstractExecModel:

    # TODO: Prateek: Find a way to put AutoGluon as default - in the case the user dones not want their own class
    def __init__(self):
        self.label = None
        self.problem_type = None

    # TODO: Prateek: Give a toggle for just fitting and saving the model, if not call predict as well
    # above to-do is mentioned again in fit_custom()
    def run_experiments(
            self,
            expname: str,
            tids: List[int],
            folds: List[int],
            methods: List[str],
            methods_dict: dict,
            task_metadata: pd.DataFrame,
            ignore_cache: bool,
            exec_func_kwargs: dict = None,
            cache_class: Callable | None = Experiment,
            cache_class_kwargs: dict = None,
            clean_features: bool = True,
    ) -> list:
        '''

        Parameters
        ----------
        expname: str, Name of the experiment given by the user
        tids: list[int], List of OpenML task IDs given by the user
        folds: list[int], Number of folds present for the given task
        methods: list[str], Models used for fit() and predict() in this experiment
        methods_dict: dict, methods (models) mapped to their respective fit_args()
        task_metadata: pd.DataFrame,OpenML task metadata
        ignore_cache: bool, whether to use cached results (if present)
        exec_func_kwargs: WIP
        cache_class: WIP
        cache_class_kwargs: WIP
        clean_features: bool, whether to clean data and labels, left to the user's discretion

        Returns
        -------
        result_lst: list, containing all metrics from fit() and predict() of all the given OpenML tasks
        '''
        # TODO: Prateek, Check usage
        if exec_func_kwargs is None:
            exec_func_kwargs = {}
        if cache_class is None:
            cache_class = DummyExperiment
        if cache_class_kwargs is None:
            cache_class_kwargs = {}
        dataset_names = [task_metadata[task_metadata["tid"] == tid]["name"].iloc[0] for tid in tids]
        print(
            f"Running Experiments for expname: '{expname}'..."
            f"\n\tFitting {len(tids)} datasets and {len(folds)} folds for a total of {len(tids) * len(folds)} tasks"
            f"\n\tFitting {len(methods)} methods on {len(tids) * len(folds)} tasks for a total of {len(tids) * len(folds) * len(methods)} jobs..."
            f"\n\tTIDs    : {tids}"
            f"\n\tDatasets: {dataset_names}"
            f"\n\tFolds   : {folds}"
            f"\n\tMethods : {methods}"
        )
        result_lst = []
        for tid in tids:
            task = OpenMLTaskWrapper.from_task_id(task_id=tid)
            task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
            self.label = task.label
            self.problem_type = task.problem_type
            for fold in folds:
                for method in methods:
                    cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                    # TODO: Prateek, yet to support fit_args
                    fit_args = methods_dict[method]
                    print(
                        f"\n\tFitting {task_name} on {fold} of {len(folds)-1} folds for method {method}"
                    )
                    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold)
                    if clean_features:
                        X_train, y_train, X_test, y_test = self.clean_pre_fit(problem_type=self.problem_type,
                                                                              X_train=X_train,
                                                                              y_train=y_train, X_test=X_test,
                                                                              y_test=y_test)
                    experiment = cache_class(
                        expname=expname,
                        name=cache_name,
                        run_fun=lambda: self.fit_custom(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test
                        ),
                        **cache_class_kwargs
                    )
                    # FIXME: The output df still needs evaluation and formatting, currently just has predictions
                    # probabilities, fit and infer times
                    out = experiment.data(ignore_cache=ignore_cache)
                    result_lst.append(out)

        return result_lst

    # TODO: Consult with Nick if we have separate toggles (functions) for label and data cleaning
    def clean_pre_fit(self, problem_type: str, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        '''
        This function uses AutoGluon's Label cleaner and Pipeline feature generator to clean both labels and features
        The user can skip using this function if they want to use their own label and feature pre-processor

        Parameters
        ----------
        problem_type : Whether the OpenML task problem type - classification or regression
        X_train : Training data, without any cleaning or pre-preprocessing
        y_train : Training labels, without any cleaning or pre-preprocessing
        X_test : Test data, without any cleaning or pre-preprocessing
        y_test : Test labels, without any cleaning or pre-preprocessing

        Returns
        -------
        pd.Dataframe, pd.Series
        Returns cleaned labels and features (both train and test)
        '''

        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train, y_uncleaned=y_train)
        y_train_clean = label_cleaner.transform(y_train)
        y_test_clean = label_cleaner.transform(y_test)

        feature_generator = AutoMLPipelineFeatureGenerator()
        X_train_clean = feature_generator.fit_transform(X=X_train, y=y_train)
        X_test_clean = feature_generator.transform(X=X_test)

        return X_train_clean, y_train_clean, X_test_clean, y_test_clean

    # TODO: Prateek, Add a toggle here to see if user wants to fit or fit and predict, also add model saving functionality
    def fit_custom(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
        '''
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Returns
        -------
        dict
        Returns predictions, probabilities, fit time and inference time
        '''
        with (Timer() as timer_fit):
            self.model = self._fit(X_train, y_train)

        if self.problem_type in ['binary', 'multiclass']:
            y_pred, y_pred_proba, timer_predict = self.predict_proba_custom(X_test=X_test)
        else:
            y_pred, y_pred_proba, timer_predict = self.predict_custom(X_test=X_test)

        out = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'time_fit': timer_fit.duration,
            'time_predict': timer_predict.duration,
        }

        df_results = pd.DataFrame([out])
        return df_results

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError

    def predict_custom(self, X_test: pd.DataFrame):
        '''
        Calls the predict function of the inheriting class and proceeds to perform predictions for regression problems
        Returns
        -------
        predictions and inference time, probabilities will be none
        '''
        with Timer() as timer_predict:
            y_pred = self._predict(X_test)
            y_pred = pd.Series(y_pred, name=self.label, index=X_test.index)
        y_pred_proba = None

        return y_pred, y_pred_proba, timer_predict

    def _predict(self, X_test: pd.DataFrame):
        raise NotImplementedError

    def predict_proba_custom(self, X_test: pd.DataFrame):
        '''
        Calls the predict function of the inheriting class and proceeds to perform predictions for classification
        problems - binary and multiclass

        Returns
        -------
        predictions and inference time, probabilities will be none
        '''
        with Timer() as timer_predict:
            y_pred_proba = self._predict_proba(X_test)
            y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X_test.index)
        y_pred = y_pred_proba.idxmax(axis=1)

        return y_pred, y_pred_proba, timer_predict

    def _predict_proba(self, X_test: pd.DataFrame):
        raise NotImplementedError

    def convert_leaderboard_to_configs(self, leaderboard: pd.DataFrame, minimal: bool = True) -> pd.DataFrame:

        df_configs = leaderboard.rename(columns=dict(
            time_fit="time_train_s",
            time_predict="time_infer_s",
            test_error="metric_error",
            eval_metric="metric",
            val_error="metric_error_val",
        ))
        if minimal:
            df_configs = df_configs[[
                "dataset",
                "fold",
                "framework",
                "metric_error",
                "metric",
                "problem_type",
                "time_train_s",
                "time_infer_s",
                "tid",
            ]]
        return df_configs

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame):

        eval_metric = ag_eval_metric_map[self.problem_type]
        scorer: Scorer = get_metric(metric=eval_metric, problem_type=self.problem_type)
        if scorer.needs_pred:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred)
        elif self.problem_type == "binary":
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba.iloc[:, 1], labels=y_pred_proba.columns)
        else:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba, labels=y_pred_proba.columns)
        return test_error
