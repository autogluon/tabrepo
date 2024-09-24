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
    def __init__(
        self,
        label: str,
        problem_type: str,
        eval_metric: str,
        preprocess_data: bool = True,
        preprocess_label: bool = True,
    ):
        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.preprocess_data = preprocess_data
        self.preprocess_label = preprocess_label
        self._label_cleaner = None

    # TODO: Prateek: Give a toggle for just fitting and saving the model, if not call predict as well
    # TODO: Nick: This should not be part of this class.
    # above to-do is mentioned again in fit_custom()
    @staticmethod
    def run_experiments(
        expname: str,
        tids: List[int],
        folds: List[int],
        methods: List[str],
        methods_dict: dict,
        method_cls,  # FIXME: Nick: This needs to be communicated on a per-method basis
        task_metadata: pd.DataFrame,
        ignore_cache: bool,
        cache_class: Callable | None = Experiment,
        cache_class_kwargs: dict = None,
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
        method_cls: WIP
        cache_class: WIP
        cache_class_kwargs: WIP

        Returns
        -------
        result_lst: list, containing all metrics from fit() and predict() of all the given OpenML tasks
        '''
        # TODO: Prateek, Check usage
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
            for fold in folds:
                for method in methods:
                    cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                    # TODO: Prateek, yet to support fit_args
                    fit_args = methods_dict[method]
                    print(
                        f"\n\tFitting {task_name} on fold {fold} for method {method}"
                    )
                    experiment_obj = method_cls(
                        label=task.label,
                        problem_type=task.problem_type,
                        eval_metric=ag_eval_metric_map[task.problem_type],
                        **fit_args,
                    )
                    experiment = cache_class(
                        expname=expname,
                        name=cache_name,
                        run_fun=lambda: experiment_obj.run_experiment(
                            task=task,
                            fold=fold,
                            task_name=task_name,
                            method=method,
                        ),
                        **cache_class_kwargs
                    )
                    # FIXME: The output df still needs evaluation and formatting, currently just has predictions
                    # probabilities, fit and infer times
                    out = experiment.data(ignore_cache=ignore_cache)
                    result_lst.append(out)

        return result_lst

    # TODO: Nick: This should not be part of this class.
    def run_experiment(self, task, fold: int, task_name: str, method: str, init_args: dict = None, **kwargs):
        X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold)

        out = self.fit_custom2(X=X_train, y=y_train, X_test=X_test, y_test=y_test)

        out["framework"] = method
        out["dataset"] = task_name
        out["tid"] = task.task_id
        out["fold"] = fold
        out["problem_type"] = task.problem_type
        out["eval_metric"] = self.eval_metric
        print(f"Task  Name: {out['dataset']}")
        print(f"Task    ID: {out['tid']}")
        print(f"Metric    : {out['eval_metric']}")
        print(f"Test Error: {out['test_error']:.4f}")
        print(f"Fit   Time: {out['time_fit']:.3f}s")
        print(f"Infer Time: {out['time_predict']:.3f}s")

        out.pop("predictions")
        out.pop("probabilities")

        df_results = pd.DataFrame([out])
        ordered_columns = ["dataset", "fold", "framework", "test_error", "eval_metric", "time_fit"]
        columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
        df_results = df_results[columns_reorder]
        return df_results

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
    def fit_custom(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
        '''
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Returns
        -------
        dict
        Returns predictions, probabilities, fit time and inference time
        '''
        with (Timer() as timer_fit):
            self._fit(X_train, y_train)

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

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series):
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
            y_pred = pd.Series(y_pred, name=self.label, index=X.index)

        return y_pred, timer_predict

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
            y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)
        y_pred = y_pred_proba.idxmax(axis=1)

        return y_pred, y_pred_proba, timer_predict

    def _predict_proba(self, X: pd.DataFrame):
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

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer):
        if scorer.needs_pred:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred)
        elif self.problem_type == "binary":
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba.iloc[:, 1])
        else:
            test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
        return test_error
