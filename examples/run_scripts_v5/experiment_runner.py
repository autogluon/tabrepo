from __future__ import annotations

import pandas as pd

from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import get_metric, Scorer
from autogluon_benchmark.frameworks.autogluon.run import ag_eval_metric_map
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper


class ExperimentRunner:
    def __init__(
        self,
        method_cls,
        task: OpenMLTaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        fit_args: dict = None,
        cleanup: bool = True,
    ):
        self.method_cls = method_cls
        self.task = task
        self.fold = fold
        self.task_name = task_name
        self.method = method
        self.fit_args = fit_args
        self.cleanup = cleanup
        self.eval_metric_name = ag_eval_metric_map[self.task.problem_type]
        self.eval_metric: Scorer = get_metric(metric=self.eval_metric_name, problem_type=self.task.problem_type)
        self.model = None
        self.X, self.y, self.X_test, self.y_test = self.task.get_train_test_split(fold=self.fold)
        self.label_cleaner = LabelCleaner.construct(problem_type=self.task.problem_type, y=self.y)

    def init_method(self):
        model = self.method_cls(
            problem_type=self.task.problem_type,
            eval_metric=self.eval_metric,
            **self.fit_args,
        )
        return model

    def run_model_fit(self):
        return self.model.fit_custom(self.X, self.y, self.X_test)

    def run(self):
        out = self._run()
        if self.cleanup:
            self._cleanup()
        return out

    def _run(self):
        self.model = self.init_method()
        out = self.run_model_fit()
        out = self.post_fit(out=out)
        out["test_error"] = self.evaluate(
            y_true=self.y_test,
            y_pred=out["predictions"],
            y_pred_proba=out["probabilities"],
        )
        out = self.post_evaluate(out=out)
        out = self.convert_to_output(out=out)
        return out

    def post_fit(self, out):
        return out

    def post_evaluate(self, out):
        out["framework"] = self.method
        out["dataset"] = self.task_name
        out["tid"] = self.task.task_id
        out["fold"] = self.fold
        out["problem_type"] = self.task.problem_type
        out["eval_metric"] = self.eval_metric_name

        if self.model.can_get_oof:
            simulation_artifact = self.model.get_oof()
            simulation_artifacts = {self.task_name: {self.fold: simulation_artifact}}
        else:
            simulation_artifacts = None
        out["simulation_artifacts"] = simulation_artifacts
        return out

    def convert_to_output(self, out):
        out.pop("predictions")
        out.pop("probabilities")
        out.pop("simulation_artifacts")

        df_results = pd.DataFrame([out])
        ordered_columns = ["dataset", "fold", "framework", "test_error", "eval_metric", "time_fit"]
        columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
        df_results = df_results[columns_reorder]

        return df_results

    def evaluate(self, y_true, y_pred, y_pred_proba):
        return evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            scorer=self.eval_metric,
            label_cleaner=self.label_cleaner,
            problem_type=self.task.problem_type,
        )

    def _cleanup(self):
        self.model.cleanup()


class OOFExperimentRunner(ExperimentRunner):
    def post_evaluate(self, out):
        out = super().post_evaluate(out=out)
        if self.model.can_get_oof:
            simulation_artifact = self.model.get_oof()
            if self.task.problem_type == "regression":
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform(out["predictions"])
            else:
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform_proba(out["probabilities"], as_pandas=True)
                if self.task.problem_type == "binary":
                    simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].iloc[:, 1]
            simulation_artifact["y_test"] = self.label_cleaner.transform(self.y_test)
            simulation_artifact["label"] = self.task.label
            simulation_artifact["eval_metric"] = self.eval_metric_name

            out["metric_error_val"] = self.model.get_metric_error_val()
            # out["metric_error_val"] = evaluate(
            #     y_true=simulation_artifact["y_val"],
            #     y_pred=self.label_cleaner.transform(out["predictions"]),
            #     y_pred_proba=self.label_cleaner.transform_proba(out["probabilities"])
            # )
            # out["metric_error_val"] = self.eval_metric.error(simulation_artifact["y_val"], simulation_artifact["pred_proba_dict_val"])

            simulation_artifact["pred_proba_dict_val"] = {self.method: simulation_artifact["pred_proba_dict_val"]}
            simulation_artifact["pred_proba_dict_test"] = {self.method: simulation_artifact["pred_proba_dict_test"]}
            simulation_artifacts = {self.task_name: {self.fold: simulation_artifact}}
        else:
            simulation_artifacts = None
        out["simulation_artifacts"] = simulation_artifacts
        return out

    def convert_to_output(self, out):
        ignored_columns = ["predictions", "probabilities", "simulation_artifacts"]
        out_keys = list(out.keys())

        ordered_columns = ["dataset", "fold", "framework", "test_error", "eval_metric", "time_fit"]
        columns_reorder = ordered_columns + [c for c in out_keys if c not in ordered_columns and c not in ignored_columns]
        df_results = pd.DataFrame([{k: out[k] for k in columns_reorder}])
        df_results = df_results[columns_reorder]

        out["df_results"] = df_results
        out.pop("predictions")
        out.pop("probabilities")

        return out


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer, problem_type: str, label_cleaner: LabelCleaner = None,):
    if label_cleaner is None:
        label_cleaner = LabelCleanerDummy(problem_type=problem_type)
    y_true = label_cleaner.transform(y_true)
    if scorer.needs_pred:
        y_pred = label_cleaner.transform(y_pred)
        test_error = scorer.error(y_true=y_true, y_pred=y_pred)
    elif problem_type == "binary":
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        test_error = scorer.error(y_true=y_true, y_pred=pd.DataFrame(y_pred_proba).iloc[:, 1])
    else:
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        test_error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
    return test_error
