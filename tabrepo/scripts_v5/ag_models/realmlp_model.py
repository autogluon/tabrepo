import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel


# TODO: This doesn't actually work for all tasks, need to improve
# pip install pytabkit
class RealMLPModel(AbstractModel):
    def get_model_cls(self):
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_S_Regressor

        if self.problem_type in ['binary', 'multiclass']:
            model_cls = RealMLP_TD_Classifier
        else:
            model_cls = RealMLP_TD_S_Regressor
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, time_limit: float = None, **kwargs):
        model_cls = self.get_model_cls()

        metric_map = {
            "roc_auc": "cross_entropy",
            "accuracy": "class_error",
            "balanced_accuracy": "1-balanced_accuracy",
            "log_loss": "cross_entropy",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "rmse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        val_metric_name = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = dict()

        if val_metric_name is not None:
            init_kwargs["val_metric_name"] = val_metric_name

        hyp = self._get_model_params()
        self.model = model_cls(**init_kwargs, **hyp)

        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        # TODO: Categorical indicator?
        print(f'start {self.name}')
        self.model = self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_to_fit_in_seconds=time_limit,
        )

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "problem_types": [BINARY, MULTICLASS, REGRESSION],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
           "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
