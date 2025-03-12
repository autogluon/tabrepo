from __future__ import annotations

import copy
import shutil
from typing import Type

import pandas as pd

from tabrepo.benchmark.models.wrapper.abstract_class import AbstractExecModel


class AGWrapper(AbstractExecModel):
    can_get_oof = True

    def __init__(
        self,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        preprocess_data: bool = False,
        preprocess_label: bool = False,
        **kwargs,
    ):
        super().__init__(preprocess_data=preprocess_data, preprocess_label=preprocess_label, **kwargs)
        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.label = "__label__"

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self.label] = y

        fit_kwargs = self.fit_kwargs.copy()

        if X_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.label] = y_val
            fit_kwargs["tuning_data"] = tuning_data

        self.predictor = TabularPredictor(label=self.label, problem_type=self.problem_type, eval_metric=self.eval_metric, **self.init_kwargs)
        self.predictor.fit(train_data=train_data, **fit_kwargs)
        # FIXME: persist
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predictor.predict(X)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.predictor.predict_proba(X)
        return y_pred_proba

    def get_oof(self):
        # TODO: Rename method
        simulation_artifact = self.predictor.simulation_artifact()
        simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"][self.predictor.model_best]
        return simulation_artifact

    def get_metric_error_val(self) -> float:
        # FIXME: this shouldn't be calculating its own val score, that should be external. This should simply give val pred and val pred proba
        leaderboard = self.predictor.leaderboard(score_format="error")
        metric_error_val = leaderboard.set_index("model").loc[self.predictor.model_best]["metric_error_val"]
        return metric_error_val

    def cleanup(self):
        shutil.rmtree(self.predictor.path, ignore_errors=True)


class AGSingleWrapper(AGWrapper):
    """
    Wrapper for a single model being fit in AutoGluon

    Parameters
    ----------
    model_cls: str | Type["AbstractModel"]
        The model_cls normally used for the model family in `predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})
    model_hyperparameters
        The model_hyperparameters normally used in `predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})
    calibrate : bool | str, default False
    init_kwargs
    fit_kwargs
    preprocess_data
    preprocess_label
    kwargs

    """
    def __init__(
        self,
        model_cls: str | Type["AbstractModel"],
        model_hyperparameters: dict,
        calibrate: bool | str = False,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        preprocess_data: bool = False,
        preprocess_label: bool = False,
        **kwargs,
    ):
        from autogluon.tabular.models import AbstractModel
        assert (isinstance(model_cls, str) or issubclass(model_cls, AbstractModel))
        assert isinstance(model_hyperparameters, dict)

        if fit_kwargs is None:
            fit_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        assert "hyperparameters" not in fit_kwargs, f"Must not specify `hyperparameters` in AGSingleWrapper."
        assert "num_stack_levels" not in fit_kwargs, f"num_stack_levels is not allowed for `AGSingleWrapper"
        assert "presets" not in fit_kwargs, f"AGSingleWrapper does not support `presets`"
        assert "fit_weighted_ensemble" not in fit_kwargs, f"Must not specify `fit_weighted_ensemble` in AGSingleWrapper... It is always set to False."
        assert "calibrate" not in fit_kwargs, f"Specify calibrate directly rather than in `fit_kwargs`"
        assert "ag_args_fit" not in fit_kwargs, f"ag_args_fit must be specified in `model_hyperparameters`, not in `fit_kwargs` for `AGSingleWrapper"
        assert "ag_args_ensemble" not in fit_kwargs, f"ag_args_ensemble must be specified in `model_hyperparameters`, not in `fit_kwargs` for `AGSingleWrapper"

        self.init_kwargs_extra = init_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["calibrate"] = calibrate

        self.fit_kwargs_extra = fit_kwargs
        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["fit_weighted_ensemble"] = False
        fit_kwargs["hyperparameters"] = {model_cls: model_hyperparameters}

        self._model_cls = model_cls
        self.model_hyperparameters = model_hyperparameters

        super().__init__(init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, preprocess_data=preprocess_data, preprocess_label=preprocess_label, **kwargs)

    def get_hyperparameters(self):
        hyperparameters = self.predictor.model_hyperparameters(model=self.predictor.model_best, output_format="user")
        return hyperparameters

    @property
    def model_cls(self) -> Type["AbstractModel"]:
        if not isinstance(self._model_cls, str):
            model_cls = self._model_cls
        else:
            # TODO: Get it from predictor instead? What if we allow passing custom model register?
            from autogluon.tabular.register import ag_model_register  # If this raises an exception, you need to update to latest mainline AutoGluon
            model_cls = ag_model_register.key_to_cls(key=self._model_cls)
        return model_cls

    def get_metadata(self) -> dict:
        metadata = {}

        model = self.predictor._trainer.load_model(self.predictor.model_best)
        metadata["info"] = model.get_info(include_feature_metadata=False)
        metadata["hyperparameters"] = self.get_hyperparameters()
        metadata["model_cls"] = self.model_cls.__name__
        metadata["model_type"] = self.model_cls.ag_key  # TODO: rename to ag_key?
        metadata["name_prefix"] = self.model_cls.ag_name  # TODO: rename to ag_name?
        metadata["model_hyperparameters"] = self.model_hyperparameters
        metadata["init_kwargs_extra"] = self.init_kwargs_extra
        metadata["fit_kwargs_extra"] = self.fit_kwargs_extra
        metadata["disk_usage"] = model.disk_usage()
        metadata["num_cpus"] = model.fit_num_cpus
        metadata["num_gpus"] = model.fit_num_gpus
        metadata["num_cpus_child"] = model.fit_num_cpus_child
        metadata["num_gpus_child"] = model.fit_num_gpus_child
        metadata["fit_metadata"] = model.get_fit_metadata()
        return metadata
