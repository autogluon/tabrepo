from __future__ import annotations

import copy
from typing import Type

from autogluon.core.models import AbstractModel

from tabrepo.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGSingleWrapper
from tabrepo.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionDummy
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper


class Experiment:
    """
    Experiment contains a method and the logic to run it on any task.
    Experiment is fully generic, and can accept any method_cls so long as it inherits from `AbstractExecModel`.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    method_cls: Type[AbstractExecModel]
        The method class to be fit and evaluated.
    method_kwargs: dict
        The kwargs passed to the init of `method_cls`.
    experiment_cls: Type[ExperimentRunner], default OOFExperimentRunner
        The experiment class that wraps the method_cls.
        This class will track metadata information such as fit time, inference time, system resources, etc.
        It will also calculate the test `metric_error`, to ensure that the method_cls is evaluated correctly.
    experiment_kwargs: dict, optional
        The kwargs passed to the init of `experiment_cls`.

    """
    def __init__(
        self,
        name: str,
        method_cls: Type[AbstractExecModel],
        method_kwargs: dict,
        *,
        experiment_cls: Type[ExperimentRunner] = OOFExperimentRunner,
        experiment_kwargs: dict = None,
    ):
        if experiment_kwargs is None:
            experiment_kwargs = {}
        assert isinstance(name, str)
        assert len(name) > 0, "Name cannot be empty!"
        assert isinstance(method_kwargs, dict)
        assert isinstance(experiment_kwargs, dict)
        self.name = name
        self.method_cls = method_cls
        self.method_kwargs = method_kwargs
        self.experiment_cls = experiment_cls
        self.experiment_kwargs = experiment_kwargs

    def construct_method(self, problem_type: str, eval_metric) -> AbstractExecModel:
        return self.method_cls(
            problem_type=problem_type,
            eval_metric=eval_metric,
            **self.method_kwargs,
        )

    def run(
        self,
        task: OpenMLTaskWrapper | None,
        fold: int,
        task_name: str,
        cacher: AbstractCacheFunction | None = None,
        ignore_cache: bool = False,
    ) -> object:
        if cacher is None:
            cacher = CacheFunctionDummy()
        if task is not None:
            out = cacher.cache(
                fun=self.experiment_cls.init_and_run,
                fun_kwargs=dict(
                    method_cls=self.method_cls,
                    task=task,
                    fold=fold,
                    task_name=task_name,
                    method=self.name,
                    fit_args=self.method_kwargs,
                    **self.experiment_kwargs,
                ),
                ignore_cache=ignore_cache,
            )
        else:
            # load cache, no need to load task
            out = cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)
        return out


# convenience wrapper
class AGModelExperiment(Experiment):
    """
    Simplified Experiment class specifically for fitting a single model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    model_cls: Type[AbstractModel]
        AutoGluon model class to fit
    model_hyperparameters: dict
        AutoGluon model hyperparameters
        Identical to what you would pass to `TabularPredictor.fit(..., hyperparameters={model_cls: [model_hyperparameters]})
    time_limit: float, optional
        The time limit in seconds the model is allowed to fit for.
        If unspecified, no time limit is enforced.
    raise_on_model_failure: bool, default True
        By default sets raise_on_model_failure to True
        so that any AutoGluon model failure will be raised in a debugger friendly manner.
    method_kwargs: dict, optional
        The kwargs passed to the init of `method_cls`.
    experiment_kwargs: dict, optional
        The kwargs passed to the init of `experiment_cls`.
    """

    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        raise_on_model_failure: bool = True,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        if time_limit is not None:
            assert isinstance(time_limit, (float, int))
            assert time_limit > 0
        if "fit_kwargs" in method_kwargs:
            assert "time_limit" not in method_kwargs["fit_kwargs"], \
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
        assert isinstance(model_hyperparameters, dict)
        if time_limit is not None:
            model_hyperparameters = self._insert_time_limit(model_hyperparameters=model_hyperparameters, time_limit=time_limit, method_kwargs=method_kwargs)
        if "fit_kwargs" not in method_kwargs:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        assert "raise_on_model_failure" not in method_kwargs["fit_kwargs"], \
            f"Set `raise_on_model_failure` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
        method_kwargs["fit_kwargs"]["raise_on_model_failure"] = raise_on_model_failure
        super().__init__(
            name=name,
            method_cls=AGSingleWrapper,
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": model_hyperparameters,
                **method_kwargs,
            },
            experiment_cls=OOFExperimentRunner,
            experiment_kwargs=experiment_kwargs,
        )

    def _insert_time_limit(self, model_hyperparameters: dict, time_limit: float | None, method_kwargs: dict) -> dict:
        is_bag = False
        if "fit_kwargs" in method_kwargs and "num_bag_folds" in method_kwargs["fit_kwargs"]:
            if method_kwargs["fit_kwargs"]["num_bag_folds"] > 1:
                is_bag = True
        model_hyperparameters = copy.deepcopy(model_hyperparameters)
        if is_bag:
            if "ag_args_ensemble" in model_hyperparameters:
                assert "ag.max_time_limit" not in model_hyperparameters["ag_args_ensemble"], \
                    f"Set `time_limit` directly in {self.__class__.__name__} rather than in `ag_args_ensemble`"
            else:
                model_hyperparameters["ag_args_ensemble"] = {}
            model_hyperparameters["ag_args_ensemble"]["ag.max_time_limit"] = time_limit
        else:
            assert "ag.max_time_limit" not in model_hyperparameters, \
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `model_hyperparameters`"
            model_hyperparameters["ag.max_time_limit"] = time_limit
        return model_hyperparameters


# convenience wrapper
class AGModelBagExperiment(AGModelExperiment):
    """
    Simplified Experiment class specifically for fitting a single bagged model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner

    All models fit this way will generate out-of-fold predictions on the entire training set,
    and will be compatible with ensemble simulations in TabRepo.

    Will fit the model with `num_bag_folds` folds and `num_bag_sets` sets (aka repeats).
    In total will fit `num_bag_folds * num_bag_sets` models in the bag.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    model_cls: Type[AbstractModel]
    model_hyperparameters: dict
        Identical to what you would pass to `TabularPredictor.fit(..., hyperparameters={model_cls: [model_hyperparameters]})
    time_limit: float, optional
    num_bag_folds: int, default 8
    num_bag_sets: int, default 1
    method_kwargs: dict, optional
    experiment_kwargs: dict, optional
    """
    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        assert isinstance(num_bag_folds, int)
        assert isinstance(num_bag_sets, int)
        assert isinstance(method_kwargs, dict)
        assert num_bag_folds >= 2
        assert num_bag_sets >= 1
        if "fit_kwargs" in method_kwargs:
            assert "num_bag_folds" not in method_kwargs["fit_kwargs"], f"Set `num_bag_folds` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            assert "num_bag_sets" not in method_kwargs["fit_kwargs"], f"Set `num_bag_sets` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        else:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"]["num_bag_folds"] = num_bag_folds
        method_kwargs["fit_kwargs"]["num_bag_sets"] = num_bag_sets
        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            time_limit=time_limit,
            method_kwargs=method_kwargs,
            experiment_kwargs=experiment_kwargs,
        )
