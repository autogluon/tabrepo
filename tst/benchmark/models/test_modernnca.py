import pytest


def test_modernnca():
    toy_model_params = {"n_epochs": 10}
    model_hyperparameters = toy_model_params

    from autogluon.tabular.testing import FitHelper
    from tabrepo.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
    model_cls = ModernNCAModel
    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
