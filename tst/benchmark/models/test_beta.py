from __future__ import annotations


def test_beta_tabpfn():
    toy_model_params = {"batch_size": 8, "max_epoch": 10}
    model_hyperparameters = toy_model_params

    from autogluon.tabular.testing import FitHelper
    from tabrepo.benchmark.models.ag.beta.beta_model import BetaModel

    model_cls = BetaModel
    FitHelper.verify_model(
        model_cls=model_cls, model_hyperparameters=model_hyperparameters
    )
