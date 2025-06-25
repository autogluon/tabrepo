from __future__ import annotations

import pytest


def test_beta_tabpfn():
    toy_model_params = {"batch_size": 8, "max_epoch": 10}
    model_hyperparameters = toy_model_params
    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.beta.beta_model import BetaModel

        model_cls = BetaModel
        FitHelper.verify_model(
            model_cls=model_cls, model_hyperparameters=model_hyperparameters
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
