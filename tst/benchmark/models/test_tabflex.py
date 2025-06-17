from __future__ import annotations

import pytest


def test_tabicl():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.tabflex.tabflex_model import TabFlexModel

        model_cls = TabFlexModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
