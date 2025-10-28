import pytest


def test_ebm():
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
        model_cls = ExplainableBoostingMachineModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
