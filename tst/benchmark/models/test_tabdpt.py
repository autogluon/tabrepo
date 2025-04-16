import pytest

from tabrepo.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel


def test_tabdpt():
    pytest.skip("Skipping TabDPT unit test. TabDPT model weights must be manually downloaded.")
    model_cls = TabDPTModel
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        print(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
