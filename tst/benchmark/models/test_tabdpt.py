import pytest



def test_tabdpt():
    pytest.skip("Skipping TabDPT unit test. TabDPT model weights must be manually downloaded.")
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
        model_cls = TabDPTModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
