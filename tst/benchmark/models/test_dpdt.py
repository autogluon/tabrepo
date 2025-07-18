import pytest


def test_dpdt():
    model_hyperparameters = {"n_estimators": 2, "cart_nodes_list":(4,3)}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag import BoostedDPDTModel
        model_cls = BoostedDPDTModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
