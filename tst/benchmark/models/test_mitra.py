import pytest


def test_mitra():
    model_hyperparameters = {"n_estimators": 1}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.mitra.mitra_model import MitraModel
        model_cls = MitraModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )

if __name__ == "__main__":
    test_mitra()