from tabrepo.benchmark.models.ag.tabicl.tabicl_model import TabICLModel


def test_tabicl():
    model_cls = TabICLModel
    model_hyperparameters = {"n_estimators": 1}

    try:
        from autogluon.tabular.testing import FitHelper
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        print(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
