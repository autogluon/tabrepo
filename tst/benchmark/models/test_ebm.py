from tabrepo.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel


def test_ebm():
    model_cls = ExplainableBoostingMachineModel
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
