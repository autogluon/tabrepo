from __future__ import annotations

import pytest


def test_tabpfnv2():
    model_hyperparameters = {
        "n_estimators": 1,
        # Check custom KDITransformer
        "inference_config/PREPROCESS_TRANSFORMS": [
            {
                "append_original": True,
                "categorical_name": "none",
                "global_transformer_name": None,
                "name": "kdi",
                "subsample_features": -1,
            }
        ],
    }

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabpfnv2.tabpfnv2_model import TabPFNV2Model

        model_cls = TabPFNV2Model
        FitHelper.verify_model(
            model_cls=model_cls, model_hyperparameters=model_hyperparameters
        )

        # Check DT-PFN version
        model_hyperparameters["model_type"] = "dt_pfn"
        model_hyperparameters["n_ensemble_repeats"] = 4
        FitHelper.verify_model(
            model_cls=model_cls, model_hyperparameters=model_hyperparameters
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
