import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from math import prod

from tabrepo.simulation.convert_memmap import convert_memmap_pred_from_pickle
from tabrepo.simulation.tabular_predictions import TabularPredictionsMemmap, \
    TabularPredictionsInMemory, TabularPredictionsDict

def generate_dummy(shape, models):
    return {
        model: np.arange(prod(shape)).reshape(shape) + int(model)
        for model in models
    }


def generate_artificial_dict(
        num_folds: int,
        models: List[str],
        dataset_shapes: dict = None,
):
    if dataset_shapes is None:
        dataset_shapes = {
            "d1": ((20,), (50,)),
            "d2": ((10,), (5,)),
            "d3": ((4, 3), (8, 3)),
        }
    # dictionary mapping dataset to fold to split to config name to predictions
    pred_dict: TabularPredictionsDict = {
        dataset: {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(num_folds)
        }
        for dataset, (val_shape, test_shape) in dataset_shapes.items()
    }
    return pred_dict


num_models = 13
num_folds = 3
dataset_shapes = {
    "d1": ((20,), (50,)),
    "d2": ((10,), (5,)),
    "d3": ((4, 3), (8, 3)),
}
models = [f"{i}" for i in range(num_models)]
pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)

@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
])
def test_predictions_shape(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        assert sorted(preds.datasets) == ["d1", "d2", "d3"]
        assert sorted(preds.models) == sorted(models)
        assert preds.folds == list(range(num_folds))

        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            val_pred_proba = preds.predict_val(dataset=dataset, fold=2, models=models)
            test_pred_proba = preds.predict_test(dataset=dataset, fold=2, models=models)
            assert val_pred_proba.shape == tuple([num_models] + list(val_shape))
            assert test_pred_proba.shape == tuple([num_models] + list(test_shape))
            for i, model in enumerate(models):
                assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, models)[model])
                assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, models)[model])


@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
])
def test_restrict_datasets(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_datasets(["d1", "d3"])
        assert sorted(preds.datasets) == ["d1", "d3"]

        print(pred_dict)
        preds.restrict_datasets([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []

@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
])
def test_restrict_models(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_models(["2", "3", "6"])
        assert sorted(preds.models) == ["2", "3", "6"]

        preds.restrict_models([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []
@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
])
def test_restrict_folds(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        preds.restrict_folds([0, 2])
        assert sorted(preds.folds) == [0, 2]

        preds.restrict_folds([])
        assert sorted(preds.datasets) == []
        assert sorted(preds.models) == []
        assert sorted(preds.folds) == []