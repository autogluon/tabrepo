import tempfile

import numpy as np
import pytest

from tabrepo.predictions import TabularPredictionsMemmap, TabularPredictionsInMemory, TabularPredictionsInMemoryOpt
from tabrepo.utils.test_utils import generate_artificial_dict, generate_dummy

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
    TabularPredictionsInMemoryOpt,
])
def test_predictions_shape(cls):
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        assert sorted(preds.datasets) == ["d1", "d2", "d3"]
        assert sorted(preds.models) == sorted(models)
        assert preds.folds == list(range(num_folds))

        single_models = models[0:1]
        # we test the shape and values obtained when querying predictions when querying 13 and 1 models
        for test_models in [models, single_models]:
            for dataset, (val_shape, test_shape) in dataset_shapes.items():
                val_pred_proba = preds.predict_val(dataset=dataset, fold=2, models=test_models)
                test_pred_proba = preds.predict_test(dataset=dataset, fold=2, models=test_models)
                assert val_pred_proba.shape == tuple([len(test_models)] + list(val_shape))
                assert test_pred_proba.shape == tuple([len(test_models)] + list(test_shape))
                for i, model in enumerate(test_models):
                    assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, test_models)[model])
                    assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, test_models)[model])


@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
    TabularPredictionsInMemoryOpt
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
    TabularPredictionsInMemoryOpt,
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

@pytest.mark.parametrize("cls", [
    TabularPredictionsMemmap,
    TabularPredictionsInMemory,
    TabularPredictionsInMemoryOpt,
])
def test_to_dict(cls):
    # Checks that to_dict returns the same dictionary as the original input
    with tempfile.TemporaryDirectory() as tmpdirname:
        preds = cls.from_dict(pred_dict, output_dir=tmpdirname)
        pred_dict2 = preds.to_dict()
        assert sorted(pred_dict.keys()) == sorted(pred_dict2.keys())
        for dataset in pred_dict.keys():
            assert sorted(pred_dict[dataset].keys()) == sorted(pred_dict2[dataset].keys())
            for fold, fold_dict in pred_dict[dataset].items():
                assert sorted(pred_dict[dataset][fold].keys()) == sorted(pred_dict2[dataset][fold].keys())
                for split, model_dict in fold_dict.items():
                    for model, model_value in model_dict.items():
                        assert np.allclose(pred_dict[dataset][fold][split][model], pred_dict2[dataset][fold][split][model])
