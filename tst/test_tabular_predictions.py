import tempfile

from math import prod
from typing import List

import numpy as np
from pathlib import Path

import pytest

from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions, TabularPredictionsDict, \
    TabularPicklePerTaskPredictions, TabularNpyPerTaskPredictions


def generate_dummy(shape, models):
    return {
        model: np.arange(prod(shape)).reshape(shape) + int(model)
        for model in models
    }


def generate_artificial_dict(
        num_folds: int,
        models: List[str],
        dataset_shapes={
            "d1": ((20,), (50,)),
            "d2": ((10,), (5,)),
            "d3": ((4, 3), (8, 3)),
        },
):
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


# def check_synthetic_data_pickle(cls=TabularPicklePredictions):
@pytest.mark.parametrize("cls", [TabularPicklePredictions, TabularPicklePerTaskPredictions, TabularNpyPerTaskPredictions])
def test_synthetic_data(cls):
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]

    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)

    with tempfile.TemporaryDirectory() as tmpdirname:

        # 1) construct pred proba from dictionary
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        assert set(pred_proba.models_available_in_dataset(dataset="d1")) == set(models)
        filename = str(Path(tmpdirname) / "dummy")

        # 2) save it and reload it
        pred_proba.save(filename)
        pred_proba = cls.load(filename)

        # 3) checks that output is as expected after serializing/deserializing
        assert pred_proba.datasets == list(dataset_shapes.keys())
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            print(dataset, val_shape, test_shape)
            val_score, test_score = pred_proba.predict(dataset=dataset, fold=2, models=models, splits=["val", "test"])
            assert val_score.shape == tuple([num_models] + list(val_shape))
            assert test_score.shape == tuple([num_models] + list(test_shape))
            for i, model in enumerate(models):
                assert np.allclose(val_score[i], generate_dummy(val_shape, models)[model])
                assert np.allclose(test_score[i], generate_dummy(test_shape, models)[model])


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePerTaskPredictions,
    # TabularNpyPerTaskPredictions
    # TODO restricting models with this format does not work which is ok as this
    #  format is not  currently used in experiments.
])
def test_restrict_models(cls):

    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    num_sub_models = num_models // 2
    sub_models = models[:num_sub_models]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        pred_proba.restrict_models(sub_models)
        assert sorted(pred_proba.models) == sorted(sub_models)

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            print(dataset, val_shape, test_shape)
            val_score, test_score = pred_proba.predict(dataset=dataset, fold=2, models=sub_models, splits=["val", "test"])
            assert val_score.shape == tuple([num_sub_models] + list(val_shape))
            assert test_score.shape == tuple([num_sub_models] + list(test_shape))
            for i, model in enumerate(sub_models):
                assert np.allclose(val_score[i], generate_dummy(val_shape, sub_models)[model])
                assert np.allclose(test_score[i], generate_dummy(test_shape, sub_models)[model])
