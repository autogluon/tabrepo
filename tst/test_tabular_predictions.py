import tempfile

import copy
from math import prod
from typing import List

import numpy as np
from pathlib import Path

import pytest

from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions, TabularPredictionsDict, \
    TabularPicklePerTaskPredictions, TabularNpyPerTaskPredictions, TabularPicklePredictionsOpt


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


# def check_synthetic_data_pickle(cls=TabularPicklePredictions):
@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
    # TabularNpyPerTaskPredictions  # TODO: Not fully implemented
])
def test_save_load_equivalence(cls):
    """
    Ensure predictions behave identically after loading from file
    """
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        3: ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]

    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)

    with tempfile.TemporaryDirectory() as tmpdirname:

        # 1) construct pred proba from dictionary
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        assert set(pred_proba.models_available_in_dataset(dataset="d1")) == set(models)
        assert set(pred_proba.models_available_in_dataset(dataset=3)) == set(models)
        filename = str(Path(tmpdirname) / "dummy")

        # 2) save it and reload it
        pred_proba.save(filename)
        pred_proba_load = cls.load(filename)

        assert pred_proba.folds == pred_proba_load.folds
        assert pred_proba.datasets == pred_proba_load.datasets
        assert pred_proba.tasks == pred_proba_load.tasks
        assert pred_proba.models == pred_proba_load.models

        # Ensure can load from int dataset names still
        assert set(pred_proba_load.models_available_in_dataset(dataset=3)) == set(models)

        # 3) checks that output is as expected after serializing/deserializing
        assert pred_proba_load.datasets == list(dataset_shapes.keys())
        for cur_pred_proba in [pred_proba, pred_proba_load]:
            for dataset, (val_shape, test_shape) in dataset_shapes.items():
                print(dataset, val_shape, test_shape)
                val_pred_proba, test_pred_proba = cur_pred_proba.predict(dataset=dataset, fold=2, models=models, splits=["val", "test"])
                assert val_pred_proba.shape == tuple([num_models] + list(val_shape))
                assert test_pred_proba.shape == tuple([num_models] + list(test_shape))
                for i, model in enumerate(models):
                    assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, models)[model])
                    assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, models)[model])


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
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
            val_pred_proba, test_pred_proba = pred_proba.predict(dataset=dataset, fold=2, models=sub_models, splits=["val", "test"])
            assert val_pred_proba.shape == tuple([num_sub_models] + list(val_shape))
            assert test_pred_proba.shape == tuple([num_sub_models] + list(test_shape))
            for i, model in enumerate(sub_models):
                assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, sub_models)[model])
                assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, sub_models)[model])


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets(cls):
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
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        pred_proba.restrict_datasets(["d1", "d3"])
        assert pred_proba.datasets == ["d1", "d3"]

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            if dataset == "d2":
                continue
            print(dataset, val_shape, test_shape)
            val_pred_proba, test_pred_proba = pred_proba.predict(dataset=dataset, fold=2, models=models, splits=["val", "test"])
            assert val_pred_proba.shape == tuple([num_models] + list(val_shape))
            assert test_pred_proba.shape == tuple([num_models] + list(test_shape))
            for i, model in enumerate(models):
                assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, models)[model])
                assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, models)[model])

@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets_dense(cls):
    val_shape = (4, 3)
    test_shape = (8, 3)
    pred_dict = {
        "d1": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["1", "2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "2", "3"]),
            }
            for fold in range(10)
        },
        "d2": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "3"]),
            }
            for fold in range(10)
        },
        3: {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, ["1", "2", "3"]),
                "pred_proba_dict_test": generate_dummy(test_shape, ["1", "2", "3"]),
            }
            for fold in range(10)
        },
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        models = ["1", "2", "3"]
        valid_datasets = [
            dataset
            for dataset in pred_proba.datasets
            if all(m in pred_proba.models_available_in_dataset(dataset) for m in models)
        ]
        assert valid_datasets == ["d1", 3]
        pred_proba.restrict_datasets(valid_datasets)
        assert pred_proba.datasets == ["d1", 3]


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
])
def test_restrict_datasets_missing_fold(cls):
    val_shape = (4, 3)
    test_shape = (8, 3)
    models = ["1", "2", "3"]

    pred_dict = {
        "d1": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(10)
        },
        "d2": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in [x for x in range(10) if x != 3]
        },
        "d3": {
            fold: {
                "pred_proba_dict_val": generate_dummy(val_shape, models),
                "pred_proba_dict_test": generate_dummy(test_shape, models),
            }
            for fold in range(10)
        },
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)
        assert pred_proba.models_available_in_dataset("d1", present_in_all=True) == models
        assert pred_proba.models_available_in_dataset("d2", present_in_all=True) == models
        assert pred_proba.models_available_in_dataset("d3", present_in_all=True) == models
        valid_datasets = [
            dataset
            for dataset in pred_proba.datasets
            if all(m in pred_proba.models_available_in_dataset(dataset, present_in_all=True) for m in models)
        ]
        assert valid_datasets == ["d1", "d2", "d3"]
        assert pred_proba.datasets == ["d1", "d2", "d3"]

        assert not pred_proba.is_dense_folds()
        pred_proba.force_to_dense_folds()
        assert pred_proba.is_dense_folds()

        assert pred_proba.datasets == ["d1", "d3"]
        assert pred_proba.models_available_in_dataset("d1", present_in_all=True) == models
        assert pred_proba.models_available_in_dataset("d3", present_in_all=True) == models


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
])
def test_advanced(cls):
    """Tests a variety of advanced functionality"""
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
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        datasets_og = pred_proba.datasets
        pred_proba.restrict_datasets(datasets_og)
        assert datasets_og == pred_proba.datasets

        folds_og = pred_proba.folds
        pred_proba.restrict_folds(folds_og)
        assert folds_og == pred_proba.folds

        models_og = pred_proba.models
        pred_proba.restrict_models(models_og)
        assert models_og == pred_proba.models

        with pytest.raises(AssertionError):
            pred_proba.restrict_datasets(["unknown_dataset"])
        with pytest.raises(AssertionError):
            pred_proba.restrict_folds(["unknown_fold"])
        with pytest.raises(AssertionError):
            pred_proba.restrict_models(["unknown_model"])

        pred_proba.restrict_datasets(["d1", "d3"])
        assert pred_proba.datasets == ["d1", "d3"]

        with pytest.raises(AssertionError):
            # Cant rename dataset that does not exist (d2)
            copy.deepcopy(pred_proba).rename_datasets({'d1': 123, 'd2': 456})
        with pytest.raises(AssertionError):
            # Cant overlap dataset names
            copy.deepcopy(pred_proba).rename_datasets({'d1': 'd3'})

        rename_dict = {'d1': 123}
        pred_proba.rename_datasets(rename_dict=rename_dict)
        assert pred_proba.datasets == [123, "d3"]
        with pytest.raises(AssertionError):
            # d1 is no longer present
            copy.deepcopy(pred_proba).rename_datasets({'d1': 'd4'})

        # ensure resilient to adversarial renaming
        pred_proba.rename_datasets(rename_dict={123: 'd3', 'd3': 123})
        assert pred_proba.datasets == ['d3', 123]
        rename_dict = {'d1': 'd3', 'd3': 123}

        filename = str(Path(tmpdirname) / "dummy")
        # 2) save it and reload it, ensure the renaming stays
        pred_proba.save(filename)
        pred_proba_load = cls.load(filename)

        assert pred_proba.folds == pred_proba_load.folds
        assert pred_proba.datasets == pred_proba_load.datasets
        assert pred_proba.tasks == pred_proba_load.tasks
        assert pred_proba.models == pred_proba_load.models
        pred_proba = pred_proba_load

        pred_proba.restrict_folds([1, 2])
        assert pred_proba.folds == [1, 2]

        pred_proba.restrict_models(["3", "7", "11"])
        assert pred_proba.models == sorted(["3", "7", "11"])

        with pytest.raises(AssertionError):
            pred_proba.restrict_datasets(datasets_og)
        with pytest.raises(AssertionError):
            pred_proba.restrict_folds(folds_og)
        with pytest.raises(AssertionError):
            pred_proba.restrict_models(models_og)

        # make sure shapes matches what is expected
        for dataset, (val_shape, test_shape) in dataset_shapes.items():
            dataset = rename_dict.get(dataset, dataset)
            for fold in folds_og:
                for models in [
                    ["3"],  # valid
                    ["11", "7"],  # valid
                    ["11", "2"],  # invalid
                    ["2", "4"],  # invalid
                ]:
                    models_are_valid = False not in [m in pred_proba.models for m in models]
                    should_raise = dataset not in pred_proba.datasets or fold not in pred_proba.folds or not models_are_valid
                    print(dataset, fold, val_shape, test_shape, models, should_raise)
                    if should_raise:
                        with pytest.raises(Exception):
                            pred_proba.predict(dataset=dataset, fold=fold, models=models, splits=["val", "test"])
                    else:
                        val_pred_proba, test_pred_proba = pred_proba.predict(dataset=dataset, fold=fold, models=models, splits=["val", "test"])
                        assert val_pred_proba.shape == tuple([len(models)] + list(val_shape))
                        assert test_pred_proba.shape == tuple([len(models)] + list(test_shape))
                        for i, model in enumerate(models):
                            assert np.allclose(val_pred_proba[i], generate_dummy(val_shape, models)[model])
                            assert np.allclose(test_pred_proba[i], generate_dummy(test_shape, models)[model])


def _make_empty_and_assert(pred_proba):
    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_folds([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.tasks == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []
    assert pred_proba_copy.is_empty()

    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_datasets([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.tasks == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []
    assert pred_proba_copy.is_empty()

    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_tasks([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.tasks == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []
    assert pred_proba_copy.is_empty()

    pred_proba_copy = copy.deepcopy(pred_proba)
    pred_proba_copy.restrict_models([])
    assert pred_proba_copy.datasets == []
    assert pred_proba_copy.tasks == []
    assert pred_proba_copy.folds == []
    assert pred_proba_copy.models == []
    assert pred_proba_copy.is_empty()


@pytest.mark.parametrize("cls", [
    TabularPicklePredictions,
    TabularPicklePredictionsOpt,
    TabularPicklePerTaskPredictions,
])
def test_sparse_to_dense(cls):
    """Tests sparse input"""
    num_models = 13
    num_folds = 3
    dataset_shapes = {
        "d1": ((20,), (50,)),
        "d2": ((10,), (5,)),
        "d3": ((4, 3), (8, 3)),
    }
    models = [f"{i}" for i in range(num_models)]
    pred_dict = generate_artificial_dict(num_folds, models, dataset_shapes)
    pred_dict['d1'].pop(0)
    pred_dict['d1'][1]['pred_proba_dict_val'].pop('8')
    pred_dict['d2'][1]['pred_proba_dict_test'].pop('5')
    pred_dict['d3'].pop(2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred_proba = cls.from_dict(pred_dict=pred_dict, output_dir=tmpdirname)

        init_folds = pred_proba.folds
        init_folds_dense = pred_proba.get_folds_dense()
        init_models = pred_proba.models
        init_models_dense = pred_proba.get_models_dense()
        init_datasets = pred_proba.datasets

        assert init_folds == [0, 1, 2]
        assert init_folds_dense == [1]
        assert init_models == ['0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9']
        assert init_models_dense == ['0', '1', '10', '11', '12', '2', '3', '4', '6', '7', '9']
        assert init_datasets == ['d1', 'd2', 'd3']
        assert not pred_proba.is_dense_models()
        assert not pred_proba.is_dense_folds()
        assert not pred_proba.is_empty()

        pred_proba_copy = copy.deepcopy(pred_proba)
        with pytest.raises(AssertionError):
            pred_proba_copy.force_to_dense(first_prune_method='task', second_prune_method='dataset')
        pred_proba_copy = copy.deepcopy(pred_proba)
        with pytest.raises(AssertionError):
            pred_proba_copy.force_to_dense(first_prune_method='task', second_prune_method='fold')

        print(f'Pre Dense')
        pred_proba.print_summary()

        _make_empty_and_assert(pred_proba=pred_proba)

        pred_proba.restrict_folds([0, 1])
        pred_proba.force_to_dense(first_prune_method='task', second_prune_method='dataset')

        print(f'Post Dense')
        pred_proba.print_summary()

        post_folds = pred_proba.folds
        post_folds_dense = pred_proba.get_folds_dense()
        post_models = pred_proba.models
        post_models_dense = pred_proba.get_models_dense()
        post_datasets = pred_proba.datasets

        assert post_folds == [0, 1]
        assert post_folds_dense == post_folds
        assert post_models == init_models
        assert post_models_dense == init_models
        assert post_datasets == ['d3']
        assert pred_proba.is_dense_models()
        assert pred_proba.is_dense_folds()
        assert not pred_proba.is_empty()

        # Additionally check that restricting to nothing results in empty

        _make_empty_and_assert(pred_proba=pred_proba)