import copy

from typing import Callable

import numpy as np
import pytest

from tabrepo.contexts.context_artificial import load_repo_artificial
from tabrepo.repository import EvaluationRepository


def verify_equivalent_repository(repo1: EvaluationRepository, repo2: EvaluationRepository):
    assert repo1.folds == repo2.folds
    assert repo1.tids() == repo2.tids()
    assert repo1.configs() == repo2.configs()
    assert repo1.datasets() == repo2.datasets()
    for dataset in repo1.datasets():
        for c in repo1.configs():
            for f in repo1.folds:
                repo1_test = repo1.predict_test(dataset=dataset, config=c, fold=f)
                repo2_test = repo2.predict_test(dataset=dataset, config=c, fold=f)
                repo1_val = repo1.predict_val(dataset=dataset, config=c, fold=f)
                repo2_val = repo2.predict_val(dataset=dataset, config=c, fold=f)
                assert np.array_equal(repo1_test, repo2_test)
                assert np.array_equal(repo1_val, repo2_val)


def test_repository():
    repo = load_repo_artificial()
    dataset = 'abalone'
    tid = repo.dataset_to_tid(dataset)
    assert tid == 359946
    config = "NeuralNetFastAI_r1"  # TODO accessor

    assert repo.datasets() == ['abalone', 'ada']
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.dataset_to_tid(dataset) == 359946
    assert repo.configs() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.metrics(datasets=[dataset], configs=[config], folds=[2]))
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.predict_val_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 123, 25)
    assert repo.predict_test_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 13, 25)
    assert repo.labels_val(dataset=dataset, fold=2).shape == (123,)
    assert repo.labels_test(dataset=dataset, fold=2).shape == (13,)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    result_errors, result_ensemble_weights = repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")
    assert result_errors.shape == (3,)
    assert len(result_ensemble_weights) == 3

    dataset_info = repo.dataset_info(dataset=dataset)
    assert dataset_info["metric"] == "root_mean_squared_error"
    assert dataset_info["problem_type"] == "regression"

    # Test ensemble weights are as expected
    task_0 = repo.task_name(dataset=dataset, fold=0)
    assert np.allclose(result_ensemble_weights.loc[(dataset, 0)], [1.0, 0.0])

    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)

    repo: EvaluationRepository = repo.subset(folds=[0, 2])
    assert repo.datasets() == ['abalone', 'ada']
    assert repo.n_folds() == 2
    assert repo.folds == [0, 2]
    assert repo.tids() == [359946, 359944]
    assert repo.configs() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    # result_errors, result_ensemble_weights = repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0],
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0].shape == (2,)
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)

    repo: EvaluationRepository = repo.subset(folds=[2], datasets=[dataset], configs=[config])
    assert repo.datasets() == ['abalone']
    assert repo.n_folds() == 1
    assert repo.folds == [2]
    assert repo.tids() == [359946]
    assert repo.configs() == [config]
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0].shape == (1,)

    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)


def test_repository_force_to_dense():
    repo1 = load_repo_artificial()

    assert repo1.folds == [0, 1, 2]
    verify_equivalent_repository(repo1, repo1)

    repo2 = copy.deepcopy(repo1)
    repo2 = repo2.force_to_dense()  # no-op because already dense

    verify_equivalent_repository(repo1, repo2)

    repo2 = copy.deepcopy(repo1)
    repo2 = repo2.subset()  # no-op because already dense

    verify_equivalent_repository(repo1, repo2)

    repo2._zeroshot_context.subset_folds([1, 2])
    assert repo2.folds == [1, 2]
    with pytest.raises(AssertionError):
        verify_equivalent_repository(repo1, repo2)
    repo2 = repo2.force_to_dense()
    with pytest.raises(AssertionError):
        verify_equivalent_repository(repo1, repo2)
    repo3 = copy.deepcopy(repo1)
    repo3 = repo3.subset(folds=[1, 2])
    verify_equivalent_repository(repo2, repo3)


def test_repository_predict_binary_as_multiclass():
    """
    Test to verify that binary_as_multiclass logic works for binary problem_type
    """
    dataset = 'abalone'
    configs = ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']

    for problem_type in ["binary", "multiclass", "regression"]:
        if problem_type == "multiclass":
            n_classes = 3
        else:
            n_classes = 2
        repo = load_repo_artificial(n_classes=n_classes, problem_type=problem_type)
        assert repo.dataset_info(dataset)["problem_type"] == problem_type
        _assert_predict_multi_binary_as_multiclass(repo=repo, fun=repo.predict_val_multi, dataset=dataset, configs=configs, n_rows=123, n_classes=n_classes)
        _assert_predict_multi_binary_as_multiclass(repo=repo, fun=repo.predict_test_multi, dataset=dataset, configs=configs, n_rows=13, n_classes=n_classes)
        _assert_predict_binary_as_multiclass(repo=repo, fun=repo.predict_val, dataset=dataset, config=configs[0], n_rows=123, n_classes=n_classes)
        _assert_predict_binary_as_multiclass(repo=repo, fun=repo.predict_test, dataset=dataset, config=configs[0], n_rows=13, n_classes=n_classes)


def _assert_predict_multi_binary_as_multiclass(repo, fun: Callable, dataset, configs, n_rows, n_classes):
    problem_type = repo.dataset_info(dataset=dataset)["problem_type"]
    predict_multi = fun(dataset=dataset, fold=2, configs=configs)
    predict_multi_as_multiclass = fun(dataset=dataset, fold=2, configs=configs, binary_as_multiclass=True)
    if problem_type in ["binary", "regression"]:
        assert predict_multi.shape == (2, n_rows)
    else:
        assert predict_multi.shape == (2, n_rows, n_classes)
    if problem_type == "binary":
        assert predict_multi_as_multiclass.shape == (2, n_rows, 2)
        predict_multi_as_multiclass_to_binary = predict_multi_as_multiclass[:, :, 1]
        assert (predict_multi == predict_multi_as_multiclass_to_binary).all()
        assert (predict_multi_as_multiclass[:, :, 0] + predict_multi_as_multiclass[:, :, 1] == 1).all()
    else:
        assert (predict_multi == predict_multi_as_multiclass).all()


def _assert_predict_binary_as_multiclass(repo, fun: Callable, dataset, config, n_rows, n_classes):
    problem_type = repo.dataset_info(dataset=dataset)["problem_type"]
    predict = fun(dataset=dataset, fold=2, config=config)
    predict_as_multiclass = fun(dataset=dataset, fold=2, config=config, binary_as_multiclass=True)
    if problem_type in ["binary", "regression"]:
        assert predict.shape == (n_rows,)
    else:
        assert predict.shape == (n_rows, n_classes)
    if problem_type == "binary":
        assert predict_as_multiclass.shape == (n_rows, 2)
        predict_as_multiclass_to_binary = predict_as_multiclass[:, 1]
        assert (predict == predict_as_multiclass_to_binary).all()
        assert (predict_as_multiclass[:, 0] + predict_as_multiclass[:, 1] == 1).all()
    else:
        assert (predict == predict_as_multiclass).all()
