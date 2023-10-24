import copy

import numpy as np
import pytest

from tabrepo.contexts.context_artificial import load_repo_artificial
from tabrepo.repository import EvaluationRepository


def verify_equivalent_repository(repo1: EvaluationRepository, repo2: EvaluationRepository):
    assert repo1.folds == repo2.folds
    assert repo1.tids() == repo2.tids()
    assert repo1.list_models() == repo2.list_models()
    assert repo1.datasets() == repo2.datasets()
    for tid in repo1.tids():
        for c in repo1.list_models():
            for f in repo1.folds:
                repo1_test = repo1.test_predictions(tid=tid, config_name=c, fold=f)
                repo2_test = repo2.test_predictions(tid=tid, config_name=c, fold=f)
                repo1_val = repo1.val_predictions(tid=tid, config_name=c, fold=f)
                repo2_val = repo2.val_predictions(tid=tid, config_name=c, fold=f)
                assert np.array_equal(repo1_test, repo2_test)
                assert np.array_equal(repo1_val, repo2_val)


def test_repository():
    repo = load_repo_artificial()
    dataset_name = 'abalone'
    tid = repo.dataset_to_tid(dataset_name)
    assert tid == 359946
    config_name = "NeuralNetFastAI_r1"  # TODO accessor

    assert repo.datasets() == ['abalone', 'ada']
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.dataset_to_tid(dataset_name) == 359946
    assert repo.list_models() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.eval_metrics(tid=tid, config_names=[config_name], fold=2))
    assert repo.val_predictions(tid=tid, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(tid=tid, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.predict_val(tid=tid, fold=2, configs=[config_name]).shape == (1, 123, 25)
    assert repo.predict_test(tid=tid, fold=2, configs=[config_name]).shape == (1, 13, 25)
    assert repo.labels_val(tid=tid, fold=2).shape == (123,)
    assert repo.labels_test(tid=tid, fold=2).shape == (13,)
    assert repo.dataset_metadata(tid=tid) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    result_errors, result_ensemble_weights = repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5, backend="native")
    assert result_errors.shape == (1, 3)
    assert len(result_ensemble_weights.keys()) == 3

    dataset_info = repo.dataset_info(tid=tid)
    assert dataset_info["metric"] == "root_mean_squared_error"
    assert dataset_info["problem_type"] == "regression"

    # Test ensemble weights are as expected
    task_0 = repo.task_name_from_dataset(dataset=dataset_name, fold=0)
    assert np.allclose(result_ensemble_weights[task_0], [1.0, 0.0])

    assert repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native")[0].shape == (1, 1)

    repo = repo.subset(folds=[0, 2])
    assert repo.datasets() == ['abalone', 'ada']
    assert repo.n_folds() == 2
    assert repo.folds == [0, 2]
    assert repo.tids() == [359946, 359944]
    assert repo.list_models() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    assert repo.val_predictions(tid=tid, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(tid=tid, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(tid=tid) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    # result_errors, result_ensemble_weights = repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5, backend="native")[0],
    assert repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5, backend="native")[0].shape == (1, 2)
    assert repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native")[0].shape == (1, 1)

    repo = repo.subset(folds=[2], tids=[359946], models=[config_name])
    assert repo.datasets() == ['abalone']
    assert repo.n_folds() == 1
    assert repo.folds == [2]
    assert repo.tids() == [359946]
    assert repo.list_models() == [config_name]
    assert repo.val_predictions(tid=tid, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(tid=tid, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(tid=tid) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5, backend="native")[0].shape == (1, 1)

    assert repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native")[0].shape == (1, 1)


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
