import copy

import numpy as np
import pytest

from autogluon_zeroshot.contexts.context_artificial import load_context_artificial
from autogluon_zeroshot.repository import EvaluationRepository


def verify_equivalent_repository(repo1: EvaluationRepository, repo2: EvaluationRepository):
    assert repo1.folds == repo2.folds
    assert repo1.tids() == repo2.tids()
    assert repo1.list_models() == repo2.list_models()
    assert repo1.dataset_names() == repo2.dataset_names()
    for tid in repo1.tids():
        dataset_name = repo1.tid_to_dataset(tid)
        for c in repo1.list_models_available(tid=tid):
            for f in repo1.folds:
                repo1_test = repo1.test_predictions(dataset_name=dataset_name, config_name=c, fold=f)
                repo2_test = repo2.test_predictions(dataset_name=dataset_name, config_name=c, fold=f)
                repo1_val = repo1.val_predictions(dataset_name=dataset_name, config_name=c, fold=f)
                repo2_val = repo2.val_predictions(dataset_name=dataset_name, config_name=c, fold=f)
                assert np.array_equal(repo1_test, repo2_test)
                assert np.array_equal(repo1_val, repo2_val)


def test_repository():
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
    repo = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )
    dataset_name = 'abalone'
    tid = repo.dataset_to_tid(dataset_name)
    assert tid == 359946
    config_name = "NeuralNetFastAI_r1"  # TODO accessor

    assert repo.dataset_names() == ['ada', 'abalone']
    assert repo.tids() == [359944, 359946]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.dataset_to_tid(dataset_name) == 359946
    assert repo.list_models_available(tid) == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.eval_metrics(dataset_name=dataset_name, config_names=[config_name], fold=2))
    assert repo.val_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset_name=dataset_name) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name], ensemble_size=5, backend="native"),
        [[2.5, 2.5, 2.5]]
    )
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native"),
        [[2.5]]
    )

    repo = repo.subset(folds=[0, 2])
    assert repo.dataset_names() == ['ada', 'abalone']
    assert repo.n_folds() == 2
    assert repo.folds == [0, 2]
    assert repo.tids() == [359944, 359946]
    assert repo.list_models() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    assert repo.val_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset_name=dataset_name) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name], ensemble_size=5, backend="native"),
        [[2.5, 2.5]]
    )
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native"),
        [[2.5]]
    )

    repo = repo.subset(folds=[2], tids=[359946], models=[config_name])
    assert repo.dataset_names() == ['abalone']
    assert repo.n_folds() == 1
    assert repo.folds == [2]
    assert repo.tids() == [359946]
    assert repo.list_models() == [config_name]
    assert repo.val_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset_name=dataset_name) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name], ensemble_size=5, backend="native"),
        [[2.5]]
    )
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native"),
        [[2.5]]
    )


def test_repository_force_to_dense():
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
    repo1 = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )

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
