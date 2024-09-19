import numpy as np
import pytest

from tabrepo import EvaluationRepositoryCollection
from tabrepo.contexts.context_artificial import load_repo_artificial

from .test_repository import verify_equivalent_repository


def test_repository_collection():
    repo = load_repo_artificial()

    assert repo.datasets() == ['abalone', 'ada']
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.configs() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']

    datasets = repo.datasets()
    folds = repo.folds
    configs = repo.configs()

    repo_1 = repo.subset(configs=['NeuralNetFastAI_r2'])
    repo_2 = repo.subset(configs=["NeuralNetFastAI_r1"])

    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])

    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)

    for dataset in datasets:
        for fold in folds:
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=fold, config="NeuralNetFastAI_r2") == 0
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=fold, config="NeuralNetFastAI_r1") == 1

            predict_test = repo.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            predict_test_collection = repo_collection.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_test, predict_test_collection)

            predict_val = repo.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            predict_val_collection = repo_collection.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_val, predict_val_collection)

    repo_1 = repo.subset(datasets=['abalone'])
    repo_2 = repo.subset(datasets=["ada"])
    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for config in configs:
        for fold in folds:
            assert repo_collection.get_result_to_repo_idx(dataset="abalone", fold=fold, config=config) == 0
            assert repo_collection.get_result_to_repo_idx(dataset="ada", fold=fold, config=config) == 1

    repo_1 = repo.subset(folds=[2, 0])
    repo_2 = repo.subset(folds=[1])
    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for config in configs:
        for dataset in datasets:
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=0, config=config) == 0
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=1, config=config) == 1
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=2, config=config) == 0


def test_repository_collection_overlap_raise():
    repo = load_repo_artificial()
    with pytest.raises(AssertionError):
        EvaluationRepositoryCollection(repos=[repo, repo])


def test_repository_collection_overlap_first():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo, repo], overlap="first")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 0


def test_repository_collection_overlap_last():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo, repo, repo], overlap="last")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 2
    repo_collection_nested = EvaluationRepositoryCollection(repos=[repo, repo_collection], overlap="last")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection_nested, verify_ensemble=True)
    for v in repo_collection_nested._mapping.values():
        assert v == 1


def test_repository_collection_single():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 0
    repo_collection_nested = EvaluationRepositoryCollection(repos=[repo_collection])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection_nested, verify_ensemble=True)
    for v in repo_collection_nested._mapping.values():
        assert v == 0
