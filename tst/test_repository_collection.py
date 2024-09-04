import numpy as np

from tabrepo.contexts.context_artificial import load_repo_artificial
from tabrepo.repository.evaluation_repository_collection import EvaluationRepositoryCollection

from .test_repository import verify_equivalent_repository


# TODO: Test more functionality
# TODO: Test different datasets, different folds, etc.
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

    verify_equivalent_repository(repo1=repo, repo2=repo_collection)

    for dataset in datasets:
        for fold in folds:
            assert repo_collection.goes_where(dataset=dataset, fold=fold, config="NeuralNetFastAI_r2") == 0
            assert repo_collection.goes_where(dataset=dataset, fold=fold, config="NeuralNetFastAI_r1") == 1            

            predict_test = repo.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            predict_test_collection = repo_collection.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_test, predict_test_collection)

            predict_val = repo.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            predict_val_collection = repo_collection.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_val, predict_val_collection)
