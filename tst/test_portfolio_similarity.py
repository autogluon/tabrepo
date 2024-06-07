import numpy as np

from tabrepo.portfolio.similarity import distance_tasks_from_numpy

n_models = 3
n_datasets = 5
evaluations = np.arange(n_models * n_datasets).reshape(n_models, n_datasets)
evaluations[1] = evaluations[1][::-1]
evaluations = evaluations / evaluations.max(axis=1, keepdims=True)


def test_distance_tasks():
    # absolute values does not matter since we apply rank normalization after
    for dataset in range(n_datasets):
        # we pick as evaluation the model values of `dataset`
        # scale does not matter since we apply rank normalization after
        evaluations_new_task = evaluations[:, dataset] * 10
        distances = distance_tasks_from_numpy(evaluations, evaluations_new_task)
        assert len(distances) == n_datasets

        # we should have the distance be zero for the dataset we chose since it appears in our evaluations
        assert distances[dataset] == 0


def test_distance_tasks_nans_in_query():
    # test that we get the expected behavior when some models from the new task are missing values
    for dataset in range(n_datasets):
        evaluations_new_task = evaluations[:, dataset] * 10
        evaluations_new_task[-1] = np.nan
        distances = distance_tasks_from_numpy(evaluations, evaluations_new_task)
        assert len(distances) == n_datasets

        # we should have the distance be zero for the dataset we chose since it appears in our evaluations
        assert distances[dataset] == 0


def test_distance_tasks_nans_in_evaluations():
    evaluations_with_missing_value = evaluations.copy()
    # set the value to be missing for the first model on the second dataset
    evaluations_with_missing_value[0][2] = np.nan

    # test that we get the expected behavior when some models from the evaluations are missing values
    for dataset in range(n_datasets):
        if dataset != 2:
            evaluations_new_task = evaluations[:, dataset] * 10
            distances = distance_tasks_from_numpy(evaluations_with_missing_value, evaluations_new_task)
            assert len(distances) == n_datasets

            # we should have the distance be zero for the dataset we chose since it appears in our evaluations
            assert distances[dataset] == 0
        else:
            evaluations_new_task = evaluations[:, dataset] * 10
            evaluations_new_task[0] = np.nan
            distances = distance_tasks_from_numpy(evaluations_with_missing_value, evaluations_new_task)
            assert len(distances) == n_datasets

            # we should have the distance be zero for the dataset we chose since it appears in our evaluations
            assert distances[dataset] == 0
