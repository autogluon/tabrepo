from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tabrepo import EvaluationRepository, load_repository


# closest tasks from numpy
def distance_tasks_from_numpy(evaluations: np.array, evaluations_new_task: np.array) -> np.array:
    """
    evaluations: (n_models, n_datasets)
    model_evaluations: (n_models,): can contains nans
    """
    # filter the models with missing values from our list
    evaluations_new_task = pd.DataFrame(evaluations_new_task).rank(pct=True).dropna()

    # compute the rank of evaluations after filtering the model with missing values
    rank_evaluations = pd.DataFrame(evaluations).loc[evaluations_new_task.index].rank(pct=True)
    # for model missing evaluations, assign the largest rank
    rank_evaluations = rank_evaluations.fillna(1)
    rank_evaluations = rank_evaluations.values

    # compute all squared distances
    distances = np.sqrt(np.square(rank_evaluations - evaluations_new_task.values).mean(axis=0))

    return distances


def distance_tasks_from_repo(repo: EvaluationRepository, model_metrics: Dict[str, float]) -> Dict[Tuple[str, int], float]:
    """
    :param model_metrics:
    :return: a dictionary from  dataset and fold to distance with the ranks from `model_metrics` provided.
    The models in `model_scores` should be present in the repository and ranking are used before computing distances.
    """
    # makes sure all models are present in the repository,
    # alternatively we could subselect the one available and print a warning
    selected_models = list(model_metrics.keys())
    repo_models = set(repo.configs())
    for model in selected_models:
        assert model in repo_models, f"{model} passed on model_metrics but not found in repo"

    # gets metrics for the selected models
    df = repo.metrics(configs=selected_models)

    # pivot metrics to dataframe with shape (n_models, n_datasets)
    df_metrics = df.reset_index().pivot_table(index="framework", columns=["dataset", "fold"], values="metric_error")
    df_metrics = df_metrics.loc[selected_models]

    distances_list = distance_tasks_from_numpy(
        evaluations=df_metrics.values,
        evaluations_new_task=model_metrics.values(),
    )
    return {
        (dataset, fold): v for (dataset, fold), v in zip(df_metrics.columns, distances_list)
    }



if __name__ == '__main__':
    # Example to show how to use
    repo = load_repository("D244_F3_C1530_200")
    model_metrics_new_task = repo.metrics() # TODO
    distances = distance_tasks_from_repo(repo, model_metrics=model_metrics_new_task)