from __future__ import annotations

import time

import pandas as pd

from tabrepo import EvaluationRepository
from tabrepo.utils.pickle_utils import fetch_all_pickles
from tabrepo.benchmark.result import ExperimentResults

from .load_artifacts import load_all_artifacts


def generate_repo(experiment_path: str, task_metadata: pd.DataFrame) -> EvaluationRepository:
    file_paths = fetch_all_pickles(dir_path=experiment_path)
    file_paths = sorted([str(f) for f in file_paths])
    print(len(file_paths))

    return generate_repo_from_paths(result_paths=file_paths, task_metadata=task_metadata)


def generate_repo_from_paths(result_paths: list[str], task_metadata: pd.DataFrame, engine: str = "ray") -> EvaluationRepository:
    results_lst = load_all_artifacts(file_paths=result_paths, engine=engine)

    tids = set(list(task_metadata["tid"].unique()))

    results_lst = [r for r in results_lst if r.result["task_metadata"]["tid"] in tids]

    exp_results = ExperimentResults(task_metadata=task_metadata)

    ts = time.time()
    repo: EvaluationRepository = exp_results.repo_from_results(results_lst=results_lst)
    te = time.time()
    print(f"{te-ts:.2f}s ExperimentResults.repo_from_results")
    return repo