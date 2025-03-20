from __future__ import annotations

import argparse
import yaml
import pandas as pd
import json


from tabflow.utils import parse_method
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import ExperimentBatchRunner, AGModelBagExperiment, Experiment
from tabrepo.benchmark.models.simple import SimpleLightGBM
from autogluon.tabular.models import *
from tabrepo.benchmark.models.ag import *

# from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
# from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGWrapper
# from tabrepo.benchmark.models.ag import RealMLPModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=True, help="Name of the context")
    parser.add_argument('--tasks', type=str, required=True, help="List of tasks to evaluate")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")

    args = parser.parse_args()

    # Load Context
    context_name = args.context_name
    expname = args.experiment_name
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    #FIXME: Cache combined with load predictions is buggy
    # repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=False)

    tasks_str = args.tasks
    if tasks_str.startswith("'") and tasks_str.endswith("'"):
        tasks_str = tasks_str[1:-1]

    tasks = json.loads(args.tasks)
    for task in tasks:
        dataset = task["dataset"]
        fold = task["fold"]
        method_name = task["method_name"]
        method = task["method"]
        tid = repo_og.dataset_to_tid(dataset)   # Solely for logging purposes
        print(f"Processing task: Dataset={dataset}, TID={tid}, Fold={fold}, Method={method_name}")
        print("\nMethod Dict: ", method)
        methods = parse_method(method, globals())
        print("\nMethods: ", methods)
        # Repo -> results list
        # Launch - 8 ExperimentBatchRunner one way to batch or do a for-loop
        repo: EvaluationRepository = ExperimentBatchRunner(expname=expname, task_metadata=repo_og.task_metadata).run(
        datasets=[dataset], 
        folds=[fold],
        methods=[methods], 
        ignore_cache=ignore_cache,
        mode="aws",
        s3_bucket=args.s3_bucket,
        )

