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
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--method_name', type=str, required=True, help="Name of the method")
    parser.add_argument('--method', type=str, required=True, help="Method to evaluate, dict as JSON string")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")

    args = parser.parse_args()

    # Load Context
    context_name = args.context_name
    expname = args.experiment_name
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    #TODO: Download the repo without pred-proba
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    datasets = args.datasets
    folds = args.folds

    method_dict = json.loads(args.method)
    print(f"Method dict: {method_dict}")
    methods = parse_method(method_dict, globals())
    print("\nMethods: ", methods)

    repo: EvaluationRepository = ExperimentBatchRunner(expname=expname, task_metadata=repo_og.task_metadata).run(
    datasets=datasets, 
    folds=folds,
    methods=[methods], 
    ignore_cache=ignore_cache,
    mode="aws",
    s3_bucket=args.s3_bucket,
    )