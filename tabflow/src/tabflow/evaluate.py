from __future__ import annotations

import argparse
import yaml
import pandas as pd
import json
import boto3
import os


from tabflow.utils import parse_method
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import ExperimentBatchRunner, AGModelBagExperiment, Experiment
from tabrepo.benchmark.models.simple import SimpleLightGBM
from autogluon.tabular.models import *
from tabrepo.benchmark.models.ag import *

# from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
# from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGWrapper
# from tabrepo.benchmark.models.ag import RealMLPModel

def download_from_s3(s3_path):
    """Download file from S3 to local path"""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    
    local_path = os.path.basename(key)
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, local_path)
    return local_path

def find_method_by_name(methods_config, method_name):
    """Find a method configuration by name in the methods configuration"""
    if "methods" in methods_config:
        for method in methods_config["methods"]:
            if method.get("name") == method_name:
                # Return copy to ensure next method if same can be popped as well
                return method.copy()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=True, help="Name of the context")
    # parser.add_argument('--tasks', type=str, required=True, help="List of tasks to evaluate")
    parser.add_argument('--tasks_s3_path', type=str, required=True, help="S3 path to batch of tasks JSON")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")
    parser.add_argument('--methods_s3_path', type=str, required=True, help="S3 path to methods config")

    args = parser.parse_args()

    # Load Context
    context_name = args.context_name
    expname = args.experiment_name
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    #FIXME: Cache combined with load predictions is buggy
    # repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=False)

    # Download methods and tasks to parse from S3
    methods_config_path = download_from_s3(args.methods_s3_path)
    with open(methods_config_path, 'r') as f:
        methods_config = yaml.safe_load(f)

    tasks_path = download_from_s3(args.tasks_s3_path)
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    # DEBUG PRINT
    # print("\nMethods Config is: ", methods_config)

    # tasks_str = args.tasks
    # if tasks_str.startswith("'") and tasks_str.endswith("'"):
    #     tasks_str = tasks_str[1:-1]

    # tasks = json.loads(args.tasks)
    print(f"Downloaded Tasks to run are: {tasks}")
    for task in tasks:
        dataset = task["dataset"]
        fold = task["fold"]
        method_name = task["method_name"]
        # method = task["method"]
        method = find_method_by_name(methods_config, method_name)
        if method is None:
            print(f"Method {method_name} not found in methods config")
            continue
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

