from __future__ import annotations

import argparse
import yaml
import json
import logging


from tabflow.utils.utils import parse_method, find_method_by_name
from tabflow.utils.s3_utils import download_from_s3
from tabflow.utils.logging_utils import setup_logging
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import ExperimentBatchRunner, AGModelBagExperiment, Experiment
from tabrepo.benchmark.models.simple import SimpleLightGBM
from autogluon.tabular.models import *
from tabrepo.benchmark.models.ag import *

logger = setup_logging(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=True, help="Name of the context")
    parser.add_argument('--tasks_s3_path', type=str, required=True, help="S3 path to batch of tasks JSON")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")
    parser.add_argument('--methods_s3_path', type=str, required=True, help="S3 path to methods config")
    parser.add_argument('--load_predictions', action='store_true', help="Load predictions from S3")
    parser.add_argument('--run_mode', type=str, default='aws', choices=['aws', 'local'], help="Run mode: aws or local")

    args = parser.parse_args()

    # Load Context
    context_name = args.context_name
    expname = args.experiment_name
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=args.load_predictions)

    # Download methods and tasks to parse from S3
    methods_config_path = download_from_s3(s3_path=args.methods_s3_path, destination_path=None)
    with open(methods_config_path, 'r') as f:
        methods_config = yaml.safe_load(f)

    tasks_path = download_from_s3(s3_path=args.tasks_s3_path, destination_path=None)
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    logger.info(f"Downloaded Tasks to run are: {tasks}")
    for task in tasks:
        dataset = task["dataset"]
        fold = task["fold"]
        method_name = task["method_name"]
        method = find_method_by_name(methods_config, method_name)
        if method is None:
            logger.info(f"Method {method_name} not found in methods config")
            continue
        tid = repo_og.dataset_to_tid(dataset)   # Solely for logging purposes
        # This print is needed for task-wise log parsing
        print(f"Processing task: Dataset={dataset}, TID={tid}, Fold={fold}, Method={method_name}")
        logger.info(f"Method Dict: {method}")
        methods = parse_method(method, globals())
        logger.info(f"Methods: {methods}")
        results_lst: list[dict] = ExperimentBatchRunner(expname=expname, task_metadata=repo_og.task_metadata).run(
            datasets=[dataset],
            folds=[fold],
            methods=[methods],
            ignore_cache=ignore_cache,
            mode=args.run_mode,  # We use AWS for TabFlow
            s3_bucket=args.s3_bucket,
        )
