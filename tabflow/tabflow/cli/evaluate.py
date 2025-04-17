from __future__ import annotations

import argparse
import logging

from autogluon.common.loaders import load_json
from tabflow.utils.utils import find_method_by_name
from tabflow.utils.s3_utils import download_from_s3
from tabflow.utils.logging_utils import setup_logging
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import ExperimentBatchRunner, AGModelBagExperiment, Experiment, YamlExperimentSerializer, AGExperiment
from tabrepo.benchmark.models.simple import SimpleLightGBM
from autogluon.tabular.models import *
from tabrepo.benchmark.models.ag import *

logger = setup_logging(level=logging.INFO)


def evaluate(
    experiment_name: str,
    context_name: str,
    tasks_s3_path: str,
    s3_bucket: str,
    methods_s3_path: str,
    load_predictions: bool = False,
    run_mode: str = "aws",
    raise_on_failure: bool = False,
    debug_mode: bool = False,
    s3_dataset_cache: str = None,
):
    # Load Context
    expname = experiment_name
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    repo: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=load_predictions)

    # Download methods and tasks to parse from S3
    methods_config_path = download_from_s3(s3_path=methods_s3_path, destination_path=None)
    methods_config: list[dict] = YamlExperimentSerializer.load_yaml(path=methods_config_path)

    tasks: list[dict] = load_json.load(path=tasks_s3_path, verbose=False)

    experiment_batch_runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=repo.task_metadata,
        mode=run_mode,  # We use AWS for TabFlow
        s3_bucket=s3_bucket,
        debug_mode=debug_mode,
        s3_dataset_cache=s3_dataset_cache,
    )

    logger.info(f"Downloaded Tasks to run are: {tasks}")
    for task in tasks:
        dataset = task["dataset"]
        tid = repo.dataset_to_tid(dataset)  # Solely for logging purposes
        fold = task["fold"]
        method_name = task["method_name"]
        method_kwargs = find_method_by_name(methods_config, method_name)
        method = YamlExperimentSerializer.parse_method(method_kwargs, globals())
        # This print is needed for task-wise log parsing
        print(f"Processing task: Dataset={dataset}, TID={tid}, Fold={fold}, Method={method_name}")
        logger.info(f"Method Dict: {method_kwargs}")
        logger.info(f"Method: {method}")
        results_lst: list[dict] = experiment_batch_runner.run(
            datasets=[dataset],
            folds=[fold],
            methods=[method],
            ignore_cache=ignore_cache,
            raise_on_failure=raise_on_failure,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=True, help="Name of the context")
    parser.add_argument('--tasks_s3_path', type=str, required=True, help="S3 path to batch of tasks JSON")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")
    parser.add_argument('--methods_s3_path', type=str, required=True, help="S3 path to methods config")
    parser.add_argument('--load_predictions', action='store_true', help="Load predictions from S3")
    parser.add_argument('--run_mode', type=str, default='aws', choices=['aws', 'local'], help="Run mode: aws or local")
    parser.add_argument('--s3_dataset_cache', type=str, required=False, default=None, help="S3 path for dataset cache")

    args = parser.parse_args()
    if args.s3_dataset_cache == "":
        args.s3_dataset_cache = None
        
    evaluate(
        experiment_name=args.experiment_name,
        context_name=args.context_name,
        tasks_s3_path=args.tasks_s3_path,
        s3_bucket=args.s3_bucket,
        methods_s3_path=args.methods_s3_path,
        load_predictions=args.load_predictions,
        run_mode=args.run_mode,
        s3_dataset_cache=args.s3_dataset_cache,
    )
