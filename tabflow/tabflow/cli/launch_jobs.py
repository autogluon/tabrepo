import boto3
import sagemaker
import argparse
import logging
import uuid
import math

from botocore.config import Config
from datetime import datetime
from pathlib import Path
from tabrepo import EvaluationRepository
from tabflow.core.resource_manager import TrainingJobResourceManager
from tabflow.utils.utils import sanitize_job_name, yaml_to_methods, create_batch
from tabflow.utils.s3_utils import check_s3_file_exists, upload_methods_config, upload_tasks_json
from tabflow.utils.logging_utils import setup_logging
from tabflow.utils.constants import DOCKER_IMAGE_ALIASES

logger = setup_logging(level=logging.ERROR)

def launch_jobs(
        experiment_name: str,
        methods_file: str,
        s3_bucket: str,
        docker_image_uri: str,
        sagemaker_role: str,
        context_name: str = "D244_F3_C1530_30", # 30 datasets. To run larger, set to "D244_F3_C1530_200"
        entry_point: str = "evaluate.py",
        source_dir: str = str(Path(__file__).parent),
        instance_type: str = "ml.m6i.4xlarge",
        keep_alive_period_in_seconds: int = 3600,
        limit_runtime: int = 24 * 60 * 60,
        max_concurrent_jobs: int = 30,
        max_retry_attempts: int = 20,
        batch_size: int = 1,
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        datasets: list = None,
        folds: list = None,
        add_timestamp: bool = False,
        wait: bool = True,
        s3_dataset_cache: str = None,
) -> None:
    """
    Launch multiple SageMaker training jobs.

    Args:
        experiment_name: Name of the experiment
        context_name: Name of the TabRepo context
        entry_point: The Python script to run in sagemaker training job
        source_dir: Directory containing the training code (here the entry point)
        instance_type: SageMaker instance type
        docker_image_uri: Docker image to use URI or alias in constants.py
        sagemaker_role: AWS IAM role for SageMaker
        aws_profile: AWS profile name
        hyperparameters: Dictionary of hyperparameters to pass to the training script
        keep_alive_period_in_seconds: Idle time before terminating the instance 
        limit_runtime: Maximum running time in seconds
        datasets: List of datasets to evaluate
        folds: List of folds to evaluate
        methods_file: Path to the YAML file containing methods
        max_concurrent_jobs: Maximum number of concurrent jobs, based on account limit
        S3 bucket: S3 bucket to store the results
        add_timestamp: Whether to add a timestamp to the experiment name
        wait: Whether to wait for all jobs to complete (no-wait from CLI)
        batch_size: Number of models to batch for each task
        s3_dataset_cache: Full S3 URI for OpenML dataset cache (format: s3://bucket/prefix), note that after prefix
        the following will be appended to the path - tasks/{task_id}/org/openml/www/tasks/{task_id}, where the xml and arff is expected to be situated
    """
    
    if add_timestamp:
        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")[:-3]
        experiment_name = f"{experiment_name}-{timestamp}"

    # Create boto3 session
    boto_session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    # Create SageMaker session + retry config
    retry_config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={'max_attempts':max_retry_attempts,
                'mode':'adaptive',
                }
    )
    sagemaker_client = boto_session.client('sagemaker', config=retry_config)
    sagemaker_session = sagemaker.Session(boto_session=boto_session, sagemaker_client=sagemaker_client)
    # Create S3 client
    s3_client = boto_session.client('s3', config=retry_config)
    
    # Initialize the resource manager
    resource_manager = TrainingJobResourceManager(sagemaker_client=sagemaker_client, max_concurrent_jobs=max_concurrent_jobs)

    methods_s3_key = f"{experiment_name}/config/methods_config.yaml"
    methods_s3_path = upload_methods_config(s3_client, methods_file, s3_bucket, methods_s3_key)

    methods = yaml_to_methods(methods_file=methods_file)
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, load_predictions=False)
    
    if "run_all" in datasets:
        datasets = repo_og.datasets()
    else:
        datasets = datasets

    if -1 in folds:
        folds = repo_og.folds
    else:
        folds = folds

    tasks = [(dataset, fold, method) for dataset in datasets for fold in folds for method in methods]
    total_jobs = math.ceil(len(tasks)/batch_size)
    resource_manager.total_jobs = total_jobs

    logger.info(f"Preparing to launch {total_jobs} jobs with batch size of {batch_size} and max concurrency of {max_concurrent_jobs}")
    logger.info(f"Instance keep-alive period set to {keep_alive_period_in_seconds} seconds to enable warm-starts")
    
    try:
        for task_batch in create_batch(tasks, batch_size):
            uncached_tasks = []
            for dataset, fold, method in task_batch:
                    
                method_name = method['name']
                cache_path = f"{experiment_name}/data/{method_name}/{repo_og.dataset_to_tid(dataset)}/{fold}"
                cache_name = f"{experiment_name}/data/{method_name}/{repo_og.dataset_to_tid(dataset)}/{fold}/results.pkl"

                # Change this check based on literals name_first or task_first
                if check_s3_file_exists(s3_client=s3_client, bucket=s3_bucket, cache_name=cache_name):
                    logger.info(f"Cache exists for {method_name} on dataset {dataset} fold {fold}. Skipping job launch.")
                    logger.info(f"Cache path: s3://{s3_bucket}/{cache_path}\n")
                else:
                    uncached_tasks.append((dataset, fold, method))

            if not uncached_tasks:
                logger.info(f"All tasks in batch are cached. Skipping job launch.")
                continue


            if docker_image_uri in DOCKER_IMAGE_ALIASES:
                logger.info(f"Expanding docker_image_uri alias '{docker_image_uri}' -> '{DOCKER_IMAGE_ALIASES[docker_image_uri]}'")
                docker_image_uri = DOCKER_IMAGE_ALIASES[docker_image_uri]

            resource_manager.wait_for_available_slot(s3_client=s3_client, s3_bucket=s3_bucket)

            # Create a unique job name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4().int)[:19]
            base_name = f"{dataset[:4]}-f{fold}-{method_name[:4]}-{timestamp}-{unique_id}"
            job_name = sanitize_job_name(base_name)

            tasks_json = []
            for dataset, fold, method in uncached_tasks:
                tasks_json.append({
                    "dataset": dataset,
                    "fold": fold,   # NOTE: Can be a 'str' as well, refer to Estimators in SM docs
                    "method_name": method["name"],
                })

            # Unique s3 key for a task
            tasks_s3_key = f"{experiment_name}/config/tasks/{job_name}_tasks_batch_{batch_size}.json"
            tasks_s3_path = upload_tasks_json(s3_client, tasks_json, s3_bucket, tasks_s3_key)
            
            # Update hyperparameters for this job
            job_hyperparameters = hyperparameters.copy() if hyperparameters else {}
            job_hyperparameters.update({
                "experiment_name": experiment_name,
                "context_name": context_name,
                "tasks_s3_path": tasks_s3_path,
                "s3_bucket": s3_bucket,
                "methods_s3_path": methods_s3_path,
            })
            if s3_dataset_cache is not None:
                job_hyperparameters["s3_dataset_cache"] = s3_dataset_cache

            # Create the estimator
            estimator = sagemaker.estimator.Estimator(
                entry_point=entry_point,
                source_dir=source_dir,
                image_uri=docker_image_uri,
                role=sagemaker_role,
                instance_count=1,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=job_hyperparameters,
                keep_alive_period_in_seconds=keep_alive_period_in_seconds,
                max_run=limit_runtime,
                disable_profiler=True,  # Prevent debug profiler from running
                # output_path=f"s3://{s3_bucket}/{experiment_name}/data/output"  #TBD: What artifact to save here?
            )

            # Launch the training job
            estimator.fit(wait=False, job_name=job_name)
            resource_manager.add_job(job_name=job_name, cache_path=cache_path)
        
        if wait:
            resource_manager.wait_for_all_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
    except Exception as e:
        logger.error(f"Error launching jobs: {e}", exc_info=True)
        raise


def main():
    """Entrypoint for Launching sagemaker jobs using CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, default="D244_F3_C1530_30", help="Name of the context")
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--methods_file', type=str, required=True, help="Path to the YAML file containing methods")
    parser.add_argument('--max_concurrent_jobs', type=int, default=50,
                        help="Maximum number of concurrent jobs, based on account limit")
    parser.add_argument('--docker_image_uri', type=str, required=True, help="Docker image URI or alias in constants.py")
    parser.add_argument('--instance_type', type=str, default="ml.m6i.4xlarge", help="SageMaker instance type")
    parser.add_argument('--sagemaker_role', type=str, required=True, help="AWS IAM role ARN for SageMaker")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")
    parser.add_argument('--add_timestamp', action='store_true', help="Add timestamp to the experiment name")
    parser.add_argument('--no-wait', dest='wait', action='store_false', help="Skip waiting for jobs to complete")
    parser.set_defaults(wait=True)
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for tasks")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('--s3_dataset_cache', type=str, required=False, default=None,
                        help="S3 URI for OpenML dataset cache (format: s3://bucket/prefix)")
    
    args = parser.parse_args()

    launch_jobs(
        experiment_name=args.experiment_name,
        context_name=args.context_name,
        datasets=args.datasets,
        folds=args.folds,
        methods_file=args.methods_file,
        max_concurrent_jobs=args.max_concurrent_jobs,
        docker_image_uri=args.docker_image_uri,
        instance_type=args.instance_type,
        sagemaker_role=args.sagemaker_role,
        s3_bucket=args.s3_bucket,
        wait=args.wait,
        batch_size=args.batch_size,
        s3_dataset_cache=args.s3_dataset_cache,
    )


if __name__ == "__main__":
    main()
