import boto3
import sagemaker
import argparse
import json
import time
import logging
import uuid
import math

from botocore.config import Config
from itertools import islice
from datetime import datetime
from pathlib import Path
from tabrepo import EvaluationRepository
from tabflow.utils import sanitize_job_name, check_s3_file_exists, yaml_to_methods, save_training_job_logs, upload_methods_config, upload_tasks_json

logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('sagemaker').setLevel(logging.ERROR)
logging.getLogger('boto3').setLevel(logging.ERROR)


DOCKER_IMAGE_ALIASES = {
    "mlflow-image": "{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO}:{IMAGE_TAG}",
}


class TrainingJobResourceManager:
    def __init__(self, sagemaker_client, max_concurrent_jobs):
        self.sagemaker_client = sagemaker_client
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_names = []
        self.job_cache_paths = {} # Job Name -> Training Log

    def add_job(self, job_name, cache_path):
        self.job_names.append(job_name)
        self.job_cache_paths[job_name] = cache_path

    def remove_completed_jobs(self, s3_client, s3_bucket):
        completed_jobs = []
        for job_name in self.job_names:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)    #FIXME:Throttling will happen here if Queue is too big
            job_status = response['TrainingJobStatus']
            if job_status in ['Completed', 'Failed', 'Stopped']:
                save_training_job_logs(
                    self.sagemaker_client, 
                    s3_client, 
                    job_name, 
                    s3_bucket, 
                    self.job_cache_paths[job_name]
                )
                completed_jobs.append(job_name)
        for job_name in completed_jobs:
            self.job_names.remove(job_name)

    def wait_for_available_slot(self, s3_client, s3_bucket, poll_interval=10):
        while True:
            if len(self.job_names) < self.max_concurrent_jobs:
                return len(self.job_names)
            self.remove_completed_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
            print(f"Currently running {len(self.job_names)} out of {self.max_concurrent_jobs} concurrent jobs. Waiting...")
            time.sleep(poll_interval)

    def wait_for_all_jobs(self, s3_client, s3_bucket, poll_interval=10):
        # Wait for a non-zero value
        while self.job_names:
            self.remove_completed_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
            print(f"Waiting for {len(self.job_names)} jobs to complete...")
            time.sleep(poll_interval)


def create_batch(tasks, batch_size):
    """Convert all tasks into batches"""
    it = iter(tasks)
    for batch in iter(lambda: tuple(islice(it, batch_size)), ()):
        yield batch


def launch_jobs(
        experiment_name: str = "tabflow-test-cache",
        context_name: str = "D244_F3_C1530_30", # 30 datasets. To run larger, set to "D244_F3_C1530_200"
        entry_point: str = "evaluate.py",
        source_dir: str = str(Path(__file__).parent),
        instance_type: str = "ml.m6i.4xlarge",
        docker_image_uri: str = "mlflow-image",
        sagemaker_role: str = "arn:aws:iam::{ACCOUNT_ID}:role/service-role/{ROLE}",
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        keep_alive_period_in_seconds: int = 3600,
        limit_runtime: int = 24 * 60 * 60,
        datasets: list = None,
        folds: list = None,
        methods_file: str = "methods.yaml",
        max_concurrent_jobs: int = 30,
        max_retry_attempts: int = 20,
        s3_bucket: str = "test-bucket",
        add_timestamp: bool = False,
        wait: bool = False,
        batch_size: int = 2,
) -> None:
    """
    Launch multiple SageMaker training jobs.

    Args:
        experiment_name: Name of the experiment
        entry_point: The Python script to run
        source_dir: Directory containing the training code
        instance_type: SageMaker instance type
        docker_image_uri: Docker image to use
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
        wait: Whether to wait for all jobs to complete
        batch_size: Number of models to batch for each task
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

    # repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
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
    # total_jobs = len(datasets) * len(folds) * len(methods)
    total_jobs = math.ceil(len(tasks)/batch_size)
    total_launched_jobs = 0

    print(f"Preparing to launch {total_jobs} jobs with batch size of {batch_size} and max concurrency of {max_concurrent_jobs}")
    print(f"Instance keep-alive period set to {keep_alive_period_in_seconds} seconds to enable warm-starts")
    
    try:
        for task_batch in create_batch(tasks, batch_size):
            uncached_tasks = []
            for dataset, fold, method in task_batch:
                    
                method_name = method['name']
                cache_path = f"{experiment_name}/data/{method_name}/{repo_og.dataset_to_tid(dataset)}/{fold}"
                cache_name = f"{experiment_name}/data/{method_name}/{repo_og.dataset_to_tid(dataset)}/{fold}/results.pkl"

                # Change this check based on literals name_first or task_first
                if check_s3_file_exists(s3_client=s3_client, bucket=s3_bucket, cache_name=cache_name):
                    print(f"Cache exists for {method_name} on dataset {dataset} fold {fold}. Skipping job launch.")
                    print(f"Cache path: s3://{s3_bucket}/{cache_path}\n")
                else:
                    uncached_tasks.append((dataset, fold, method))

            if not uncached_tasks:
                print(f"All tasks in batch are cached. Skipping job launch.")
                continue

            current_jobs = resource_manager.wait_for_available_slot(s3_client=s3_client, s3_bucket=s3_bucket)
            print(f"\nSlot available. Currently running {current_jobs}/{max_concurrent_jobs} jobs")

            # Create a unique job name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4().int)[:19]
            base_name = f"{dataset[:4]}-f{fold}-{method_name[:4]}-{timestamp}-{unique_id}"
            job_name = sanitize_job_name(base_name)

            if docker_image_uri in DOCKER_IMAGE_ALIASES:
                print(f"Expanding docker_image_uri alias '{docker_image_uri}' -> '{DOCKER_IMAGE_ALIASES[docker_image_uri]}'")
                docker_image_uri = DOCKER_IMAGE_ALIASES[docker_image_uri]

            tasks_json = []
            for dataset, fold, method in uncached_tasks:
                tasks_json.append({
                    "dataset": dataset,
                    "fold": fold,   # NOTE: Can be a 'str' as well, refer to Estimators in SM docs
                    "method_name": method["name"],
                    # "method": method,   # We do not need this if we read yaml in evaluate.py
                })

            # Unique s3 key for a task
            tasks_s3_key = f"{experiment_name}/config/tasks/{job_name}_tasks_batch_{batch_size}.json"
            tasks_s3_path = upload_tasks_json(s3_client, tasks_json, s3_bucket, tasks_s3_key)
            
            # Update hyperparameters for this job
            job_hyperparameters = hyperparameters.copy() if hyperparameters else {}
            job_hyperparameters.update({
                "experiment_name": experiment_name,
                "context_name": context_name,
                # "tasks": f"'{json.dumps(tasks_json)}'",
                "tasks_s3_path": tasks_s3_path,
                "s3_bucket": s3_bucket,
                "methods_s3_path": methods_s3_path,
            })

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
                # output_path=f"s3://{s3_bucket}/{experiment_name}/data/output"  # Add output path configuration to ensure task logs are accessible
            )

            # Launch the training job
            estimator.fit(wait=False, job_name=job_name)
            resource_manager.add_job(job_name=job_name, cache_path=cache_path)
            total_launched_jobs += 1
            print(f"Launched job {total_launched_jobs} out of a total of {total_jobs} jobs: {job_name}\n")
        
        if wait:
            resource_manager.wait_for_all_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
    except Exception as e:
        print(f"Error launching jobs: {e}")
        raise


def main():
    """Entrypoint for CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default="tabflow-test-cache", help="Name of the experiment")
    parser.add_argument('--context_name', type=str, default="D244_F3_C1530_30", help="Name of the context")
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--methods_file', type=str, required=True, help="Path to the YAML file containing methods")
    parser.add_argument('--max_concurrent_jobs', type=int, default=50,
                        help="Maximum number of concurrent jobs, based on account limit")
    parser.add_argument('--s3_bucket', type=str, default="test-bucket", help="S3 bucket for the experiment")
    parser.add_argument('--add_timestamp', action='store_true', help="Add timestamp to the experiment name")
    parser.add_argument('--wait', action='store_true', help="Wait for all jobs to complete")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for tasks")

    args = parser.parse_args()

    launch_jobs(
        experiment_name=args.experiment_name,
        context_name=args.context_name,
        datasets=args.datasets,
        folds=args.folds,
        methods_file=args.methods_file,
        max_concurrent_jobs=args.max_concurrent_jobs,
        s3_bucket=args.s3_bucket,
        wait=args.wait,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

