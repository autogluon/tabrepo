import boto3
import sagemaker
import yaml
import argparse
import json
import time

from datetime import datetime
from pathlib import Path
from tabrepo import EvaluationRepository
from utils import sanitize_job_name, check_s3_file_exists


DOCKER_IMAGE_ALIASES = {
    "mlflow-image": "097403188315.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo",
}


class TrainingJobResourceManager:
    def __init__(self, sagemaker_client, max_concurrent_jobs):
        self.sagemaker_client = sagemaker_client
        self.max_concurrent_jobs = max_concurrent_jobs

    def get_running_jobs_count(self):
        job_count = 0
        next_token = None
        while True:
            if next_token:
                response = self.sagemaker_client.list_training_jobs(StatusEquals='InProgress', MaxResults=100, NextToken=next_token)
            else:
                response = self.sagemaker_client.list_training_jobs(StatusEquals='InProgress', MaxResults=100)

            job_count += len(response['TrainingJobSummaries'])
            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break
        
        return job_count

    def wait_for_available_slot(self, poll_interval=30):
        while True:
            print("\nTraining Jobs: ", self.get_running_jobs_count())
            current_jobs = self.get_running_jobs_count()
            if current_jobs < (self.max_concurrent_jobs):
                return current_jobs
            print(f"Currently running {current_jobs}/{self.max_concurrent_jobs} jobs. Waiting...")
            time.sleep(poll_interval)


def launch_jobs(
        experiment_name: str = "tabflow-test-cache",
        context_name: str = "D244_F3_C1530_30", # 30 datasets. To run larger, set to "D244_F3_C1530_200"
        entry_point: str = "evaluate.py",
        source_dir: str = ".",
        instance_type: str = "ml.m6i.4xlarge",
        docker_image_uri: str = "mlflow-image",
        sagemaker_role: str = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145",
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        job_name: str = None,
        keep_alive_period_in_seconds: int = 3600,
        limit_runtime: int = 24 * 60 * 60,
        datasets: list = None,
        folds: list = None,
        methods_file: str = "methods.yaml",
        max_concurrent_jobs: int = 30,
        s3_bucket: str = "prateek-ag",
        add_timestamp: bool = False,
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
        job_name: Name for the training job
        keep_alive_period_in_seconds: Idle time before terminating the instance 
        limit_runtime: Maximum running time in seconds
        datasets: List of datasets to evaluate
        folds: List of folds to evaluate
        methods_file: Path to the YAML file containing methods
        max_concurrent_jobs: Maximum number of concurrent jobs, based on account limit
        S3 bucket: S3 bucket to store the results
        add_timestamp: Whether to add a timestamp to the experiment name
    """
    
    if add_timestamp:
        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")[:-3]
        experiment_name = f"{experiment_name}-{timestamp}"

    # Create a SageMaker client session
    boto_session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    sagemaker_client = boto_session.client('sagemaker')
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    s3_client = boto_session.client('s3')

    # Initialize the resource manager
    resource_manager = TrainingJobResourceManager(sagemaker_client=sagemaker_client, max_concurrent_jobs=max_concurrent_jobs)

    # Load methods from YAML file
    with open(methods_file, 'r') as file:
        methods_data = yaml.safe_load(file)

    methods = [(method["name"], method["wrapper_class"], method["fit_kwargs"]) for method in methods_data["methods"]]

    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    
    if "run_all" in datasets:
        datasets = repo_og.datasets()
    else:
        datasets = datasets

    if -1 in folds:
        folds = repo_og.folds
    else:
        folds = folds

    total_jobs = len(datasets) * len(folds) * len(methods)
    total_launched_jobs = 0

    print(f"Preparing to launch {total_jobs} jobs with max concurrency of {max_concurrent_jobs}")
    print(f"Instance keep-alive period set to {keep_alive_period_in_seconds} seconds to enable warm-starts")
    
    try:
        for dataset in datasets:
            for fold in folds:
                for method in methods:

                    method_name, wrapper_class, fit_kwargs = method
                    cache_name = f"{experiment_name}/data/tasks/{repo_og.dataset_to_tid(dataset)}/{fold}/{method_name}/results.pkl"

                    if check_s3_file_exists(s3_client=s3_client, bucket=s3_bucket, cache_name=cache_name):
                        print(f"Cache exists for {method_name} on dataset {dataset} fold {fold}. Skipping job launch.")
                        print(f"Cache path: s3://{s3_bucket}/{cache_name}\n")
                        continue

                    current_jobs = resource_manager.wait_for_available_slot()
                    print(f"\nSlot available. Currently running {current_jobs}/{max_concurrent_jobs} jobs")

                    # Create a unique job name
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    base_name = f"{dataset[:4]}-f{fold}-{method_name[:4]}-{timestamp}"
                    job_name = sanitize_job_name(base_name)


                    if docker_image_uri in DOCKER_IMAGE_ALIASES:
                        print(f"Expanding docker_image_uri alias '{docker_image_uri}' -> '{DOCKER_IMAGE_ALIASES[docker_image_uri]}'")
                        docker_image_uri = DOCKER_IMAGE_ALIASES[docker_image_uri]

                    # Update hyperparameters for this job
                    job_hyperparameters = hyperparameters.copy() if hyperparameters else {}
                    job_hyperparameters.update({
                        "experiment_name": experiment_name,
                        "context_name": context_name,
                        "dataset": dataset,
                        "fold": fold,   # NOTE: Can be a 'str' as well, refer to Estimators in SM docs
                        "method_name": method_name,
                        "wrapper_class": wrapper_class,
                        "fit_kwargs": f"'{json.dumps(fit_kwargs)}'",
                        "s3_bucket": s3_bucket,
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
                    )

                    # Launch the training job
                    estimator.fit(wait=False, job_name=job_name)
                    total_launched_jobs += 1
                    print(f"Launched job {total_launched_jobs} out of a total of {total_jobs}: {job_name}")
                    # print(f"Launched training job: {estimator.latest_training_job.name}")
    except Exception as e:
        print(f"Error launching jobs: {e}")
        raise


def main():
    """Entrypoint for CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=False, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=False, help="Name of the context")
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--methods_file', type=str, required=True, help="Path to the YAML file containing methods")
    parser.add_argument('--s3_bucket', type=str, required=False, help="S3 bucket for the experiment")
    parser.add_argument('--add_timestamp', action='store_true', help="Add timestamp to the experiment name")

    args = parser.parse_args()

    launch_jobs(
        datasets=args.datasets,
        folds=args.folds,
        methods_file=args.methods_file,
        s3_bucket=args.s3_bucket,
    )


if __name__ == "__main__":
    main()