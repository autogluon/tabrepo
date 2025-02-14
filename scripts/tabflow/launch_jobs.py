import boto3
import sagemaker
import re
import yaml
import argparse
import json
import time

from datetime import datetime
from pathlib import Path
from tabrepo import EvaluationRepository


DOCKER_IMAGE_ALIASES = {
    "mlflow-image": "ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo",
}


class TrainingJobResourceManager:
    def __init__(self, sagemaker_client, max_concurrent_jobs):
        self.sagemaker_client = sagemaker_client
        self.max_concurrent_jobs = max_concurrent_jobs

    def get_running_jobs_count(self):
        response = self.sagemaker_client.list_training_jobs(StatusEquals='InProgress', MaxResults=100)
        return len(response['TrainingJobSummaries'])

    def wait_for_available_slot(self, poll_interval=30):
        while True:
            print("\nTraining Jobs: ", self.get_running_jobs_count())
            current_jobs = self.get_running_jobs_count()
            if current_jobs < (self.max_concurrent_jobs):
                return current_jobs
            print(f"Currently running {current_jobs}/{self.max_concurrent_jobs} jobs. Waiting...")
            time.sleep(poll_interval)



def sanitize_job_name(name: str) -> str:
    """
    Sanitize the job name to meet SageMaker requirements:
    - Must be 1-63 characters long
    - Must use only alphanumeric characters and hyphens
    - Must start with a letter or number
    - Must not end with a hyphen
    """
    # Replace invalid characters with hyphens
    name = re.sub('[^a-zA-Z0-9-]', '-', name)
    # Remove consecutive hyphens
    name = re.sub('-+', '-', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    # Ensure it starts with a letter or number
    if not name[0].isalnum():
        name = 'j-' + name
    # Truncate to 63 characters
    return name[:63]


def launch_jobs(
        experiment_name: str = "tabflow",
        context_name: str = "D244_F3_C1530_30", # 30 datasets. To run larger, set to "D244_F3_C1530_200"
        entry_point: str = "evaluate.py",
        source_dir: str = ".",
        instance_type: str = "ml.m6i.4xlarge",
        docker_image_uri: str = "mlflow-image",
        sagemaker_role: str = "arn:aws:iam::ACCOUNT_ID:role/service-role/AmazonSageMakerRole",
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        job_name: str = None,
        keep_alive_period_in_seconds: int = 3600,
        limit_runtime: int = 24 * 60 * 60,
        datasets: list = None,
        folds: list = None,
        methods_file: str = "methods.yaml",
        max_concurrent_jobs: int = 30,
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
    """
    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")[:-3]
    experiment_name = f"{experiment_name}-{timestamp}"

    # Create a SageMaker client session
    boto_session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    sagemaker_client = boto_session.client('sagemaker')
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

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

                    current_jobs = resource_manager.wait_for_available_slot()
                    print(f"\nSlot available. Currently running {current_jobs}/{max_concurrent_jobs} jobs")

                    method_name, wrapper_class, fit_kwargs = method
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
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--methods_file', type=str, required=True, help="Path to the YAML file containing methods")

    args = parser.parse_args()

    launch_jobs(
        datasets=args.datasets,
        folds=args.folds,
        methods_file=args.methods_file,
    )


if __name__ == "__main__":
    main()