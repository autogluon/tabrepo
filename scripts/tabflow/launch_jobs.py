import boto3
import sagemaker
from pathlib import Path
import fire
import re
from datetime import datetime


DOCKER_IMAGE_ALIASES = {
    "mlflow-image": "097403188315.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo",
}


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
        entry_point: str = "evaluate.py",
        source_dir: str = ".",
        instance_type: str = "ml.m6i.4xlarge",
        docker_image_uri: str = "mlflow-image",
        sagemaker_role: str = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145",
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        job_name: str = None,
        keep_alive_period_in_seconds: int = 300,
        limit_runtime: int = 24 * 60 * 60,
) -> None:
    """
    Launch multiple SageMaker training jobs.

    Args:
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
    """

    # Load methods from YAML file
    with open(args.methods, 'r') as file:
        methods_data = yaml.safe_load(file)

    methods = [(method["name"], eval(method["wrapper_class"]), method["fit_kwargs"]) for method in methods_data["methods"]]


    # combinations = [
    #     {
    #         "dataset": "Australian",
    #         "fold": 0,
    #     },
    #     {
    #         "dataset": "Australian",
    #         "fold": 1,
    #     },
    #     {
    #         "dataset": "blood-transfusion-service-center",
    #         "fold": 0,
    #     },
    #     {
    #         "dataset": "blood-transfusion-service-center",
    #         "fold": 1,
    #     },
    # ]

    for combination in combinations:
        # Create a unique job name
        # job_name = f"mlflow-job_{combination['dataset']}_fold_{combination['fold']}"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"mlflow-{combination['dataset']}-f{combination['fold']}-{timestamp}"
        job_name = sanitize_job_name(base_name)


        if docker_image_uri in DOCKER_IMAGE_ALIASES:
            print(f"Expanding docker_image_uri alias '{docker_image_uri}' -> '{DOCKER_IMAGE_ALIASES[docker_image_uri]}'")
            docker_image_uri = DOCKER_IMAGE_ALIASES[docker_image_uri]

        # Create SageMaker session
        sagemaker_session = (
            sagemaker.Session(boto_session=boto3.Session(profile_name=aws_profile))
            if aws_profile is not None
            else sagemaker.Session()
        )

        # Update hyperparameters for this job
        job_hyperparameters = hyperparameters.copy() if hyperparameters else {}
        job_hyperparameters.update({
            "dataset": combination["dataset"],
            "fold": combination["fold"],    # Can be a 'str' as well, refer to Estimators in SM docs
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
        print(f"Launched training job: {estimator.latest_training_job.name}")


def main():
    """Entrypoint for CLI"""
    fire.Fire(launch_jobs)


if __name__ == "__main__":
    main()