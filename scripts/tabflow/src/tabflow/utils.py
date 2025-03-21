import re
import yaml
import json
import boto3
import logging

from botocore.config import Config
from tabrepo.benchmark.models.model_register import infer_model_cls


def yaml_to_methods(methods_file: str) -> list:
    with open(methods_file, 'r') as file:
        methods_config = yaml.safe_load(file)

    return methods_config['methods']


def parse_method(method_config: dict, context=None):
    # Creating copy as we perform pop() which can lead to errors in subsequent calls
    method_config = method_config.copy()

    if context is None:
        context = globals()
    # Convert string class names to actual class references
    # This assumes the classes are already defined or imported in evaluate.py
    if 'model_cls' in method_config:
        method_config["model_cls"] = infer_model_cls(method_config["model_cls"])
        # method_config['model_cls'] = eval(method_config['model_cls'], context)
    if 'method_cls' in method_config:
        method_config['method_cls'] = eval(method_config['method_cls'], context)
    
    # Evaluate all values in ag_args_fit
    if "model_hyperparameters" in method_config:
        if "ag_args_fit" in method_config["model_hyperparameters"]:
            for key, value in method_config["model_hyperparameters"]["ag_args_fit"].items():
                if isinstance(value, str):
                    try:
                        method_config["model_hyperparameters"]["ag_args_fit"][key] = eval(value, context)
                    except NameError:
                        pass  # If eval fails, keep the original string value



    method_type = eval(method_config.pop('type'), context)
    method_obj = method_type(**method_config)
    return method_obj


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


def check_s3_file_exists(s3_client, bucket: str, cache_name: str) -> bool:
    s3_path = f"s3://{bucket}/{cache_name}"
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise


def upload_methods_config(s3_client, methods_file, bucket, key):
    """Upload methods config to S3, overwriting if it exists."""
    s3_client.upload_file(methods_file, bucket, key)
    return f"s3://{bucket}/{key}"


def upload_tasks_json(s3_client, tasks_json, bucket, key):
    """Upload batched tasks JSON to S3, overwriting if it exists."""
    s3_client.put_object(
        Body=json.dumps(tasks_json).encode('utf-8'),
        Bucket=bucket,
        Key=key
    )
    return f"s3://{bucket}/{key}"


def save_training_job_logs(sagemaker_client, s3_client, job_name, bucket, cache_path):
    """
    Retrieve logs for a completed SageMaker training job and save to S3.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        s3_client: Boto3 S3 client
        job_name: Name of the SageMaker training job
        bucket: S3 bucket name
        cache_path: Path prefix where the logs should be saved (without the .log extension)
    """
    try:        
        # Create CloudWatch logs client + standard log_group
        retry_config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={'max_attempts':20,
                'mode':'adaptive',
                }
        )
        cloudwatch_logs = boto3.client('logs', config=retry_config)
        log_group ='/aws/sagemaker/TrainingJobs'

        response = cloudwatch_logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name
        )

        # Find the matching log stream
        log_stream = None
        for stream in response.get('logStreams', []):
            if stream['logStreamName'].startswith(job_name):
                log_stream = stream['logStreamName']
                break
        
        if not log_stream:
            print(f"No log stream found for job {job_name}")
            return
        
        logs_response = cloudwatch_logs.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            startFromHead=True,
            limit=10000
        )
        
        log_events = logs_response['events']
        
        # Continue retrieving if more logs are available
        while 'nextForwardToken' in logs_response:
            next_token = logs_response['nextForwardToken']
            logs_response = cloudwatch_logs.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                nextToken=next_token,
                limit=10000
            )
            
            if next_token == logs_response['nextForwardToken']:
                break
                
            log_events.extend(logs_response['events'])
        
        full_log_content = ""
        for event in log_events:
            full_log_content += f"{event['message']}\n"

        # Parse tasks from the logs
        task_logs = {}
        task_paths = {}
        current_task = None
        current_dataset = None
        current_tid = None
        current_fold = None
        current_method = None
        task_content = []
        
        for line in full_log_content.split('\n'):
            # Check for task start
            if "Processing task: Dataset=" in line:
                if current_task and task_content:
                    task_logs[current_task] = '\n'.join(task_content)
                    task_paths[current_task] = f"{current_method}/{current_tid}/{current_fold}"
                    task_content = []
                
                # Extract task identifiers
                parts = line.split(",")
                current_dataset = parts[0].split("Dataset=")[1].strip()
                current_tid = parts[1].split("TID=")[1].strip()  # Extract TID directly
                current_fold = parts[2].split("Fold=")[1].strip()
                current_method = parts[3].split("Method=")[1].strip()

                current_task = f"{current_tid}_{current_fold}_{current_method}"
                    
            if current_task:
                task_content.append(line)
        
        if current_task and task_content:
            task_logs[current_task] = '\n'.join(task_content)
            task_paths[current_task] = f"{current_method}/{current_tid}/{current_fold}"
        
        # Extract experiment name from cache_path
        # Assuming cache_path format: "{experiment_name}/data/{method_name}/{tid}/{fold}"
        experiment_name = cache_path.split('/')[0]

        # Save individual task + full batch logs
        for task_name, content in task_logs.items():
            path_parts = task_paths[task_name].split('/')
            method_name = path_parts[0]
            dataset_tid = path_parts[1]
            fold = path_parts[2]
            task_file_path = f"{experiment_name}/data/{method_name}/{dataset_tid}/{fold}/task.log"
            s3_client.put_object(
                Body=content.encode('utf-8'),
                Bucket=bucket,
                Key=task_file_path
            )
            print(f"Task log saved to s3://{bucket}/{task_file_path}")

            full_log_file_path = f"{experiment_name}/data/{method_name}/{dataset_tid}/{fold}/full_log.log"
            s3_client.put_object(
                Body=full_log_content.encode('utf-8'),
                Bucket=bucket,
                Key=full_log_file_path
            )
            print(f"Logs saved to s3://{bucket}/{full_log_file_path}")
    except Exception as e:
        logging.exception(f"Error saving logs for job {job_name}")
