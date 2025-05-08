import logging
import boto3
from botocore.config import Config

def setup_logging(level=logging.INFO, verbose=False):
    """
    Set up logging configuration for cli.
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
    )
    
    if not verbose:
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('sagemaker').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def save_training_job_logs(sagemaker_client, s3_client, job_name, bucket, cache_path):
    """
    Retrieve logs for each completed SageMaker training job and save to S3.
    
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
            logging.warning(f"No log stream found for job {job_name}")
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
        current_repeat = None
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
                current_repeat = parts[2].split("Repeat=")[1].strip()
                current_fold = parts[3].split("Fold=")[1].strip()
                current_method = parts[4].split("Method=")[1].strip()

                current_task = f"{current_tid}_{current_repeat}_{current_fold}_{current_method}"
                    
            if current_task:
                task_content.append(line)
        
        if current_task and task_content:
            task_logs[current_task] = '\n'.join(task_content)
            task_paths[current_task] = f"{current_method}/{current_tid}/{current_repeat}_{current_fold}"
        
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
            # print(f"Task log saved to s3://{bucket}/{task_file_path}")

            full_log_file_path = f"{experiment_name}/data/{method_name}/{dataset_tid}/{fold}/full_log.log"
            s3_client.put_object(
                Body=full_log_content.encode('utf-8'),
                Bucket=bucket,
                Key=full_log_file_path
            )
            # print(f"Logs saved to s3://{bucket}/{full_log_file_path}")
    except Exception as e:
        logging.exception(f"Error saving logs for job {job_name}")
        