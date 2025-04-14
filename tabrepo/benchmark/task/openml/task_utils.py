from __future__ import annotations


import boto3
import os
import openml
import pandas as pd
import time
from pathlib import Path

from openml import OpenMLSupervisedTask
from openml.exceptions import OpenMLServerException
# from tabflow.utils.constants import OPENML_S3_BUCKET, OPENML_S3_PREFIX
from tabflow.utils.s3_utils import parse_s3_uri


def download_task_from_s3(task_id: int, s3_dataset_cache: str = None) -> bool:
    """
    Attempt to download an OpenML task from S3 before hitting the OpenML server.
    
    Parameters
    ----------
    task_id : int
        The OpenML task ID
    s3_dataset_cache : str, optional
        Full S3 URI to the dataset cache (format: s3://bucket/prefix)
        If None, skip S3 download attempt
        
    Returns
    -------
    bool
        True if successfully downloaded from S3, False otherwise
    """
    if s3_dataset_cache is None:
        return False

    # OpenML cache directory location for tasks
    cache_dir = Path(os.path.expanduser("~/.cache/openml/org/openml/www/tasks"))
    task_cache_dir = cache_dir / str(task_id)
    
    s3_bucket, s3_prefix = parse_s3_uri(s3_dataset_cache=s3_dataset_cache)
    s3_key_prefix = f"{s3_prefix}/tasks/{task_id}/org/openml/www/tasks/{task_id}"
    
    os.makedirs(task_cache_dir, exist_ok=True)
    
    try:
        s3_client = boto3.client('s3')
        
        xml_key = f"{s3_key_prefix}/task.xml"
        try:
            print(f"Attempting to download task {task_id} from S3 bucket {s3_bucket}")
            s3_client.download_file(
                Bucket=s3_bucket,
                Key=xml_key,
                Filename=str(task_cache_dir / "task.xml")
            )
            
            try:
                splits_key = f"{s3_key_prefix}/datasplits.arff"
                s3_client.download_file(
                    Bucket=s3_bucket,
                    Key=splits_key,
                    Filename=str(task_cache_dir / "datasplits.arff")
                )
                print(f"Downloaded datasplits.arff for task {task_id} from S3")
            except s3_client.exceptions.ClientError:
                print(f"No datasplits.arff found in S3 for task {task_id}")
                
            print(f"Successfully downloaded task {task_id} from S3")
            return True
            
        except s3_client.exceptions.ClientError as e:
            print(f"Task {task_id} not found in S3 ({e}), will try local cache or OpenML server")
            return False
            
    except Exception as e:
        print(f"Error accessing S3 for task {task_id}: {str(e)}")
        return False


def get_task(task_id: int) -> OpenMLSupervisedTask:
    task = openml.tasks.get_task(
        task_id,
        download_splits=False,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    if isinstance(task, OpenMLSupervisedTask):
        return task
    else:
        raise AssertionError(f"Invalid task type: {type(task)}")


def get_ag_problem_type(task: OpenMLSupervisedTask) -> str:
    if task.task_type_id.name == 'SUPERVISED_CLASSIFICATION':
        if len(task.class_labels) > 2:
            problem_type = 'multiclass'
        else:
            problem_type = 'binary'
    elif task.task_type_id.name == 'SUPERVISED_REGRESSION':
        problem_type = 'regression'
    else:
        raise AssertionError(f'Unsupported task type: {task.task_type_id.name}')
    return problem_type


def get_task_with_retry(task_id: int, s3_dataset_cache: str = None, max_delay_exp: int = 8) -> OpenMLSupervisedTask:
    # Try to download from S3 first
    download_task_from_s3(task_id, s3_dataset_cache=s3_dataset_cache)
    # If that fails, try to get from OpenML server
    delay_exp = 0
    while True:
        try:
            print(f'Getting task {task_id}')
            task = get_task(task_id=task_id)
            print(f'Got task {task_id}')
            return task
        except OpenMLServerException as e:
            delay = 2 ** delay_exp
            delay_exp += 1
            if delay_exp > max_delay_exp:
                raise ValueError("Unable to get task after 10 retries")
            print(e)
            print(f'Retry in {delay}s...')
            time.sleep(delay)
            continue


def get_task_data(task: OpenMLSupervisedTask) -> tuple[pd.DataFrame, pd.Series]:
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y
