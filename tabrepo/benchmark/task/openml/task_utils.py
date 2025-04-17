from __future__ import annotations


import boto3
import logging
import os
import openml
import pandas as pd
import time
import xmltodict
from pathlib import Path

from openml import OpenMLSupervisedTask
from openml.exceptions import OpenMLServerException
from tabflow.utils.s3_utils import parse_s3_uri

logger = logging.getLogger(__name__)


def download_task_from_s3(task_id: int, s3_dataset_cache: str = None) -> bool:
    """
    Downloads the task and dataset from S3 if available.
    """
    if s3_dataset_cache is None:
        return False

    # OpenML cache directory location for tasks + datasets
    local_cache_dir = Path(os.path.expanduser("~/.cache/openml/org/openml/www"))
    os.makedirs(local_cache_dir, exist_ok=True)
    
    try:
        s3_client = boto3.client('s3')
        s3_bucket, s3_prefix = parse_s3_uri(s3_uri=s3_dataset_cache)

        task_cache_dir = local_cache_dir /  "tasks" / str(task_id)
        os.makedirs(task_cache_dir, exist_ok=True)

        logger.info(f"Attempting to download task {task_id} from S3 bucket {s3_bucket}")
        s3_key_prefix = f"{s3_prefix}/tasks/{task_id}/org/openml/www/tasks/{task_id}"
        
        try:
            s3_client.download_file(
                Bucket=s3_bucket,
                Key=f"{s3_key_prefix}/task.xml",
                Filename=str(task_cache_dir / "task.xml")
            )
            logger.info(f"Downloaded task.xml for task {task_id} from S3")
            
            try:
                s3_client.download_file(
                    Bucket=s3_bucket,
                    Key=f"{s3_key_prefix}/datasplits.arff",
                    Filename=str(task_cache_dir / "datasplits.arff")
                )
                logger.info(f"Downloaded datasplits.arff for task {task_id} from S3")
            except s3_client.exceptions.ClientError:
                logger.info(f"No datasplits.arff found in S3 for task {task_id}")

            with open(task_cache_dir / "task.xml", 'r') as f:
                task_xml = f.read()

            task_dict = xmltodict.parse(task_xml)["oml:task"]
            dataset_id = None
            if isinstance(task_dict["oml:input"], list):
                for input_item in task_dict["oml:input"]:
                    if input_item["@name"] == "source_data":
                        dataset_id = int(input_item["oml:data_set"]["oml:data_set_id"])
                        break
            else:
                if task_dict["oml:input"]["@name"] == "source_data":
                    dataset_id = int(task_dict["oml:input"]["oml:data_set"]["oml:data_set_id"])
            
            if dataset_id is not None:
                dataset_id_str = str(dataset_id)
                dataset_cache_dir = local_cache_dir / "datasets" / dataset_id_str
                os.makedirs(dataset_cache_dir, exist_ok=True)
                
                logger.info(f"Attempting to download dataset {dataset_id} data from S3")
                s3_dataset_prefix = f"{s3_prefix}/tasks/{task_id}/org/openml/www/datasets/{dataset_id}"
                
                try:
                    response = s3_client.list_objects_v2(
                        Bucket=s3_bucket,
                        Prefix=f"{s3_dataset_prefix}/"
                    )

                    if 'Contents' in response:
                        for obj in response['Contents']:
                            s3_key = obj['Key']
                            filename = os.path.basename(s3_key)
                            if filename == '':
                                continue
                            try:
                                local_file_path = str(dataset_cache_dir / filename)
                                s3_client.download_file(
                                    Bucket=s3_bucket,
                                    Key=s3_key,
                                    Filename=local_file_path
                                )
                                logger.info(f"Downloaded {filename} for dataset {dataset_id} from S3")
                            except s3_client.exceptions.ClientError as e:
                                logger.info(f"Error downloading {filename} for dataset {dataset_id}: {e}")
                    else:
                        logger.info(f"No files found in S3 for dataset {dataset_id}")
                except s3_client.exceptions.ClientError as e:
                    logger.warning(f"Error listing objects in S3 for dataset {dataset_id}: {e}")

            logger.info(f"Successfully downloaded task {task_id} from S3")
            return True
            
        except s3_client.exceptions.ClientError as e:
            logger.info(f"Task {task_id} not found in S3 ({e}), now trying OpenML server...")
            return False
            
    except Exception as e:
        logger.error(f"Error accessing S3 for task {task_id}: {str(e)}")
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
