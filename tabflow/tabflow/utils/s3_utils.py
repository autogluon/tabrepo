import os
import boto3
import json
from botocore.config import Config

def check_s3_file_exists(s3_client, bucket: str, cache_name: str) -> bool:
    """Helper function to check if a file exists in S3"""
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


def download_from_s3(s3_path, destination_path=None):
    """
    Download a file or folder from S3 to a local path
    Used by evaluate.py to download methods and tasks when destination_path is None
    Can also be used as a utility function by user to download experiment results
    
    Args:
        s3_path: S3 URI in the format 's3://bucket/key'
        destination_path: Optional local destination path
            - If None: downloads a single file with original filename (current behavior)
            - If provided: downloads recursively if s3_path points to a prefix/folder
    
    Returns:
        The path to the downloaded file or folder
    """
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    s3_retry_config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={'max_attempts':20,
                'mode':'adaptive',
                }
    )
    s3_client = boto3.client('s3', config=s3_retry_config)
    
    if destination_path is None:
        local_path = os.path.basename(key)
        s3_client.download_file(bucket, key, local_path)
        return local_path
    
    else:
        os.makedirs(destination_path, exist_ok=True)
        
        # Check if it's a single file or directory
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            local_path = os.path.join(destination_path, os.path.basename(key))
            s3_client.download_file(bucket, key, local_path)
            return local_path
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                if not key.endswith('/'):
                    key = f"{key}/"
                
                paginator = s3_client.get_paginator('list_objects_v2')
                result = paginator.paginate(Bucket=bucket, Prefix=key)
                
                downloaded = False
                for page in result:
                    if "Contents" in page:
                        downloaded = True
                        for obj in page['Contents']:
                            rel_path = obj['Key'][len(key):] if obj['Key'] != key else ''
                            if rel_path:
                                local_file_path = os.path.join(destination_path, rel_path)
                                local_dir = os.path.dirname(local_file_path)
                                if local_dir:
                                    os.makedirs(local_dir, exist_ok=True)
                                    
                                if not obj['Key'].endswith('/'):
                                    s3_client.download_file(bucket, obj['Key'], local_file_path)
                
                if not downloaded:
                    raise FileNotFoundError(f"No objects found with prefix: {s3_path}")
                    
                return destination_path
            else:
                raise
            