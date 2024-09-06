import re
import yaml
import boto3

from tabrepo.benchmark.models.model_register import infer_model_cls


def yaml_to_methods(methods_file: str) -> list:
    with open(methods_file, 'r') as file:
        methods_config = yaml.safe_load(file)

    return methods_config['methods']

def parse_method(method_config: dict, context=None):

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