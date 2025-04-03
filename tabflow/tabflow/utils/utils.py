import re
import yaml

from itertools import islice
from tabrepo.benchmark.models.model_register import infer_model_cls

def create_batch(tasks, batch_size):
    """Convert all tasks into batches"""
    it = iter(tasks)
    for batch in iter(lambda: tuple(islice(it, batch_size)), ()):
        yield batch


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


def yaml_to_methods(methods_file: str) -> list:
    """
    Load methods configuration from a YAML file and return the list of methods.
    """
    with open(methods_file, 'r') as file:
        methods_config = yaml.safe_load(file)

    return methods_config['methods']


def parse_method(method_config: dict, context=None):
    """
    Parse a method configuration dictionary and return an instance of the method class.
    This function evaluates the 'type' field in the method_config to determine the class to instantiate.
    It also evaluates any string values in the configuration that are meant to be Python expressions.
    """
    # Creating copy as we perform pop() which can lead to errors in subsequent calls
    method_config = method_config.copy()

    if context is None:
        context = globals()
    # Convert string class names to actual class references
    # This assumes the classes are already defined or imported in evaluate.py
    if 'model_cls' in method_config:
        method_config["model_cls"] = infer_model_cls(method_config["model_cls"])
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


def find_method_by_name(methods_config, method_name):
    """Find a method configuration by name in the methods configuration"""
    if "methods" in methods_config:
        for method in methods_config["methods"]:
            if method.get("name") == method_name:
                # Return copy to ensure next method if same can be popped as well
                return method.copy()
    return None
