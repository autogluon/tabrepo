from __future__ import annotations

import re
import yaml

from itertools import islice


def create_batch(tasks: list[tuple[str, int, int, dict]], batch_size) -> list[list[tuple[str, int, int, dict]]]:
    """Convert all tasks into batches"""
    it = iter(tasks)
    for batch in iter(lambda: tuple(islice(it, batch_size)), ()):
        yield list(batch)


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


def find_method_by_name(methods_config, method_name):
    """Find a method configuration by name in the methods configuration"""
    for method in methods_config:
        if method.get("name") == method_name:
            # Return copy to ensure next method if same can be popped as well
            return method.copy()
    raise ValueError(f"Unknown method: {method_name}")
