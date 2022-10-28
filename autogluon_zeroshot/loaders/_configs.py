from typing import List

from autogluon.common.loaders import load_json


def load_configs(config_files: List[str]) -> dict:
    configs = {}
    for c in config_files:
        configs.update(load_json.load(path=c))
    return configs
