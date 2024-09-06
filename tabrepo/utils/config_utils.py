from __future__ import annotations

import copy
from typing import Type

from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher
from autogluon.core.models import AbstractModel

from ..constants.model_constants import MODEL_TYPE_DICT


def configs_to_name_dict(configs, name_prefix, model_type):
    configs_dict = {}
    for c in configs:
        name_suffix = c["ag_args"]["name_suffix"]
        c_name = f'{name_prefix}{name_suffix}'
        config_dict = dict()
        config_dict['hyperparameters'] = c
        config_dict['name_prefix'] = name_prefix
        config_dict['name_suffix'] = name_suffix
        config_dict['model_type'] = model_type
        configs_dict[c_name] = config_dict
    return configs_dict


def combine_manual_and_random_configs(manual_configs, random_configs):
    combined_configs = []
    for i, config in enumerate(manual_configs):
        combined_configs.append(add_suffix_to_config(config=config, suffix=f'_c{i+1}'))
    for i, config in enumerate(random_configs):
        combined_configs.append(add_suffix_to_config(config=config, suffix=f'_r{i+1}'))
    return combined_configs


def add_suffix_to_config(config, suffix):
    if "ag_args" in config:
        raise AssertionError(f"ag_args already exists in config!")
    config = copy.deepcopy(config)
    config['ag_args'] = {'name_suffix': suffix}
    return config


def get_random_searcher(search_space):
    searcher = LocalRandomSearcher(search_space=search_space)
    searcher.get_config()  # Clean out default
    return searcher


class ConfigGenerator:
    def __init__(self, name, manual_configs, search_space, model_cls: Type[AbstractModel] | None = None):
        if model_cls is not None:
            self.name = model_cls.ag_name
            self.model_type = model_cls.ag_key
        else:
            self.name = name
            self.model_type = MODEL_TYPE_DICT[name]
        self.manual_configs = manual_configs
        self.search_space = search_space
        self.searcher = get_random_searcher(search_space)

    @classmethod
    def from_cls(cls, model_cls, manual_configs, search_space):
        name = model_cls
        return cls(name=name, manual_configs=manual_configs, search_space=search_space)

    def get_searcher_config(self):
        return self.searcher.get_config()

    def get_searcher_configs(self, num_configs):
        return [self.get_searcher_config() for _ in range(num_configs)]

    def generate_all_configs(self, num_random_configs):
        if num_random_configs > 0:
            random_configs = self.get_searcher_configs(num_random_configs)
        else:
            random_configs = []
        configs = combine_manual_and_random_configs(manual_configs=self.manual_configs, random_configs=random_configs)
        configs_dict = configs_to_name_dict(configs=configs, name_prefix=self.name, model_type=self.model_type)
        return configs_dict
