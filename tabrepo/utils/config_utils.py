from __future__ import annotations

import copy
from typing import Type

from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher
from autogluon.core.models import AbstractModel

from ..constants.model_constants import MODEL_TYPE_DICT
from tabrepo.benchmark.experiment import AGModelBagExperiment


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


class AbstractConfigGenerator:
    def __init__(
        self,
        manual_configs: list[dict] | None = None,
    ):
        if manual_configs is None:
            manual_configs = []
        self.manual_configs = manual_configs


class AGConfigGenerator(AbstractConfigGenerator):
    def __init__(
        self,
        name: str,
        model_type: str,
        model_cls,
        manual_configs: list[dict] | None = None,
    ):
        super().__init__(manual_configs=manual_configs)
        self.name = name
        self.model_type = model_type
        self.model_cls = model_cls

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        raise NotImplementedError

    def generate_all_configs(self, num_random_configs):
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs)
        configs_dict = configs_to_name_dict(configs=configs, name_prefix=self.name, model_type=self.model_type)
        return configs_dict

    def generate_all_configs_lst(self, num_random_configs: int) -> list[dict]:
        if num_random_configs > 0:
            random_configs = self.get_searcher_configs(num_random_configs)
        else:
            random_configs = []
        configs = combine_manual_and_random_configs(manual_configs=self.manual_configs, random_configs=random_configs)
        return configs

    def generate_all_bag_experiments(self, num_random_configs: int) -> list:
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs)
        experiments = generate_bag_experiments(model_cls=self.model_cls, configs=configs, name_suffix_from_ag_args=True)
        return experiments


class ConfigGenerator(AGConfigGenerator):
    def __init__(
        self,
        search_space: dict,
        model_cls: Type[AbstractModel] | None = None,
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        if model_cls is not None:
            if name is None:
                name = model_cls.ag_name
            model_type = model_cls.ag_key
        else:
            model_type = MODEL_TYPE_DICT[name]
        assert name is not None
        super().__init__(
            name=name,
            model_type=model_type,
            model_cls=model_cls,
            manual_configs=manual_configs,
        )
        self.search_space = search_space

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        searcher = get_random_searcher(self.search_space)
        return [searcher.get_config() for _ in range(num_configs)]


class CustomAGConfigGenerator(AGConfigGenerator):
    def __init__(
        self,
        model_cls: Type[AbstractModel],
        search_space_func,
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        if name is None:
            name = model_cls.ag_name
        model_type = model_cls.ag_key
        super().__init__(
            name=name,
            model_type=model_type,
            model_cls=model_cls,
            manual_configs=manual_configs,
        )
        self.search_space_func = search_space_func
        self.model_cls = model_cls

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        return self.search_space_func(num_configs)


def generate_bag_experiments(
    model_cls,
    configs: list[dict],
    time_limit: float | None = 3600,
    num_bag_folds: int = 8,
    num_bag_sets: int = 1,
    name_suffix_from_ag_args: bool = False,
    name_id_prefix: str = "r",
    name_bag_suffix: str = "_BAG_L1",
    add_name_suffix_to_params: bool = True,
    **kwargs
) -> list[AGModelBagExperiment]:
    experiments = []
    if kwargs is None:
        kwargs = {}

    for i, config in enumerate(configs):
        if name_suffix_from_ag_args:
            name_suffix = config.get("ag_args", {}).get("name_suffix", "")
        else:
            name_suffix = f"_{name_id_prefix}{i+1}"
            if add_name_suffix_to_params:
                config = add_suffix_to_config(config=config, suffix=name_suffix)
        name = f"{model_cls.ag_name}{name_suffix}{name_bag_suffix}"
        experiment = AGModelBagExperiment(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=config,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            time_limit=time_limit,
            **kwargs,
        )
        experiments.append(experiment)
    return experiments
