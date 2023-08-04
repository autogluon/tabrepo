from autogluon.common.space import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'ExtraTrees'
manual_configs = [
    {},
]
search_space = {
    'max_leaf_nodes': Int(5000, 50000),
    'min_samples_leaf': Categorical(1, 2, 3, 4, 5, 10, 20, 40, 80),
    'max_features': Categorical('sqrt', 'log2', 0.5, 0.75, 1.0)
}


def generate_configs_extra_trees(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
