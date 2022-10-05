from autogluon.core import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'RandomForest'
manual_configs = [
    {},
]
search_space = {
    'max_leaf_nodes': Categorical(5000, 15000, 30000, 50000),
    'min_samples_leaf': Categorical(1, 2, 3, 4, 5, 10, 20, 40, 80),
    'max_features': Categorical('sqrt', 'log2', 0.5)
}


def generate_configs_random_forest(num_random_configs=50):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
