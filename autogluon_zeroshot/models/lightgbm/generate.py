from autogluon.core import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'LightGBM'
manual_configs = [
    {},
    {'extra_trees': True},
    {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 5},
    {'extra_trees': True, 'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 5},
]
search_space = {
    'learning_rate': Real(lower=5e-3, upper=0.2, default=0.05, log=True),
    'feature_fraction': Real(lower=0.65, upper=1.0, default=1.0),
    'min_data_in_leaf': Int(lower=2, upper=60, default=20),
    'num_leaves': Int(lower=16, upper=128, default=31),
    'extra_trees': Categorical(False, True),
}


def generate_configs_lightgbm(num_random_configs=100):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
