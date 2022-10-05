from autogluon.core import Real, Int

from ...utils.config_utils import ConfigGenerator


name = 'CatBoost'
manual_configs = [
    {},
]
search_space = {
    'learning_rate': Real(lower=5e-3, upper=0.1, default=0.05, log=True),
    'depth': Int(lower=4, upper=10, default=6),
    'l2_leaf_reg': Real(lower=1, upper=5, default=3),
}


def generate_configs_catboost(num_random_configs=100):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
