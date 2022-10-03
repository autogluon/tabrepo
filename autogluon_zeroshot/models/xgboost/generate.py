from autogluon.core import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'XGBoost'
manual_configs = [
    {},
    {'learning_rate': 1e-2},
]
search_space = {
    'learning_rate': Real(lower=5e-3, upper=0.1, default=0.1, log=True),
    'max_depth': Int(lower=4, upper=10, default=6),
    'min_child_weight': Categorical(1.0, 0.5, 0.75, 1.25, 1.5),
    'colsample_bytree': Categorical(1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
}


def generate_configs_xgboost(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
