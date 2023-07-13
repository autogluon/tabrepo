from autogluon.common.space import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'XGBoost'
manual_configs = [
    {},
    {'learning_rate': 1e-2},
    {'enable_categorical': True},
    {'learning_rate': 1e-2, 'enable_categorical': True},
]
search_space = {
    'learning_rate': Real(lower=5e-3, upper=0.1, default=0.1, log=True),
    'max_depth': Int(lower=4, upper=10, default=6),
    'min_child_weight': Real(0.5, 1.5, default=1.0),
    'colsample_bytree': Real(0.5, 1.0, default=1.0),
    'enable_categorical': Categorical(True, False),
}


def generate_configs_xgboost(num_random_configs=100):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
