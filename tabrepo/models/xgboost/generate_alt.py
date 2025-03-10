from autogluon.common.space import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'XGBoostAlt'
manual_configs = [
]
search_space = {
    # kept this from the other search space since I had a different number of trees
    'learning_rate': Real(lower=5e-3, upper=0.1, default=0.1, log=True),
    'max_depth': Int(lower=1, upper=11, default=6),
    'min_child_weight': Real(1e-3, 5.0, default=1.0, log=True),
    'colsample_bytree': Real(0.5, 1.0, default=1.0),
    'colsample_bylevel': Real(0.5, 1.0, default=1.0),
    'subsample': Real(0.5, 1.0, default=1.0),
    'lambda': Real(5e-3, 5.0, default=1.0, log=True),
    'alpha': Real(1e-5, 5.0, default=1e-3, log=True),
    'enable_categorical': Categorical(True, False),  # kept this from the other search space
}


def generate_configs_xgboost_alt(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
