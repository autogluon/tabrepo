from autogluon.common.space import Categorical, Real, Int

from ...utils.config_utils import ConfigGenerator


name = 'CatBoost'
manual_configs = [
    {},
]
search_space = {
    'learning_rate': Real(lower=5e-3, upper=0.1, default=0.05, log=True),
    'depth': Int(lower=4, upper=8, default=6),
    'l2_leaf_reg': Real(lower=1, upper=5, default=3),
    'max_ctr_complexity': Int(lower=1, upper=5, default=4),
    'one_hot_max_size': Categorical(2, 3, 5, 10),
    'grow_policy': Categorical("SymmetricTree", "Depthwise")
}


def generate_configs_catboost(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
