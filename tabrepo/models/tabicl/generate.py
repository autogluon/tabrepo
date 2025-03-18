from autogluon.common.space import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator

name = 'TabICL'
manual_configs = [
    {},
]
search_space = {
    'norm_methods': Categorical('none', 'power', 'robust', 'quantile_rtdl', ['none', 'power']),
    # just in case, tuning between TabICL and TabPFN defaults
    'outlier_threshold': Real(4.0, 12.0),
    'average_logits': Categorical(False, True),
    # if average_logits=True this is equivalent to temperature scaling
    'softmax_temperature': Real(0.7, 1.0),
}


def generate_configs_tabicl(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
