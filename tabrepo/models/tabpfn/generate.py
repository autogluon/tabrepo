from ...utils.config_utils import ConfigGenerator


name = 'TabPFN'
manual_configs = [
    {},
    {'N_ensemble_configurations': 4},
    {'N_ensemble_configurations': 8},
]
search_space = {}


def generate_configs_tabpfn(num_random_configs=0):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
