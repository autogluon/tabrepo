from autogluon.common.space import Real, Int, Categorical

from tabrepo.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from ...utils.config_utils import ConfigGenerator

name = 'EBM'
manual_configs = [
]
search_space = {
    'max_leaves': Int(2, 3, default=2),
    'smoothing_rounds': Int(0, 1000, default=200),
    'learning_rate': Real(0.0025, 0.2, default=0.02, log=True),
    'interactions': Real(0.95, 0.999, default=0.999),
    'interaction_smoothing_rounds': Int(0, 200, default=90),
    'min_hessian': Real(1e-10, 1e-2, default=1e-5, log=True),
    'min_samples_leaf': Int(2, 20, default=4),
    'validation_size': Real(0.05, 0.25, default=0.15),
    'early_stopping_tolerance': Real(1e-10, 1e-5, default=1e-5, log=True),
    'gain_scale': Real(0.5, 5.0, default=5.0, log=True),
}

gen_ebm = ConfigGenerator(model_cls=ExplainableBoostingMachineModel, search_space=search_space, manual_configs=[{}])


def generate_configs_ebm(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
