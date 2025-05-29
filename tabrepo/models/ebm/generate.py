from autogluon.common.space import Real, Int, Categorical

from tabrepo.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from ...utils.config_utils import ConfigGenerator

name = 'EBM'
manual_configs = [
]
search_space = {
    'max_leaves': Int(2, 3, default=2),
    'smoothing_rounds': Categorical(0, 25, 50, 75, 100, 150, 200, 350, 500, 750, 1000),
    'learning_rate': Real(0.0025, 0.2, default=0.02, log=True),
    'interactions': Real(0.95, 0.999, default=0.999),
    'interaction_smoothing_rounds':  Categorical(0, 25, 50, 75, 100, 200, 500),
    'min_hessian': Real(1e-10, 1e-2, default=1e-4, log=True),
    'min_samples_leaf': Int(2, 20, default=4),
    'gain_scale': Real(0.5, 5.0, default=5.0, log=True),
    'min_cat_samples': Categorical(5, 10, 15, 20),
    'cat_smooth': Categorical(5.0, 10.0, 20.0, 100.0),
    'missing': Categorical('separate', 'low', 'high', 'gain'),
}

gen_ebm = ConfigGenerator(model_cls=ExplainableBoostingMachineModel, search_space=search_space, manual_configs=[{}])


def generate_configs_ebm(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
