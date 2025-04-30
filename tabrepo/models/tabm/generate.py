from autogluon.common.space import Real, Int, Categorical
from autogluon.tabular.models import XGBoostModel

from ..utils import convert_numpy_dtypes
from ...benchmark.models.ag.tabm.tabm_model import TabMModel
from ...utils.config_utils import ConfigGenerator, CustomAGConfigGenerator
import numpy as np


# name = 'TabM'
# manual_configs = [
# ]
# search_space = {
#     'batch_size': Categorical('auto'),
#     'patience': Categorical(16),
#     'amp': Categorical(True),  # this only makes sense for GPU
#     'lr': Real(lower=5e-3, upper=0.1, default=0.1, log=True),
#     'max_depth': Int(lower=4, upper=10, default=6),
#     'min_child_weight': Real(0.5, 1.5, default=1.0),
#     'colsample_bytree': Real(0.5, 1.0, default=1.0),
#     'enable_categorical': Categorical(True, False),
# }
#
# gen_xgboost = ConfigGenerator(model_cls=XGBoostModel, manual_configs=[{}], search_space=search_space)
#
#
# def generate_configs_xgboost(num_random_configs=200):
#     config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
#     return config_generator.generate_all_configs(num_random_configs=num_random_configs)



def generate_single_config_tabm(rng):
    # taken from https://github.com/yandex-research/tabm/blob/main/exp/tabm-mini-piecewiselinear/adult/0-tuning.toml
    params = {
        'batch_size': 'auto',
        'patience': 16,
        'amp': 'True',  # only for GPU
        'arch_type': 'tabm-mini',
        'tabm_k': 32,
        'gradient_clipping_norm': 1.0,
        # this makes it probably slower with numerical embeddings, and also more RAM intensive
        # according to the paper it's not very important (?)
        'share_training_batches': False,
        'lr': np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))),
        'weight_decay': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))]),
        # removed n_blocks=1 according to Yury Gurishniy's advice
        'n_blocks': rng.choice([2, 3, 4, 5]),
        # increased lower limit from 64 to 128 according to Yury Gorishniy's advice
        'd_block': rng.choice([i for i in range(128, 1024+1) if i%16==0]),
        'dropout': rng.choice([0.0, rng.uniform([0.0, 0.5])]),

        # numerical embeddings
        # todo: num_emb_type: new version?
        'num_emb_type': 'pwl',
        'd_embedding': rng.choice([i for i in range(8, 32+1) if i%4==0]),
        'num_emb_n_bins': rng.integers(2, 128, endpoint=True),
    }

    params = convert_numpy_dtypes(params)
    return params


def generate_configs_tabm(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_tabm(rng) for _ in range(num_random_configs)]


gen_tabm = CustomAGConfigGenerator(model_cls=TabMModel, search_space_func=generate_configs_tabm,
                                          manual_configs=[{}])

