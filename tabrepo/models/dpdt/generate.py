from autogluon.common.space import Categorical, Real, Int
import numpy as np

from tabrepo.benchmark.models.ag.dpdt.dpdt_model import BoostedDPDTModel
from tabrepo.utils.config_utils import ConfigGenerator

name = 'BoostedDPDT'
manual_configs = [
    {},
]

# get config from paper

# Generate 1000 samples from log-normal distribution
# Parameters: mu = log(0.01), sigma = log(10.0)
mu = float(np.log(0.01))
sigma = float(np.log(10.0))
samples = np.random.lognormal(mean=mu, sigma=sigma, size=1000)

# Generate 1000 samples from q_log_uniform_values distribution
# Parameters: min=1.5, max=50.5, q=1
min_val = 1.5
max_val = 50.5
q = 1
# Generate log-uniform samples and quantize
log_min = np.log(min_val)
log_max = np.log(max_val)
log_uniform_samples = np.random.uniform(log_min, log_max, size=1000)
min_samples_leaf_samples = np.round(np.exp(log_uniform_samples) / q) * q
min_samples_leaf_samples = np.clip(min_samples_leaf_samples, min_val, max_val).astype(int)

# Generate 1000 samples for min_weight_fraction_leaf
# Values: [0.0, 0.01], probabilities: [0.95, 0.05]
min_weight_fraction_leaf_samples = np.random.choice([0.0, 0.01], size=1000, p=[0.95, 0.05])

# Generate 1000 samples for max_features
# Values: ["sqrt", "log2", 10000], probabilities: [0.5, 0.25, 0.25]
max_features_samples = np.random.choice(["sqrt", "log2", 10000], size=1000, p=[0.5, 0.25, 0.25])

search_space = {
    'learning_rate': Categorical(*samples),  # log_normal distribution equivalent
    'n_estimators': 1000,  # Fixed value as per old config
    'max_depth': Categorical(2, 2, 2, 2, 3, 3, 3, 3, 3, 3),
    'min_samples_split': Categorical(*np.random.choice([2, 3], size=1000, p=[0.95, 0.05])),
    'min_impurity_decrease': Categorical(*np.random.choice([0, 0.01, 0.02, 0.05], size=1000, p=[0.85, 0.05, 0.05, 0.05])),
    'cart_nodes_list': Categorical((8, 4), (4, 8), (16, 2), (4, 4, 2)),
    'min_samples_leaf': Categorical(*min_samples_leaf_samples),  # q_log_uniform equivalent
    'min_weight_fraction_leaf': Categorical(*min_weight_fraction_leaf_samples),
    'max_features': Categorical(*max_features_samples),
    'random_state': Categorical(0, 1, 2, 3, 4)
}

gen_boosteddpdt = ConfigGenerator(model_cls=BoostedDPDTModel, manual_configs=manual_configs, search_space=search_space)


def generate_configs_boosted_dpdt(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
