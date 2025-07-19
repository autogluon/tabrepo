from __future__ import annotations

import numpy as np

from tabrepo.benchmark.models.ag.dpdt.dpdt_model import BoostedDPDTModel
from tabrepo.models.utils import convert_numpy_dtypes
from tabrepo.utils.config_utils import CustomAGConfigGenerator


def generate_configs_bdpdt(num_random_configs=200):
    # TODO: transform this to a ConfigSpace configuration space or similar
    # TODO: and/or switch to better random seed logic

    # Generate 1000 samples from log-normal distribution
    # Parameters: mu = log(0.01), sigma = log(10.0)
    np.random.seed(42)  # For reproducibility
    mu = float(np.log(0.01))
    sigma = float(np.log(10.0))
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_random_configs)

    # Generate 1000 samples from q_log_uniform_values distribution
    # Parameters: min=1.5, max=50.5, q=1
    np.random.seed(43)
    min_val = 1.5
    max_val = 50.5
    q = 1
    # Generate log-uniform samples and quantize
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    log_uniform_samples = np.random.uniform(log_min, log_max, size=num_random_configs)
    min_samples_leaf_samples = np.round(np.exp(log_uniform_samples) / q) * q
    min_samples_leaf_samples = np.clip(
        min_samples_leaf_samples, min_val, max_val
    ).astype(int)

    # Generate 1000 samples for min_weight_fraction_leaf
    # Values: [0.0, 0.01], probabilities: [0.95, 0.05]
    np.random.seed(44)
    min_weight_fraction_leaf_samples = np.random.choice(
        [0.0, 0.01], size=num_random_configs, p=[0.95, 0.05]
    )

    # Generate 1000 samples for max_features
    # Values: ["sqrt", "log2", 10000], probabilities: [0.5, 0.25, 0.25]
    np.random.seed(45)
    max_features_samples = np.random.choice(
        ["sqrt", "log2", 10000], size=num_random_configs, p=[0.5, 0.25, 0.25]
    )

    np.random.seed(46)
    max_depth_samples = np.random.choice([2, 3], size=num_random_configs, p=[0.4, 0.6])

    np.random.seed(47)
    min_samples_split = np.random.choice(
        [2, 3], size=num_random_configs, p=[0.95, 0.05]
    )

    np.random.seed(48)
    min_impurity_decrease_samples = np.random.choice(
        [0, 0.01, 0.02, 0.05], size=num_random_configs, p=[0.85, 0.05, 0.05, 0.05]
    )

    np.random.seed(49)
    choices = [[8, 4], [4, 8], [16, 2], [4, 4, 2]]
    indices = np.random.choice(len(choices), size=num_random_configs)
    cart_nodes_list = [choices[i] for i in indices]

    configs = []
    for i in range(num_random_configs):
        config = {
            "learning_rate": samples[i],
            "max_depth": max_depth_samples[i],
            "min_samples_split": min_samples_split[i],
            "min_impurity_decrease": min_impurity_decrease_samples[i],
            "cart_nodes_list": cart_nodes_list[i],
            "min_samples_leaf": min_samples_leaf_samples[i],
            "min_weight_fraction_leaf": min_weight_fraction_leaf_samples[i],
            "max_features": max_features_samples[i],
        }
        configs.append(config)

    return [convert_numpy_dtypes(config) for config in configs]


gen_boosteddpdt = CustomAGConfigGenerator(
    model_cls=BoostedDPDTModel,
    search_space_func=generate_configs_bdpdt,
    manual_configs=[{}],
)


if __name__ == "__main__":
    print(generate_configs_bdpdt(3))
