from ConfigSpace import ConfigurationSpace, Float, Categorical, Integer

name = 'CatBoost'

search_space = ConfigurationSpace(space=[
    Float('learning_rate', (5e-3, 1e-1), log=True),

    Categorical('bootstrap_type', ['Bernoulli']),  # this is a bit faster than 'Bayesian'
    Float('subsample', (0.7, 1.0)),

    Categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
    Integer('depth', (4, 8)),  # not too large for compute/memory reasons

    # leaving this out for now because catboost complains when it's supplied in SymmetricTree mode
    # Integer('min_data_in_leaf', (1, 100), log=True),  # todo: this only works for Depthwise!

    Float('colsample_bylevel', (0.85, 1.0)),
    Float('l2_leaf_reg', (1e-4, 5.0), log=True),
    # could add random_strength here but leaving it out for now

    Integer('leaf_estimation_iterations', (1, 20), log=True),

    # categorical hyperparameters
    Integer('one_hot_max_size', (8, 100), log=True),
    Float('model_size_reg', (0.1, 1.5), log=True),
    Integer('max_ctr_complexity', (2, 5)),


    # make sure the GPU version uses the same settings
    # (at least these are the two problematic parameters that I know of)
    Categorical('boosting_type', ['Plain']),
    Categorical('max_bin', [254]),  # could be tuned, in principle

    # could search max_bin but this is expensive
], seed=1234)


def generate_configs_catboost_alt(num_random_configs=200):
    return [dict(config) for config in search_space.sample_configuration(num_random_configs)]

if __name__ == '__main__':
    print(generate_configs_catboost_alt(3))
