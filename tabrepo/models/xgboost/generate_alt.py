from ConfigSpace import ConfigurationSpace, Float, Categorical, Integer

name = 'XGBoost'

search_space = ConfigurationSpace(space=[
    Float('learning_rate', (5e-3, 1e-1), log=True),
    Integer('max_depth', (4, 10), log=True),
    Float('min_child_weight', (1e-3, 5.0), log=True),
    Float('subsample', (0.6, 1.0)),
    Float('colsample_bylevel', (0.6, 1.0)),
    Float('colsample_bynode', (1e-4, 5.0)),
    Float('reg_alpha', (1e-4, 5.0)),
    Float('reg_lambda', (1e-4, 5.0)),
    Categorical('grow_policy', ['depthwise', 'lossguide']),
    Integer('max_cat_to_onehot', (8, 100), log=True),
    # todo: do we still need to set enable_categorical?
    # could search max_bin and num_parallel_tree but this is expensive
], seed=1234)


def generate_configs_xgboost_alt(num_random_configs=200):
    return [dict(config) for config in search_space.sample_configuration(num_random_configs)]

if __name__ == '__main__':
    print(generate_configs_xgboost_alt(3))
