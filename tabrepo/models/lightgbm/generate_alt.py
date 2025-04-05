from ConfigSpace import ConfigurationSpace, Float, Categorical, Integer

from tabrepo.models.utils import convert_numpy_dtypes
from tabrepo.benchmark.experiment import YamlExperimentSerializer
from tabrepo.utils.config_utils import generate_bag_experiments


def generate_configs_lightgbm_alt(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(space=[
        Float('learning_rate', (5e-3, 1e-1), log=True),
        Float('feature_fraction', (0.4, 1.0)),
        Float('bagging_fraction', (0.7, 1.0)),
        Categorical('bagging_freq', [1]),
        Integer('num_leaves', (2, 200), log=True),
        Integer('min_data_in_leaf', (1, 64), log=True),
        Categorical('extra_trees', [False, True]),

        # categorical hyperparameters
        Integer('min_data_per_group', (2, 100), log=True),
        Float('cat_l2', (5e-3, 2), log=True),
        Float('cat_smooth', (1e-3, 100), log=True),
        Integer('max_cat_to_onehot', (8, 100), log=True),

        # these seem to help a little bit but can also make things slower
        Float('lambda_l1', (1e-4, 1.0)),
        Float('lambda_l2', (1e-4, 2.0)),

        # could search max_bin but this is expensive
    ], seed=1234)

    configs = [dict(config) for config in search_space.sample_configuration(num_random_configs)]
    configs = [convert_numpy_dtypes(config) for config in configs]
    return configs


if __name__ == '__main__':
    lightgbm_configs = generate_configs_lightgbm_alt(200)

    from autogluon.tabular.models import LGBModel
    experiments = generate_bag_experiments(model_cls=LGBModel, configs=lightgbm_configs, time_limit=3600)
    YamlExperimentSerializer.to_yaml(experiments=experiments, path="configs_lightgbm_alt.yaml")
