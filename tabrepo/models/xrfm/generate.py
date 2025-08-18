from __future__ import annotations

from autogluon.tabular.models import LGBModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabrepo.benchmark.experiment import YamlExperimentSerializer
from tabrepo.benchmark.models.ag.xrfm.xrfm_model import xRFMModel
from tabrepo.models.utils import convert_numpy_dtypes
from tabrepo.utils.config_utils import CustomAGConfigGenerator


def generate_configs_xrfm(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(
        space=[
            Float("bandwidth", (0.5, 200), log=True),
            Categorical("bandwidth_mode", ['constant', 'adaptive']),
            # could tune categorical preprocessing if ordinal encoding is implemented
            Categorical("diag", [False, True]),
            Categorical("early_stop_multiplier", [1.06]),
            Float("exponent", (0.7, 1.4)),
            Categorical("kernel", ['l1_kermac', 'l2']),
            Float("reg", (1e-6, 1e1), log=True),
            # todo: refill size?
            Categorical("classification_mode", ['prevalence']),  # this differs from the paper
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    return [convert_numpy_dtypes(config) for config in configs]


gen_xrfm = CustomAGConfigGenerator(
    model_cls=xRFMModel,
    search_space_func=generate_configs_xrfm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    experiments = gen_xrfm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_xrfm.yaml"
    )
