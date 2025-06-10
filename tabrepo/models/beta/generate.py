from __future__ import annotations

from tabrepo.benchmark.models.ag.beta.beta_model import BetaModel
from tabrepo.utils.config_utils import ConfigGenerator

name = "BETA"
manual_configs = [
    {},
]

gen_beta = ConfigGenerator(
    model_cls=BetaModel, manual_configs=manual_configs, search_space={}
)

if __name__ == "__main__":
    from tabrepo.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_beta.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
