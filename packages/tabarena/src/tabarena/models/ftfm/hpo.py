from __future__ import annotations

from tabarena.models.ftfm.model import FTFMModel
from tabarena.utils.config_utils import ConfigGenerator

# One released checkpoint with the library's default eight-view ensemble; no search space.
gen_ftfm = ConfigGenerator(
    model_cls=FTFMModel,
    search_space={},
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_ftfm.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
