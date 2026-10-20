from __future__ import annotations

from tabarena.models.compacttab.model import CompactTabFoundationMediumModel, CompactTabFoundationModel
from tabarena.utils.config_utils import ConfigGenerator

gen_compacttab = ConfigGenerator(
    model_cls=CompactTabFoundationModel,
    search_space={},
    manual_configs=[{}],
)

gen_compacttab_medium = ConfigGenerator(
    model_cls=CompactTabFoundationMediumModel,
    search_space={},
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_compacttab.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
