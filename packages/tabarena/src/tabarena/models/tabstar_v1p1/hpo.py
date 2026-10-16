from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.tabstar_v1p1.model import TabSTARv1p1Model
from tabarena.utils.config_utils import ConfigGenerator

gen_tabstar_v1p1 = ConfigGenerator(
    model_cls=TabSTARv1p1Model,
    manual_configs=[{}],
    search_space={
        "lora_lr": Categorical(0.0001, 0.0002, 0.0005, 0.001, 0.002),
        "lora_wd": Categorical(0, 0.00001, 0.0001, 0.001),
        "lora_r": Categorical(16, 32, 64),
        "lora_alpha": Categorical(1, 2),
        "lora_dropout": Categorical(0, 0.05, 0.1),
        "global_batch": Categorical(64, 128, 256),
    },
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabstar_v1p1.generate_all_bag_experiments(
                num_random_configs=0,
            ),
        ),
    )
