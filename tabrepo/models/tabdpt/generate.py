from __future__ import annotations

from tabrepo.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
from tabrepo.utils.config_utils import ConfigGenerator

name = "TabDPT"
manual_configs = [
    # Default config with refit after cross-validation.
    {"ag_args_ensemble": {"refit_folds": True}},
]

gen_tabdpt = ConfigGenerator(
    model_cls=TabDPTModel, manual_configs=manual_configs, search_space={}
)

if __name__ == "__main__":
    from tabrepo.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabdpt.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
