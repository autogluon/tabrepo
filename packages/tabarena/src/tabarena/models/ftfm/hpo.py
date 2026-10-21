from __future__ import annotations

from tabarena.models.ftfm.model import FTFMModel
from tabarena.utils.config_utils import ConfigGenerator


def default_config() -> dict:
    """The default config: fitted contexts only in the refit model.

    The bagged fold models predict their out-of-fold rows once and are then dropped (``refit_folds``), so
    they build their contexts inside that predict; the refit model, which serves the test predictions,
    builds them in ``fit``.
    """
    return {"cache_context": False, "ag.refit_hyperparameters": {"cache_context": True}}


# One released checkpoint with the library's default eight-view ensemble; no search space.
gen_ftfm = ConfigGenerator(
    model_cls=FTFMModel,
    search_space={},
    manual_configs=[default_config()],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_ftfm.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
