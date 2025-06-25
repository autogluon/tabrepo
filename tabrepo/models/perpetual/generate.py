from __future__ import annotations

from tabrepo.benchmark.models.ag.perpetual.perpetual_model import PerpetualBoostingModel
from tabrepo.utils.config_utils import ConfigGenerator

# TODO: ask authors for search space / come up with something.
search_space = {}

gen_perpetual = ConfigGenerator(
    model_cls=PerpetualBoostingModel,
    search_space=search_space,
    manual_configs=[{}],
)
