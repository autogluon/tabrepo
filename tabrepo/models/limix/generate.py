from __future__ import annotations

from tabrepo.benchmark.models.ag.limix.limix_model import LimiXModel
from tabrepo.utils.config_utils import ConfigGenerator

gen_limix = ConfigGenerator(model_cls=LimiXModel, manual_configs=[{}], search_space={})
