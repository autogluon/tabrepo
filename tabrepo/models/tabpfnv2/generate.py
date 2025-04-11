from ...utils.config_utils import ConfigGenerator

from tabrepo.benchmark.models.ag.tabpfnv2.tabpfnv2_model import TabPFNV2Model

manual_configs = [
    {},
]
search_space = {}

gen_tabpfnv2 = ConfigGenerator(model_cls=TabPFNV2Model, search_space=search_space, manual_configs=manual_configs)
