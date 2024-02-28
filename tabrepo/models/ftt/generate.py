from autogluon.common.space import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'FTTransformer'
manual_configs = [
    {},
    {
        "model.ft_transformer.n_blocks": 3,
        "model.ft_transformer.d_token": 192,
        "model.ft_transformer.adapter_output_feature": 192,
        "model.ft_transformer.ffn_d_hidden": 192,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
    {
        "model.ft_transformer.n_blocks": 2,
        "model.ft_transformer.d_token": 192,
        "model.ft_transformer.adapter_output_feature": 192,
        "model.ft_transformer.ffn_d_hidden": 192,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
    {
        "model.ft_transformer.n_blocks": 4,
        "model.ft_transformer.d_token": 192,
        "model.ft_transformer.adapter_output_feature": 192,
        "model.ft_transformer.ffn_d_hidden": 192,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
    {
        "model.ft_transformer.n_blocks": 3,
        "model.ft_transformer.d_token": 256,
        "model.ft_transformer.adapter_output_feature": 256,
        "model.ft_transformer.ffn_d_hidden": 256,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
    {
        "model.ft_transformer.n_blocks": 3,
        "model.ft_transformer.d_token": 192,
        "model.ft_transformer.adapter_output_feature": 192,
        "model.ft_transformer.ffn_d_hidden": 288,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
    {
        "model.ft_transformer.n_blocks": 3,
        "model.ft_transformer.d_token": 192,
        "model.ft_transformer.adapter_output_feature": 192,
        "model.ft_transformer.ffn_d_hidden": 384,
        "optimization.learning_rate": 1e-4,
        "optimization.weight_decay": 1e-5
    },
]
search_space = {}


def generate_configs_ftt(num_random_configs=0):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
