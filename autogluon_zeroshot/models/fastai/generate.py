from autogluon.core import Real, Int, Categorical

from ...utils.config_utils import ConfigGenerator


name = 'NeuralNetFastAI'
manual_configs = [
    {},
]
search_space = {
    # See docs: https://docs.fast.ai/tabular.models.html
    'layers': Categorical(None, [200], [400], [200, 100], [400, 200], [800, 400], [200, 100, 50], [400, 200, 100]),
    'emb_drop': Real(0.0, 0.5, default=0.1),
    'ps': Real(0.0, 0.5, default=0.1),
    'bs': Categorical(256, 128, 512, 1024, 2048),
    'lr': Real(5e-4, 1e-1, default=1e-2, log=True),
    'epochs': Int(lower=20, upper=50, default=30),
}


def generate_configs_fastai(num_random_configs=100):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
