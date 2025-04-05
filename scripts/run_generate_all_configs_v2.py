from tabrepo.models.lightgbm.generate import generate_experiments_lightgbm
from tabrepo.models.tabicl.generate import generate_experiments_tabicl
from tabrepo.benchmark.experiment import YamlExperimentSerializer


if __name__ == '__main__':
    experiments_lightgbm = generate_experiments_lightgbm(num_random_configs=10)
    experiments_tabicl = generate_experiments_tabicl(num_random_configs=10)

    experiments_all = experiments_lightgbm + experiments_tabicl

    YamlExperimentSerializer.to_yaml(experiments=experiments_all, path="configs_all.yaml")
