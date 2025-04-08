from autogluon.core.models import DummyModel

from tabrepo.utils.config_utils import ConfigGenerator
from tabrepo.models.lightgbm.generate import gen_lightgbm
from tabrepo.models.tabicl.generate import gen_tabicl
from tabrepo.models.lightgbm.generate_alt import gen_lightgbm_alt
from tabrepo.models.knn.generate import gen_knn
from tabrepo.models.fastai.generate import gen_fastai
from tabrepo.models.extra_trees.generate import gen_extratrees
from tabrepo.models.catboost.generate import gen_catboost
from tabrepo.models.catboost.generate_alt import gen_catboost_alt
from tabrepo.models.xgboost.generate import gen_xgboost
from tabrepo.models.xgboost.generate_alt import gen_xgboost_alt
from tabrepo.models.nn_torch.generate import gen_nn_torch
from tabrepo.models.random_forest.generate import gen_randomforest
from tabrepo.models.lr.generate import gen_linear
from tabrepo.models.ftt.generate import gen_fttransformer
from tabrepo.models.tabpfn_v2.generate import gen_tabpfnv2
from tabrepo.models.realmlp.generate import gen_realmlp
from tabrepo.benchmark.experiment import YamlExperimentSerializer


if __name__ == '__main__':
    n_random_configs = 3



    # Original Search Space
    experiments_linear = gen_linear.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_knn = gen_knn.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_randomforest = gen_randomforest.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_extratrees = gen_extratrees.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_lightgbm = gen_lightgbm.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_xgboost = gen_xgboost.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_catboost = gen_catboost.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_nn_torch = gen_nn_torch.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_fttransformer = gen_fttransformer.generate_all_bag_experiments(num_random_configs=0)  # No search space defined
    # tabpfnv1?

    # New Search Space
    experiments_lightgbm_alt = gen_lightgbm_alt.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_xgboost_alt = gen_xgboost_alt.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_catboost_alt = gen_catboost_alt.generate_all_bag_experiments(num_random_configs=n_random_configs)

    # New methods
    experiments_fastai = gen_fastai.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_realmlp = gen_realmlp.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_tabicl = gen_tabicl.generate_all_bag_experiments(num_random_configs=n_random_configs)
    experiments_tabpfnv2 = gen_tabpfnv2.generate_all_bag_experiments(num_random_configs=0)  # No search space defined
    # TODO: EBM
    # TODO: TabDPT
    # TODO: TabPFNMix

    # Dummy (constant predictor)
    experiments_dummy = ConfigGenerator(model_cls=DummyModel, search_space={}, manual_configs=[{}]).generate_all_bag_experiments(num_random_configs=0)

    experiments_lst = [
        experiments_linear,
        experiments_knn,
        experiments_randomforest,
        experiments_extratrees,
        experiments_lightgbm,
        experiments_xgboost,
        experiments_catboost,
        experiments_nn_torch,
        experiments_fttransformer,

        experiments_lightgbm_alt,
        experiments_xgboost_alt,
        experiments_catboost_alt,

        experiments_fastai,
        experiments_realmlp,
        experiments_tabicl,
        experiments_tabpfnv2,

        experiments_dummy,
    ]

    experiments_all = [exp for exp_family_lst in experiments_lst for exp in exp_family_lst]

    # TODO: Verify no duplicate names, move to `to_yaml?`
    experiment_names = set()
    for experiment in experiments_all:
        if experiment.name not in experiment_names:
            experiment_names.add(experiment.name)
        else:
            raise AssertionError(f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!")

    YamlExperimentSerializer.to_yaml(experiments=experiments_all, path="configs_all.yaml")
