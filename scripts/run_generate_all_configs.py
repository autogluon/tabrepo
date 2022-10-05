from autogluon.common.savers import save_json

from autogluon_zeroshot.models.lightgbm.generate import generate_configs_lightgbm
from autogluon_zeroshot.models.catboost.generate import generate_configs_catboost
from autogluon_zeroshot.models.xgboost.generate import generate_configs_xgboost
from autogluon_zeroshot.models.fastai.generate import generate_configs_fastai
from autogluon_zeroshot.models.nn_torch.generate import generate_configs_nn_torch
from autogluon_zeroshot.models.knn.generate import generate_configs_knn
from autogluon_zeroshot.models.random_forest.generate import generate_configs_random_forest
from autogluon_zeroshot.models.extra_trees.generate import generate_configs_extra_trees


if __name__ == '__main__':
    configs_lightgbm = generate_configs_lightgbm()
    configs_catboost = generate_configs_catboost()
    configs_xgboost = generate_configs_xgboost()
    configs_fastai = generate_configs_fastai()
    configs_nn_torch = generate_configs_nn_torch()
    configs_knn = generate_configs_knn()
    configs_rf = generate_configs_random_forest()
    configs_xt = generate_configs_extra_trees()

    save_json.save(path='../data/configs/configs_lightgbm.json', obj=configs_lightgbm)
    save_json.save(path='../data/configs/configs_catboost.json', obj=configs_catboost)
    save_json.save(path='../data/configs/configs_xgboost.json', obj=configs_xgboost)
    save_json.save(path='../data/configs/configs_fastai.json', obj=configs_fastai)
    save_json.save(path='../data/configs/configs_nn_torch.json', obj=configs_nn_torch)
    save_json.save(path='../data/configs/configs_knn.json', obj=configs_knn)
    save_json.save(path='../data/configs/configs_rf.json', obj=configs_rf)
    save_json.save(path='../data/configs/configs_xt.json', obj=configs_xt)
