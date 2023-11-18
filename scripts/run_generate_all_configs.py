from pathlib import Path
from autogluon.common.savers import save_json

from tabrepo.models.lightgbm.generate import generate_configs_lightgbm
from tabrepo.models.catboost.generate import generate_configs_catboost
from tabrepo.models.xgboost.generate import generate_configs_xgboost
from tabrepo.models.fastai.generate import generate_configs_fastai
from tabrepo.models.nn_torch.generate import generate_configs_nn_torch
from tabrepo.models.knn.generate import generate_configs_knn
from tabrepo.models.random_forest.generate import generate_configs_random_forest
from tabrepo.models.extra_trees.generate import generate_configs_extra_trees
from tabrepo.models.lr.generate import generate_configs_lr
from tabrepo.models.tabpfn.generate import generate_configs_tabpfn


if __name__ == '__main__':
    configs_lightgbm = generate_configs_lightgbm()
    configs_catboost = generate_configs_catboost()
    configs_xgboost = generate_configs_xgboost()
    configs_fastai = generate_configs_fastai()
    configs_nn_torch = generate_configs_nn_torch()
    configs_knn = generate_configs_knn()
    configs_rf = generate_configs_random_forest()
    configs_xt = generate_configs_extra_trees()
    configs_lr = generate_configs_lr()
    configs_tabpfn = generate_configs_tabpfn()
    json_root = Path(__file__).parent.parent / 'data' / 'configs'
    save_json.save(path=str(json_root / 'configs_lightgbm.json'), obj=configs_lightgbm)
    save_json.save(path=str(json_root / 'configs_catboost.json'), obj=configs_catboost)
    save_json.save(path=str(json_root / 'configs_xgboost.json'), obj=configs_xgboost)
    save_json.save(path=str(json_root / 'configs_fastai.json'), obj=configs_fastai)
    save_json.save(path=str(json_root / 'configs_nn_torch.json'), obj=configs_nn_torch)
    save_json.save(path=str(json_root / 'configs_knn.json'), obj=configs_knn)
    save_json.save(path=str(json_root / 'configs_rf.json'), obj=configs_rf)
    save_json.save(path=str(json_root / 'configs_xt.json'), obj=configs_xt)
    save_json.save(path=str(json_root / 'configs_lr.json'), obj=configs_lr)
    save_json.save(path=str(json_root / 'configs_tabpfn.json'), obj=configs_tabpfn)
