from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl


if __name__ == '__main__':
    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2022_10_13_zs/'
    path_gt_name = 'zeroshot_gt_2022_10_13_zs.pkl'
    path_pred_proba_name = 'zeroshot_pred_proba_2022_10_13_zs.pkl'
    path_gt = path_prefix + path_gt_name
    path_pred_proba = path_prefix + path_pred_proba_name

    zeroshot_gt = load_pkl.load(path_gt)
    zeroshot_pred_proba = load_pkl.load(path_pred_proba)

    save_path = Path(__file__).parent.parent / 'data' / 'results' / 'all_v3'

    save_pkl.save(path=str(save_path / path_gt_name), object=zeroshot_gt)
    save_pkl.save(path=str(save_path / path_pred_proba_name), object=zeroshot_pred_proba)
