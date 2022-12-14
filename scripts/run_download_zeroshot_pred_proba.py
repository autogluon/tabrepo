import time
from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon_zeroshot.loaders import Paths


def download_zs_metadata(
        path_prefix_in,
        path_prefix_out,
        name_in_gt,
        name_in_pred_proba,
        name_out_gt=None,
        name_out_pred_proba=None):
    if name_out_gt is None:
        name_out_gt = name_in_gt
    if name_out_pred_proba is None:
        name_out_pred_proba = name_in_pred_proba
    path_gt = path_prefix_in + name_in_gt
    path_pred_proba = path_prefix_in + name_in_pred_proba
    save_path_gt = str(path_prefix_out / name_out_gt)
    save_path_pred_proba = str(path_prefix_out / name_out_pred_proba)

    ts = time.time()
    print(f'Downloading Ground Truth File: {path_gt}')
    zeroshot_gt = load_pkl.load(path_gt)
    print(f'Downloaded Ground Truth File... {round(time.time() - ts, 2)}s')

    ts = time.time()
    print(f'Saving Ground Truth File: {save_path_gt}')
    save_pkl.save(path=save_path_gt, object=zeroshot_gt)
    print(f'Saved Ground Truth File... {round(time.time() - ts, 2)}s')

    # WARNING: This file is 98 GB on disk. Ensure your machine has >250 GB of memory to avoid crashing.
    #  Additionally, ensure your internet connection is strong, otherwise this will take an extremely long time.
    ts = time.time()
    print(f'Downloading Pred Proba File: {path_pred_proba}')
    zeroshot_pred_proba = load_pkl.load(path_pred_proba)
    print(f'Downloaded Pred Proba File... {round(time.time() - ts, 2)}s')

    ts = time.time()
    print(f'Saving Pred Proba File: {save_path_pred_proba}')
    save_pkl.save(path=save_path_pred_proba, object=zeroshot_pred_proba)
    print(f'Saved Pred Proba File... {round(time.time() - ts, 2)}s')


if __name__ == '__main__':
    path_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2022_12_11_zs/'
    path_gt_name = 'zeroshot_gt_2022_12_11_zs_mini.pkl'
    path_pred_proba_name = 'zeroshot_pred_proba_2022_12_11_zs_mini.pkl'

    download_zs_metadata(
        path_prefix_in=path_prefix,
        path_prefix_out=Paths.bagged_results_root / "all",
        name_in_gt=path_gt_name,
        name_in_pred_proba=path_pred_proba_name,
    )
