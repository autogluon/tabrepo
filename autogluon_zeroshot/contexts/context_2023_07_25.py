from .context import BenchmarkContext
from ..loaders import Paths


result_prefix = 'results/2023_07_25/'
_s3_download_map = {
    "evaluation/compare/results_ranked_by_dataset_valid.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/evaluation/compare/results_ranked_by_dataset_valid.csv",
    "evaluation/configs/results_ranked_by_dataset_all.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/evaluation/configs/results_ranked_by_dataset_all.csv",
    "leaderboard_preprocessed_configs.csv": "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/leaderboard_preprocessed_configs.csv",
    "zeroshot_gt_50_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/zeroshot_gt_50_mb.pkl",
    "zeroshot_pred_proba_50_mb.pkl": "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/zeroshot_pred_proba_50_mb.pkl"
}
_s3_download_map = {f'{result_prefix}{k}': v for k, v in _s3_download_map.items()}
_s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in _s3_download_map.items()}

_path_bagged_root = Paths.bagged_2023_07_25_results_root

_result_paths = dict(
    results_by_dataset=str(_path_bagged_root / "evaluation/configs/results_ranked_by_dataset_all.csv"),
    comparison=str(_path_bagged_root / "evaluation/compare/results_ranked_by_dataset_valid.csv"),
    raw=str(_path_bagged_root / "leaderboard_preprocessed_configs.csv"),
)

_task_metadata_path = dict(
    task_metadata=str(Paths.data_root / "metadata" / "task_metadata_244.csv"),
)

_bag_zs_50_mb_path = dict(
    zs_pp=str(_path_bagged_root / f'zeroshot_pred_proba_50_mb.pkl'),
    zs_gt=str(_path_bagged_root / f'zeroshot_gt_50_mb.pkl'),
)

context_bag_2023_07_25_50_mb: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D244_F1_C1416',
    description='TMP',
    date='2023_07_25',
    folds=[0],
    s3_download_map=_s3_download_map,
    **_result_paths,
    **_bag_zs_50_mb_path,
    **_task_metadata_path,
)
