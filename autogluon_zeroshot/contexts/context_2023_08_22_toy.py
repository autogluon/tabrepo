from .context import BenchmarkContext
from ..loaders import Paths


result_prefix = 'results/2023_08_22_toy/'
result_s3_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_08_22_toy/'
_s3_download_map = {
    "evaluation/compare/results_ranked_by_dataset_valid.csv": "evaluation/compare/results_ranked_by_dataset_valid.csv",
    "evaluation/configs/results_ranked_by_dataset_all.csv": "evaluation/configs/results_ranked_by_dataset_all.csv",
    "leaderboard_preprocessed_configs.csv": "leaderboard_preprocessed_configs.csv",
}
_s3_download_map = {f'{result_prefix}{k}': f'{result_s3_prefix}{v}' for k, v in _s3_download_map.items()}

split_key = "results/2023_08_22_toy/zeroshot_metadata/"
split_value = "s3://automl-benchmark-ag/aggregated/ec2/2023_08_22_toy/zeroshot_metadata/"
_files = [
    '2dplanes/2/zeroshot_gt.pkl',
    '2dplanes/2/zeroshot_pred_proba.pkl',
    'Australian/1/zeroshot_gt.pkl',
    'Australian/1/zeroshot_pred_proba.pkl',
    'Australian/2/zeroshot_gt.pkl',
    'Australian/2/zeroshot_pred_proba.pkl',
    'Bioresponse/1/zeroshot_gt.pkl',
    'Bioresponse/1/zeroshot_pred_proba.pkl',
    'Bioresponse/2/zeroshot_gt.pkl',
    'Bioresponse/2/zeroshot_pred_proba.pkl',
    'Brazilian_houses/1/zeroshot_gt.pkl',
    'Brazilian_houses/1/zeroshot_pred_proba.pkl',
    'Brazilian_houses/2/zeroshot_gt.pkl',
    'Brazilian_houses/2/zeroshot_pred_proba.pkl',
]
_s3_download_map_metadata = {f"{split_key}{f}": f"{split_value}{f}" for f in _files}
_s3_download_map.update(_s3_download_map_metadata)

_files_pp = [f for f in _files if "_pred_proba.pkl" in f]  # FIXME: HACK
_s3_download_metadata = [f"{split_key}{f}" for f in _files_pp]
_s3_download_metadata = [Paths.rel_to_abs(k, relative_to=Paths.data_root) for k in _s3_download_metadata]

_files_gt = [f for f in _files if "_gt.pkl" in f]  # FIXME: HACK
_s3_download_metadata_gt = [f"{split_key}{f}" for f in _files_gt]
_s3_download_metadata_gt = [Paths.rel_to_abs(k, relative_to=Paths.data_root) for k in _s3_download_metadata_gt]

_s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in _s3_download_map.items()}

_path_bagged_root = Paths.bagged_2023_08_22_toy_results_root

_result_paths = dict(
    results_by_dataset=str(_path_bagged_root / "evaluation/configs/results_ranked_by_dataset_all.csv"),
    comparison=str(_path_bagged_root / "evaluation/compare/results_ranked_by_dataset_valid.csv"),
    raw=str(_path_bagged_root / "leaderboard_preprocessed_configs.csv"),
)

_task_metadata_path = dict(
    task_metadata=str(Paths.data_root / "metadata" / "task_metadata_244.csv"),
)

_bag_zs_200_mb_path = dict(
    zs_pp=_s3_download_metadata,
    zs_gt=_s3_download_metadata_gt,
)

context_bag_2023_08_22_toy: BenchmarkContext = BenchmarkContext.from_paths(
    name='BAG_D244_F2_C32',
    description='Toy',
    date='2023_08_22_toy',
    folds=[1, 2],
    s3_download_map=_s3_download_map,
    **_result_paths,
    **_bag_zs_200_mb_path,
    **_task_metadata_path,
)
