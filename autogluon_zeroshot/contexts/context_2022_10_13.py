from .context import BenchmarkContext
from ..loaders import Paths


_s3_download_map = {
    'results/all_v3/zeroshot_gt_2022_10_13_zs.pkl': 's3://automl-benchmark-ag/aggregated/ec2/2022_10_13_zs/zeroshot_gt_2022_10_13_zs.pkl',
    'results/all_v3/zeroshot_pred_proba_2022_10_13_zs.pkl': 's3://automl-benchmark-ag/aggregated/ec2/2022_10_13_zs/zeroshot_pred_proba_2022_10_13_zs.pkl'
}
_s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in _s3_download_map.items()}


_path_root = Paths.all_v3_results_root

_all_result_paths = dict(
    result=str(_path_root / "results_ranked_valid.csv"),
    results_by_dataset=str(_path_root / "results_ranked_by_dataset_valid.parquet"),
    raw=str(_path_root / "openml_ag_2022_10_13_zs_models.parquet"),
    comparison=str(Paths.results_root / 'automl' / 'results_ranked_by_dataset_valid.csv'),
)

_task_metadata_104_path = dict(
    task_metadata=str(Paths.data_root / "metadata" / "task_metadata.csv"),
)

_zs_path = dict(
    zs_pp=str(_path_root / 'zeroshot_pred_proba_2022_10_13_zs.pkl'),
    zs_gt=str(_path_root / 'zeroshot_gt_2022_10_13_zs.pkl'),
)

context_104_bench: BenchmarkContext = BenchmarkContext.from_paths(
    name='D104_F10_C608_FULL',
    description='Non-bagged results from 104 datasets (non-trivial), 10-fold CV, 608 configs.',
    date='2022_10_13',
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    s3_download_map=_s3_download_map,
    **_all_result_paths,
    **_zs_path,
    **_task_metadata_104_path,
)


def get_configs_default():
    autogluon_configs = [
        'CatBoost_c1',
        'LightGBM_c1',
        'LightGBM_c2',
        'LightGBM_c3',
        'NeuralNetFastAI_c1',
        'RandomForest_c1',
        'ExtraTrees_c1',
    ]
    return autogluon_configs


def get_configs_small(num_per_type=12):
    small_extra_configs = []
    for m in [
        'LightGBM',
        'CatBoost',
        'RandomForest',
        'ExtraTrees',
        'NeuralNetFastAI',
    ]:
        for i in range(1, num_per_type):
            small_extra_configs.append(m + f'_r{i}')
    small_configs = get_configs_default() + small_extra_configs
    return small_configs
