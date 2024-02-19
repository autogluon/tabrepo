from .context import BenchmarkContext, construct_context
from ..loaders import Paths
from .context_2023_08_21 import datasets


s3_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_11_14/'

folds = [0, 1, 2]
local_prefix = "2023_11_14"
date = "2023_11_14"
metadata_join_column = "tid"

configs_prefix = Paths.data_root / 'configs'
configs = [
    f'{configs_prefix}/configs_catboost.json',
    f'{configs_prefix}/configs_fastai.json',
    f'{configs_prefix}/configs_lightgbm.json',
    f'{configs_prefix}/configs_nn_torch.json',
    f'{configs_prefix}/configs_xgboost.json',
    f'{configs_prefix}/configs_rf.json',
    f'{configs_prefix}/configs_xt.json',
    f'{configs_prefix}/configs_ftt.json',
    f'{configs_prefix}/configs_lr.json',
    f'{configs_prefix}/configs_knn.json',
    f'{configs_prefix}/configs_tabpfn.json',
]

kwargs = dict(
    local_prefix=local_prefix,
    s3_prefix=s3_prefix,
    folds=folds,
    date=date,
    task_metadata="task_metadata_244.csv",
    metadata_join_column=metadata_join_column,
    configs_hyperparameters=configs,
)


D244_F3_REBUTTAL: BenchmarkContext = construct_context(
    name="D244_F3_REBUTTAL",
    description="FTT Test",
    datasets=datasets,
    has_baselines=True,
    **kwargs,
)

D244_F3_REBUTTAL_200: BenchmarkContext = construct_context(
    name="D244_F3_REBUTTAL_200",
    description="FTT Test (200 smallest datasets)",
    datasets=datasets[-200:],
    has_baselines=True,
    **kwargs,
)

D244_F3_REBUTTAL_100: BenchmarkContext = construct_context(
    name="D244_F3_REBUTTAL_100",
    description="FTT Test (100 smallest datasets)",
    datasets=datasets[-100:],
    has_baselines=True,
    **kwargs,
)

D244_F3_REBUTTAL_30: BenchmarkContext = construct_context(
    name="D244_F3_REBUTTAL_30",
    description="FTT Test (30 smallest datasets)",
    datasets=datasets[-30:],
    has_baselines=True,
    **kwargs,
)

D244_F3_REBUTTAL_10: BenchmarkContext = construct_context(
    name="D244_F3_REBUTTAL_10",
    description="FTT Test (10 smallest datasets)",
    datasets=datasets[-10:],
    has_baselines=True,
    **kwargs,
)
