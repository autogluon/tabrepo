from .context import BenchmarkContext, construct_context
from ..loaders import Paths
from .context_2023_08_21 import datasets


# s3_prefix = 's3://automl-benchmark-ag/aggregated/ec2/2023_11_14/'
s3_prefix = "https://tabrepo.s3.us-west-2.amazonaws.com/contexts/2023_11_14/"

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


D244_F3_C1530: BenchmarkContext = construct_context(
    name="D244_F3_C1530",
    description="Large-scale Benchmark on 244 datasets and 3 folds (330 GB, 211 datasets)",
    datasets=datasets,
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_200: BenchmarkContext = construct_context(
    name="D244_F3_C1530_200",
    description="Large-scale Benchmark on 244 datasets and 3 folds (120 GB, 200 smallest datasets)",
    datasets=datasets[-200:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_100: BenchmarkContext = construct_context(
    name="D244_F3_C1530_100",
    description="Large-scale Benchmark on 244 datasets and 3 folds (9.5 GB, 100 smallest datasets)",
    datasets=datasets[-100:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_30: BenchmarkContext = construct_context(
    name="D244_F3_C1530_30",
    description="Large-scale Benchmark on 244 datasets and 3 folds (1.1 GB, 30 smallest datasets)",
    datasets=datasets[-30:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_10: BenchmarkContext = construct_context(
    name="D244_F3_C1530_10",
    description="Large-scale Benchmark on 244 datasets and 3 folds (220 MB, 10 smallest datasets)",
    datasets=datasets[-10:],
    has_baselines=True,
    **kwargs,
)

D244_F3_C1530_3: BenchmarkContext = construct_context(
    name="D244_F3_C1530_3",
    description="Large-scale Benchmark on 244 datasets and 3 folds (33 MB, 3 smallest datasets)",
    datasets=datasets[-3:],
    has_baselines=True,
    **kwargs,
)
