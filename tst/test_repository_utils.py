import numpy as np

from autogluon_zeroshot.repository.evaluation_repository import load
from autogluon_zeroshot.repository.time_utils import get_runtime, filter_configs_by_runtime, sort_by_runtime

# TODO replace test with artificial data
repo = load(version="BAG_D244_F10_C608_FULL")

def test_get_runtime():

    config_names = ['CatBoost_r10_BAG_L1', 'CatBoost_r46_BAG_L1', 'CatBoost_r79_BAG_L1', ]
    runtime_dict = get_runtime(
        repo,
        tid=9983,
        fold=1,
        config_names=config_names,
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [170.68276143074036, 261.43372082710266, 262.73713970184326])


def test_get_runtime_time_infer_s():
    config_names = ['CatBoost_r10_BAG_L1', 'CatBoost_r46_BAG_L1', 'CatBoost_r79_BAG_L1', ]
    runtime_dict = get_runtime(
        repo,
        tid=9983,
        fold=1,
        config_names=config_names,
        runtime_col='time_infer_s',
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [0.3531279563903808, 0.6377890110015869, 0.0613722801208496])


def test_sort_by_runtime():
    config_names = ['CatBoost_r46_BAG_L1', 'RandomForest_r6_BAG_L1', 'LightGBM_r151_BAG_L1']
    assert sort_by_runtime(repo, config_names) == ['LightGBM_r151_BAG_L1', 'RandomForest_r6_BAG_L1', 'CatBoost_r46_BAG_L1']


def test_filter_configs_by_runtime():
    config_names = [
        'CatBoost_r10_BAG_L1',
        'CatBoost_r46_BAG_L1',
        'CatBoost_r79_BAG_L1',
        'CatBoost_r58_BAG_L1',
        'CatBoost_r20_BAG_L1',
    ]

    for max_cumruntime, num_config_expected in [
        (None, len(config_names)),
        (0, len(config_names)),
        (10, 0),
        (200, 1),
        (800, 3),
        (3000, len(config_names)),
        (np.inf, len(config_names))
    ]:
        selected_configs = filter_configs_by_runtime(
            repo,
            tid=9983,
            fold=1,
            config_names=config_names,
            max_cumruntime=max_cumruntime,
        )
        assert selected_configs == config_names[:num_config_expected]
