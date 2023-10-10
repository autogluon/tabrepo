import numpy as np

from tabrepo.contexts.context_artificial import load_context_artificial
from tabrepo.repository.evaluation_repository import EvaluationRepository
from tabrepo.repository.time_utils import get_runtime, filter_configs_by_runtime, sort_by_runtime

zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
repo = EvaluationRepository(
    zeroshot_context=zsc,
    tabular_predictions=zeroshot_pred_proba,
    ground_truth=zeroshot_gt,
)


def test_get_runtime():

    config_names = repo.list_models()
    runtime_dict = get_runtime(
        repo,
        tid=359944,
        fold=1,
        config_names=config_names,
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [1.0, 2.0])


def test_get_runtime_time_infer_s():
    config_names = repo.list_models()
    runtime_dict = get_runtime(
        repo,
        tid=359944,
        fold=1,
        config_names=config_names,
        runtime_col='time_infer_s',
    )
    assert list(runtime_dict.keys()) == config_names
    assert np.allclose(list(runtime_dict.values()), [2.0, 4.0])


def test_sort_by_runtime():
    config_names = repo.list_models()
    assert sort_by_runtime(repo, config_names) == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']


def test_filter_configs_by_runtime():
    config_names = repo.list_models()
    for max_cumruntime, num_config_expected in [
        (None, len(config_names)),
        (0, len(config_names)),
        (0.5, 0),
        (2.0, 1),
        (3.01, len(config_names)),
        (6.0, len(config_names)),
        (np.inf, len(config_names))
    ]:
        selected_configs = filter_configs_by_runtime(
            repo,
            tid=359944,
            fold=1,
            config_names=config_names,
            max_cumruntime=max_cumruntime,
        )
        assert selected_configs == config_names[:num_config_expected]
