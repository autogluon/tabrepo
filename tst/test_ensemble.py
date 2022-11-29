# File to check that we get the same error as run_evaluate_config.py. It can be removed.
import pytest

from autogluon_zeroshot.contexts.context_2022_10_13 import get_configs_small
from autogluon_zeroshot.utils import catchtime
from scripts.method_comparison.evaluate_ensemble import evaluate_ensemble


@pytest.mark.parametrize("backend", ["native", "ray"])
def test_ensemble_computation(backend):
    with catchtime("eval"):
        configs = get_configs_small()[:10]
        datasets = ['146818_0', '146818_1', '146820_0', '146820_1']

        error, _ = evaluate_ensemble(
            configs=configs,
            train_datasets=datasets,
            test_datasets=[],
            ensemble_size=5,
            backend=backend,
        )
        assert error == 4.75