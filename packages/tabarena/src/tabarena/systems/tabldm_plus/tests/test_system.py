"""Xiaomi-TabLDM+ fit wiring against a fake ``tabldm`` (no checkpoint download, no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.systems.tabldm_plus import system as system_mod
from tabarena.systems.tabldm_plus.system import TabLDMPlusSystemModel


class _FakeEstimator:
    """Records its init kwargs and the ``time_limit`` its ``fit`` got."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def fit(self, X, y, *, time_limit=None):
        self.fit_time_limit = time_limit
        return self


@pytest.fixture
def clock(monkeypatch):
    """Fake ``tabldm`` and a checkpoint resolve that takes ``clock.download_s`` seconds of fake time."""
    clock = types.SimpleNamespace(now=100.0, download_s=0.0)
    fake_tabldm = types.ModuleType("tabldm")
    fake_tabldm.TabLDMClassifier = fake_tabldm.TabLDMRegressor = _FakeEstimator
    monkeypatch.setitem(sys.modules, "tabldm", fake_tabldm)
    monkeypatch.setattr(system_mod, "time", types.SimpleNamespace(monotonic=lambda: clock.now))

    def fake_checkpoint_path(filename: str) -> Path:
        clock.now += clock.download_s
        return Path("/fake") / filename

    monkeypatch.setattr(system_mod, "_checkpoint_path", fake_checkpoint_path)
    return clock


def _fit(problem_type: str = "binary", **fit_kwargs) -> _FakeEstimator:
    metadata = ValidationMetadata(target_name="t")
    model = TabLDMPlusSystemModel(problem_type=problem_type, eval_metric=None, validation_metadata=metadata)
    kwargs = {
        "target_name": "t",
        "problem_type": problem_type,
        "eval_metric": None,
        "validation_metadata": metadata,
        "num_cpus": 4,
        "num_gpus": 1,
        "memory_limit": None,
        "time_limit": None,
        "random_state": 3,
        **fit_kwargs,
    }
    model._fit_system(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([0, 1]), **kwargs)
    return model._estimator


def test_estimator_gets_pinned_checkpoint_budget_and_seed(clock):
    assert _fit().init_kwargs == {
        "model_path": Path("/fake/checkpoints/clf_default.ckpt"),
        "enhance_candidates": True,
        "validation": True,
        "device": "cuda",
        "n_jobs": 4,
        "random_state": 3,
    }
    assert _fit("regression").init_kwargs["model_path"] == Path("/fake/checkpoints/reg_default.ckpt")


@pytest.mark.parametrize(("num_gpus", "device"), [(None, None), (0, "cpu"), (1, "cuda")])
def test_device_follows_gpu_budget(clock, num_gpus, device):
    assert _fit(num_gpus=num_gpus).init_kwargs["device"] == device


@pytest.mark.parametrize(
    ("time_limit", "download_s", "expected"),
    [(None, 45.0, None), (600, 0.0, 600), (600, 45.0, 555), (60, 500.0, 0.0)],
)
def test_checkpoint_resolve_is_charged_to_the_fit_budget(clock, time_limit, download_s, expected):
    clock.download_s = download_s
    assert _fit(time_limit=time_limit).fit_time_limit == expected
