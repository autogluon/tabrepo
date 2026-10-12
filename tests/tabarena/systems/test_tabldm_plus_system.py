from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.systems.tabldm_plus import system as system_mod
from tabarena.systems.tabldm_plus.system import TabLDMPlusSystemModel, _time_limit_report


class _StubCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _StubTorch:
    """Just the device probe, the only part of torch these tests touch."""

    cuda = _StubCuda()


@pytest.fixture(autouse=True)
def stub_torch(monkeypatch):
    """Keep the real torch out of these tests.

    ``_fit_system`` imports torch lazily to pick a device, which costs ~20s of import and warns
    on hosts with an older driver. These tests assert the budget wiring, not the device choice,
    so a two-attribute stub is enough and keeps the file runnable on CPU-only machines. Replace
    it if a test here ever needs a real tensor.
    """
    monkeypatch.setitem(sys.modules, "torch", _StubTorch())


class _FakeBudget:
    """Stands in for the estimator's ``_fit_budget_`` (upstream's ``_FitBudget``)."""

    def __init__(self, *, n_truncated: int = 0, elapsed: float = 1.5):
        self.n_truncated = n_truncated
        self.elapsed = elapsed


class _FakeEstimator:
    """Records the ``time_limit`` its ``fit`` was called with; no model, no data."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_time_limit = "not-fitted"
        self._fit_budget_ = _FakeBudget()

    def fit(self, X, y, *, time_limit=None):
        self.fit_time_limit = time_limit
        return self


class _FakeTime:
    """Deterministic clock: each ``monotonic()`` call advances by ``step`` seconds."""

    def __init__(self, *, start: float = 100.0, step: float = 0.0):
        self.now = start
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def strftime(self, fmt: str) -> str:  # the log line timestamps itself
        return "00:00:00"


@pytest.fixture
def patched_system(monkeypatch):
    """Route the estimator imports and the checkpoint resolve through test doubles.

    ``_checkpoint_path`` is where the pinned download lives, so a test double that advances the
    clock models the real pre-fit cost the wrapper has to charge against ``time_limit``.
    """
    state = {"estimators": [], "clock": None, "checkpoint_step": 0.0}

    def fake_import_estimators():
        return _FakeEstimator, _FakeEstimator

    def fake_checkpoint_path(filename: str) -> Path:
        if state["clock"] is not None:
            state["clock"].now += state["checkpoint_step"]
        return Path("/fake") / filename

    monkeypatch.setattr(system_mod, "_import_estimators", fake_import_estimators)
    monkeypatch.setattr(system_mod, "_checkpoint_path", fake_checkpoint_path)

    def install_clock(*, start: float = 100.0, step: float = 0.0) -> _FakeTime:
        clock = _FakeTime(start=start, step=step)
        state["clock"] = clock
        monkeypatch.setattr(system_mod, "time", clock)
        return clock

    state["install_clock"] = install_clock

    def build(**kwargs) -> TabLDMPlusSystemModel:
        model = TabLDMPlusSystemModel(
            problem_type=kwargs.pop("problem_type", "binary"),
            eval_metric=None,
            validation_metadata=ValidationMetadata(target_name="t"),
            **kwargs,
        )
        state["estimators"].append(model)
        return model

    state["build"] = build
    return state


def _fit(model: TabLDMPlusSystemModel, *, time_limit, **overrides) -> _FakeEstimator:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1], name="t")
    kwargs = {
        "target_name": "t",
        "problem_type": "binary",
        "eval_metric": None,
        "validation_metadata": ValidationMetadata(target_name="t"),
        "num_cpus": 1,
        "num_gpus": 0,
        "memory_limit": None,
        "time_limit": time_limit,
        "random_state": 0,
    }
    kwargs.update(overrides)
    model._fit_system(X, y, **kwargs)
    return model._estimator


class TestTimeLimitReport:
    """The fit-done log line says whether the budget cut anything."""

    def test_no_budget_object(self):
        class _NoBudget:
            pass

        assert _time_limit_report(_NoBudget()) == "no time_limit"

    def test_no_early_stop(self):
        class _Est:
            _fit_budget_ = _FakeBudget(n_truncated=0, elapsed=2.25)

        assert _time_limit_report(_Est()) == "no early-stop, elapsed=2.2s"

    def test_early_stop_dropped_candidates(self):
        class _Est:
            _fit_budget_ = _FakeBudget(n_truncated=7, elapsed=60.0)

        assert _time_limit_report(_Est()) == "early-stop dropped 7 candidate(s), elapsed=60.0s"


class TestTimeLimitWiring:
    """``_fit_system`` hands the estimator a budget and leaves none of it unaccounted for."""

    def test_time_limit_none_is_forwarded_as_none(self, patched_system):
        patched_system["install_clock"]()
        estimator = _fit(patched_system["build"](), time_limit=None)
        assert estimator.fit_time_limit is None

    def test_budget_is_passed_through_when_no_pre_fit_work(self, patched_system):
        patched_system["install_clock"](step=0.0)
        estimator = _fit(patched_system["build"](), time_limit=600)
        assert estimator.fit_time_limit == 600

    def test_checkpoint_resolve_is_charged_against_the_budget(self, patched_system):
        # The pinned download runs before the estimator's own clock starts, so it has to come
        # out of the budget the estimator sees.
        patched_system["install_clock"](step=0.0)
        patched_system["checkpoint_step"] = 45.0
        estimator = _fit(patched_system["build"](), time_limit=600)
        assert estimator.fit_time_limit == 555

    def test_budget_never_goes_negative(self, patched_system):
        patched_system["install_clock"](step=0.0)
        patched_system["checkpoint_step"] = 500.0
        estimator = _fit(patched_system["build"](), time_limit=60)
        assert estimator.fit_time_limit == 0.0

    def test_regression_also_forwards_the_budget(self, patched_system):
        patched_system["install_clock"](step=0.0)
        patched_system["checkpoint_step"] = 10.0
        estimator = _fit(patched_system["build"](problem_type="regression"), time_limit=120)
        assert estimator.fit_time_limit == 110

    def test_estimator_receives_the_pinned_checkpoint_path(self, patched_system):
        patched_system["install_clock"]()
        estimator = _fit(patched_system["build"](), time_limit=None)
        assert estimator.init_kwargs["model_path"] == Path("/fake/checkpoints/clf_default.ckpt")
