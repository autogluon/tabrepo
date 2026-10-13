"""The estimator kwargs ``TabLDMModel._fit`` builds, against a fake estimator (no checkpoint download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("autogluon.tabular")

from tabarena.models.tabldm import model as model_mod
from tabarena.models.tabldm.model import TabLDMModel


class _FakeEstimator:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def fit(self, X, y):
        return self


def _estimator_kwargs(monkeypatch, tmp_path, problem_type: str, params: dict) -> dict:
    model = TabLDMModel(path=str(tmp_path), name="m", problem_type=problem_type)
    monkeypatch.setattr(model, "get_model_cls", lambda: _FakeEstimator)
    monkeypatch.setattr(model, "_get_model_params", lambda: dict(params))
    monkeypatch.setattr(model, "preprocess", lambda X, **kwargs: X)
    monkeypatch.setattr(model_mod, "_checkpoint_path", lambda filename: Path("/fake") / filename)
    model._fit(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([0, 1]), num_cpus=2, num_gpus=0)
    return model.model.init_kwargs


def test_classifier_keeps_the_defaults_the_hosted_results_used(monkeypatch, tmp_path):
    kwargs = _estimator_kwargs(monkeypatch, tmp_path, "binary", {"n_estimators": 1})
    assert kwargs == {
        "device": "cpu",
        "n_jobs": 2,
        "model_path": Path("/fake/checkpoints/clf_default.ckpt"),
        "n_estimators": 1,
        "feat_shuffle_method": "latin",
        "cat_random_encode": False,
        "use_amp": "auto",
        "enhance_candidates": False,
    }


def test_candidate_enhancement_stays_off(monkeypatch, tmp_path):
    kwargs = _estimator_kwargs(monkeypatch, tmp_path, "regression", {"enhance_candidates": True})
    assert kwargs == {
        "device": "cpu",
        "n_jobs": 2,
        "model_path": Path("/fake/checkpoints/reg_default.ckpt"),
        "enhance_candidates": False,
    }
