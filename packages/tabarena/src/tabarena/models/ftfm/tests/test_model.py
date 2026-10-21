from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ftfm")

from tabarena.models.ftfm.model import FTFMModel


def _frames(n_rows: int = 160, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=list("abcd"))
    X["cat"] = pd.Series(rng.choice(["u", "v", "w"], size=n_rows), dtype="category")
    y = pd.Series((X["a"] + (X["cat"] == "u") > 0).astype(int))
    return X.iloc[:120], y.iloc[:120], X.iloc[120:]


def _fit(tmp_path, name: str, X, y, cache_context: bool) -> FTFMModel:
    model = FTFMModel(
        path=str(tmp_path / name),
        name=name,
        problem_type="binary",
        hyperparameters={"n_estimators": 2, "cache_context": cache_context},
    )
    model.fit(X=X, y=y, num_cpus=2, num_gpus=0)
    return model


def test_uncached_predicts_like_cached_and_keeps_no_context(tmp_path):
    X, y, X_test = _frames()
    cached = _fit(tmp_path, "cached", X, y, cache_context=True)
    uncached = _fit(tmp_path, "uncached", X, y, cache_context=False)

    assert cached.model.contexts_ is not None
    assert getattr(uncached.model, "contexts_", None) is None
    np.testing.assert_allclose(uncached.predict_proba(X_test), cached.predict_proba(X_test), rtol=1e-5, atol=1e-6)
    # The contexts built for that predict are released again; a second call rebuilds them.
    assert uncached.model.contexts_ is None
    np.testing.assert_allclose(uncached.predict_proba(X_test), cached.predict_proba(X_test), rtol=1e-5, atol=1e-6)


def test_view_batched_contexts_pickle_at_their_own_size():
    """On CUDA the library prefills views in one batch; each member's context must not carry the others."""
    import pickle

    from tabarena.models.ftfm._estimators import FTFMClassifier

    X, y, X_test = _frames(n_rows=300)
    kwargs = {"device": "cpu", "model_path": FTFMModel.prefetch_weights(), "n_estimators": 4}
    estimator = FTFMClassifier(view_batch_size=4, offload_contexts=True, **kwargs).fit(X, y)
    reference = FTFMClassifier(view_batch_size=1, **kwargs).fit(X, y)

    np.testing.assert_allclose(estimator.predict_proba(X_test), reference.predict_proba(X_test), rtol=1e-5, atol=1e-6)
    estimator.member.model = reference.member.model = None
    assert len(pickle.dumps(estimator)) < 1.1 * len(pickle.dumps(reference))
