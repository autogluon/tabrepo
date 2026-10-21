"""FTFM's ensemble classifier with a network loader on the top-level estimator and a deferred fit.

Developer fix. ``ftfm.FTFMClassifier`` (ftfm 0.1.0, commit 7e5e237) is an ensemble that holds its
network on a nested ``member`` estimator; only the member has ``_load_model``. AutoGluon's shared
weights rebuild a reloaded model's network by calling the declared loader method on the top-level
estimator, so a weightless pickle loaded into a process without the network failed with
``AttributeError``. This subclass adds ``_load_model`` that delegates to the member. The library
could offer the same method on ``_FTFMEnsemble`` itself; once it does, that part can go.

Developer fix. On CUDA the library prefills up to eight views in one batch and keeps each member's
context as a slice of the batched tensors (``context_slice``). A pickle writes a tensor's whole
storage, so every member carried all members' contexts: the fitted model pickled at eight times its
size (34 GB and growing for a 22k-row table), filling the worker disk and, through AutoGluon's
in-memory pickle that sizes the model, its RAM. ``fit`` here copies each slice into its own storage,
bit-exactly, as the library already does before spilling a context to disk (``ftfm.ensemble._owned``).
The library could do that in ``_prefill_group_once``; once it does, that part can go.

``defer_fit`` is the wrapper's uncached mode (``cache_context=False``). FTFM's ``fit`` builds the
fitted contexts, about 68 KB per support row per ensemble member, which only the next predictions
read. A bagged fold model predicts its out-of-fold rows once and is dropped, so it keeps the support
rows instead and builds the contexts inside that predict, releasing them right after.
"""

from __future__ import annotations

import torch
from ftfm import FTFMClassifier as _FTFMEnsembleClassifier


def _own_storage(value):
    """``value`` with every tensor that views a larger storage copied into a storage of its own."""
    if torch.is_tensor(value):
        if value.untyped_storage().nbytes() != value.numel() * value.element_size():
            return value.clone()
        return value
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_own_storage(v) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_own_storage(v) for v in value)
    if isinstance(value, dict):
        return {k: _own_storage(v) for k, v in value.items()}
    return value


class FTFMClassifier(_FTFMEnsembleClassifier):
    """``ftfm.FTFMClassifier`` plus a top-level ``_load_model`` and a deferred fit."""

    def _load_model(self) -> None:
        self.member._load_model()

    def fit(self, X, y, cat_features=None) -> FTFMClassifier:
        super().fit(X, y, cat_features)
        for estimator in (self, *(getattr(self, "ecoc_members_", None) or ())):
            if getattr(estimator, "contexts_", None) is not None:
                estimator.contexts_ = [_own_storage(context) for context in estimator.contexts_]
        return self

    def defer_fit(self, X, y) -> FTFMClassifier:
        """Load the network now and keep the support rows; every predict call fits on them first."""
        self.member._ensure_model()
        self.deferred_support_ = (X, y)
        return self

    def predict_proba(self, X, batch_size=None):
        support = getattr(self, "deferred_support_", None)
        if support is None:
            return super().predict_proba(X, batch_size)
        self.fit(*support)
        try:
            return super().predict_proba(X, batch_size)
        finally:
            self._invalidate_fit()
