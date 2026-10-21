"""FTFM's ensemble classifier with the network loader reachable from the top-level estimator.

Developer fix. ``ftfm.FTFMClassifier`` (ftfm 0.1.0, commit 7e5e237) is an ensemble that holds its
network on a nested ``member`` estimator; only the member has ``_load_model``. AutoGluon's shared
weights rebuild a reloaded model's network by calling the declared loader method on the top-level
estimator, so a weightless pickle loaded into a process without the network failed with
``AttributeError``. This subclass adds ``_load_model`` that delegates to the member. The library
could offer the same method on ``_FTFMEnsemble`` itself; once it does, drop this module and use
``ftfm.FTFMClassifier`` directly.
"""

from __future__ import annotations

from ftfm import FTFMClassifier as _FTFMEnsembleClassifier


class FTFMClassifier(_FTFMEnsembleClassifier):
    """``ftfm.FTFMClassifier`` plus a ``_load_model`` that builds the member's network."""

    def _load_model(self) -> None:
        self.member._load_model()
