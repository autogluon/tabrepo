from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

logger = logging.getLogger(__name__)

# Checkpoints on the public HF repo, under the same names as the model entry's. The regression
# checkpoint was replaced on the Hub on 2026-09-28: the model entry pins the revision before it,
# this system the one after.
_HF_REPO = "occams/Xiaomi-TabLDM"
_CLASSIFIER_CHECKPOINT = "checkpoints/clf_default.ckpt"
_REGRESSOR_CHECKPOINT = "checkpoints/reg_default.ckpt"
#: Hub revision holding the checkpoints this system is benchmarked with. The upstream loader
#: downloads from ``main``, whose weights change, so the fit hands the estimator a pinned file.
_HF_REVISION = "b8364615b202cd823833395ec12c644d85895089"


def _checkpoint_path(filename: str) -> Path:
    """Download ``filename`` at the pinned Hub revision (a no-op once cached) and return its path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=_HF_REPO, filename=filename, revision=_HF_REVISION))


class TabLDMPlusSystemModel(ExternalSystemModel):
    """Xiaomi-TabLDM+: TabLDM run with candidate enhancement, benchmarked as a system.

    The enhanced estimator fits its candidate/ensemble weights on a hold-out split it carves out
    of ``X``/``y`` itself (``validation=True``), which is what makes this a system rather than a
    model: TabArena's model protocol hands the wrapper a validation split of its own and does not
    permit a second internal holdout, while :class:`ExternalSystemModel` documents that a system
    "carves its own internal validation from ``X``/``y`` if it wants one".

    Init hyperparameters (each a per-config knob for the system generator):

    * ``enhance_candidates``: ``True`` (default) runs the candidate-enhancement path; ``False``
      falls back to upstream's plain single-group inference with no holdout weights.
    * ``validation``: whether the enhanced path may carve its hold-out split.
    * ``n_estimators``: base foundation-model forward passes per group; ``None`` keeps upstream's
      default. Exposed so a smoke fit can drop it to 1.

    The device follows the split's GPU budget, the seed is the per-split ``random_state`` and
    ``time_limit`` is the estimator's fit budget: it stops building candidates when the budget runs
    out and leaves the ones it never ran out of the fusion.

    Codebase: https://github.com/xiaomi-research/xiaomi-tabldm (pip name ``Xiaomi-TabLDM``, import
    name ``tabldm``, not on PyPI), at the same commit as the ``Xiaomi-TabLDM`` model entry.
    Checkpoints: https://huggingface.co/occams/Xiaomi-TabLDM
    License: Apache-2.0 (Copyright Xiaomi Corporation)
    """

    warmup_modules: ClassVar[tuple[str, ...]] = ("tabldm",)
    warmup_torch_device: ClassVar[bool] = True

    def __init__(
        self,
        *,
        enhance_candidates: bool = True,
        validation: bool = True,
        n_estimators: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enhance_candidates = enhance_candidates
        self.validation = validation
        self.n_estimators = n_estimators
        self._estimator = None

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        """Neither TabLDM nor this wrapper starts Ray: the estimator fits and predicts in-process."""
        return False

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        """Fit one enhanced TabLDM estimator on all the training data.

        The frames are passed through raw: the estimator ordinal-encodes categorical dtypes,
        mean-imputes numeric NaNs and standardizes a regression target itself. The estimator's
        budget clock starts inside its ``fit``, so it gets what is left of ``time_limit`` after
        the checkpoint resolve. ``memory_limit`` has no estimator knob to bind to.
        """
        from tabldm import TabLDMClassifier, TabLDMRegressor

        start = time.monotonic()
        is_classification = problem_type in ("binary", "multiclass")
        model_path = _checkpoint_path(_CLASSIFIER_CHECKPOINT if is_classification else _REGRESSOR_CHECKPOINT)
        model_cls = TabLDMClassifier if is_classification else TabLDMRegressor
        extra = {} if self.n_estimators is None else {"n_estimators": self.n_estimators}
        self._estimator = model_cls(
            model_path=model_path,
            enhance_candidates=self.enhance_candidates,
            validation=self.validation,
            # No GPU budget (a direct fit outside the runner): the estimator picks CUDA when visible.
            device=None if num_gpus is None else ("cuda" if num_gpus > 0 else "cpu"),
            n_jobs=num_cpus,
            random_state=0 if random_state is None else random_state,
            **extra,
        )
        if time_limit is not None:
            time_limit = max(time_limit - (time.monotonic() - start), 0.0)
        logger.info(f"Xiaomi-TabLDM+: fitting {problem_type} on X={X.shape} with time_limit={time_limit}")
        self._estimator.fit(X, y, time_limit=time_limit)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._estimator.predict(X), index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # `classes_` is the original label space (the estimator inverse-transforms its own internal
        # encoding), so the columns line up with the task's labels.
        return pd.DataFrame(self._estimator.predict_proba(X), index=X.index, columns=self._estimator.classes_)

    def cleanup(self):
        """Drop the fitted estimator once the post-evaluate consumers are done with it."""
        self._estimator = None


def prefetch_weights() -> None:
    """Download both checkpoints at the pinned Hub revision (``SystemInfo.prefetch_weights``).

    The system's own tooling hook: the benchmark setup does not call it.
    """
    for filename in (_CLASSIFIER_CHECKPOINT, _REGRESSOR_CHECKPOINT):
        _checkpoint_path(filename)
