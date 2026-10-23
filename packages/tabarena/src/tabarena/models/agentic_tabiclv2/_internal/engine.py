"""Predict with one agentic TabICLv2 recipe using the modules of ``bin.tabicl``.

Follows ``TabICLEvaluator.fit_predict`` in ``bin/tabicl/pipeline.py`` (agentic-tabular 3d7349b) on arrays in
memory, with three differences:

- the whole training set is the context (the package's dataset loader subsamples it to 40,000 rows and
  3.84M cells);
- only the query rows are predicted: for classification the evaluator also scores the training rows, for the
  postprocessing's ``fit``; its one classification postprocessing (``ProbaPostproc``, a softmax at a fixed
  temperature) does not use them;
- no metrics, and the class probabilities are widened to AutoGluon's ``n_classes`` (a bag fold can miss a
  rare class);
- attention runs in float32 on CUDA as well (``fp32_attention``).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from bin.tabicl.core import TabularDataset
from bin.tabicl.modules.ensemble import aggregate_prediction_pool
from bin.tabicl.modules.nanotabicl import MODEL_REGISTRY
from bin.tabicl.modules.nanotabicl.model import _SDPA_MAX_BATCH, TransformerBlock
from bin.tabicl.modules.postproc_clf import ClassificationTargetEncoder
from bin.tabicl.pipeline import build_evaluator_from_indices, build_members
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def torch_state(n_threads: int):
    """Use ``n_threads`` torch threads inside the block; restore the old count afterwards."""
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n_threads)
        yield
    finally:
        torch.set_num_threads(old_threads)


def _attn_fp32(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``TransformerBlock.attn`` of the package, with q, k and v kept in float32 on CUDA."""
    q, k, v = torch.nn.functional._in_projection_packed(q, k, k, self.in_proj_weight, self.in_proj_bias)
    q, k, v = (t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) for t in (q, k, v))
    q = q if self.ssmax_layer is None else self.ssmax_layer(q=q, n=k.size(-2))
    q, k = (t if self.rope is None else self.rope(t) for t in (q, k))
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    step = _SDPA_MAX_BATCH if q.is_cuda else q.shape[0]
    with sdpa_kernel(backends):
        attn_output = torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(q[i : i + step], k[i : i + step], v[i : i + step])
                for i in range(0, q.shape[0], step)
            ]
        )
    return self.out_proj(attn_output.transpose(-3, -2).flatten(-2, -1))


@contextmanager
def fp32_attention(model: torch.nn.Module):
    """Attention precision. On CUDA the package runs attention in bfloat16 for FlashAttention, its speed setting for
    the search; on CPU it stays in float32. TabArena runs the model in float32, with which the reference recipe
    reproduces TabICLv2's TabArena results. Inside the block this model's attention blocks use the float32 method (an
    instance attribute, so the pickled model and every other instance keep the package's method).
    """
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    for block in blocks:
        block.attn = types.MethodType(_attn_fp32, block)
    try:
        yield
    finally:
        for block in blocks:
            block.__dict__.pop("attn", None)


class AgenticTabICLv2Implementation:
    """One agentic TabICLv2 (a single bag child): the frozen network, the context, and the recipe."""

    def __init__(self, *, problem_type: str, n_classes: int, device: str, n_threads: int, indices, seed: int):
        self.task_type = "binclass" if problem_type == "binary" else problem_type
        self.n_classes = n_classes
        self.device_ = torch.device(device)
        self.n_threads = n_threads
        self.indices = indices
        self.seed = seed
        self.evaluator = build_evaluator_from_indices(self.task_type, indices)
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_ = X.reset_index(drop=True)
        self.y_ = y.to_numpy()
        self.model_ = MODEL_REGISTRY[self.indices["model_idx"]](task_type=self.task_type).to(self.device_).eval()
        return self

    def _dataset(self, X: pd.DataFrame) -> TabularDataset:
        y = {"train": self.y_}
        return TabularDataset(
            x={"train": self.X_, "test": X.reset_index(drop=True)}, y=y, y_raw=y, task_type=self.task_type
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dataset, ev = self._dataset(X), self.evaluator
        with torch_state(self.n_threads), fp32_attention(self.model_), torch.no_grad():
            encoded = ev._encode_parts(dataset, ("test",))
            ev.members_ = build_members(ev.spec, n_features=encoded["train"].shape[1], seed=self.seed)
            per_member, spread = ev._regression_member_pool(
                dataset=dataset,
                encoded=encoded,
                part="test",
                model=self.model_,
                device=self.device_,
                seed=self.seed,
                preproc_cache={},
            )
        return aggregate_prediction_pool(
            task_type=self.task_type,
            ensemble_idx=ev.spec.ensemble,
            per_member={"test": per_member},
            uncertainty={"test": spread},
        )["test"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dataset, ev = self._dataset(X), self.evaluator
        target = ClassificationTargetEncoder.fit(dataset)
        if target.n_classes > self.model_.max_classes:
            raise ValueError(f"{target.n_classes} classes, the model supports {self.model_.max_classes}.")
        with torch_state(self.n_threads), fp32_attention(self.model_), torch.no_grad():
            encoded = ev._encode_parts(dataset, ("test",))
            ev.members_ = build_members(
                ev.spec, n_features=encoded["train"].shape[1], n_classes=target.n_classes, seed=self.seed
            )
            logits = {
                "test": ev._classification_logits_pool(
                    encoded=encoded,
                    part="test",
                    target=target,
                    model=self.model_,
                    device=self.device_,
                    seed=self.seed,
                    preproc_cache={},
                ),
                "train": np.empty((len(ev.members_), 0, target.n_classes), dtype=np.float32),
            }
            per_member, uncertainty = ev._classification_postprocess_pool(
                logits=logits, y_train=target.y_train, target=target, parts=("test",)
            )
        proba = aggregate_prediction_pool(
            task_type=self.task_type,
            ensemble_idx=ev.spec.ensemble,
            per_member=per_member,
            n_classes=target.n_classes,
            uncertainty=uncertainty,
        )["test"]
        out = np.zeros((len(proba), self.n_classes), dtype=np.float32)
        out[:, target.encoder.classes_] = proba
        return out
