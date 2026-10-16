from __future__ import annotations

import math


def get_split_idx(
    fold: int = 0,
    repeat: int = 0,
    sample: int = 0,
    n_folds: int = 1,
    n_repeats: int = 1,
    n_samples: int = 1,
) -> int:
    """Return the split index of an OpenML ``(repeat, fold, sample)`` triple.

    Folds vary fastest, then samples, then repeats:
    ``split_idx = n_folds * n_samples * repeat + n_samples * fold + sample``. With
    ``n_samples == 1`` (as in TabArena) this is ``n_folds * repeat + fold``.

    This split index is what the ``fold`` column of the results DataFrames holds (see
    ``BaselineResult``). With TabArena v0.1's 3 folds, ``fold == 13`` is repeat 4, fold 1.
    Use :func:`get_split_vals_from_split_idx` to recover the OpenML values.
    """
    assert fold < n_folds
    assert repeat < n_repeats
    assert sample < n_samples
    return n_folds * n_samples * repeat + n_samples * fold + sample


def get_split_vals_from_split_idx(
    split_idx: int,
    n_folds: int = 1,
    n_repeats: int = 1,
    n_samples: int = 1,
) -> tuple[int, int, int]:
    """Invert :func:`get_split_idx`: return the ``(repeat, fold, sample)`` of a split index.

    Use this to map the ``fold`` column of the results DataFrames, which holds the split
    index rather than the OpenML fold, back to OpenML values. For example, with
    ``n_folds=3`` and ``n_repeats=10``, split 13 is ``(4, 1, 0)``: repeat 4, fold 1.
    """
    repeat = math.floor(split_idx / (n_folds * n_samples))
    remainder = split_idx % (n_folds * n_samples)
    fold = math.floor(remainder / n_samples)
    sample = remainder % n_samples

    assert fold < n_folds
    assert repeat < n_repeats
    assert sample < n_samples
    return repeat, fold, sample
