"""The tuning-trajectory data covers the context's tasks only, as ``compare`` does."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from tabarena.contexts.abstract_arena_context import AbstractArenaContext
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import (
    _prepare_tuning_trajectories_data,
    _subset_trajectory_data,
)

#: The context's task grid: two splits of `alpha`. The suite's results also cover alpha's split 2 and `beta`.
GRID = [("alpha", 0), ("alpha", 1)]
SUITE = [*GRID, ("alpha", 2), ("beta", 0)]


def _rows(method: str, splits: list[tuple[str, int]], error: float) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "fold": fold,
            "method": method,
            "metric_error": error,
            "metric_error_val": error,
            "time_train_s": 10.0,
            "time_infer_s": 1.0,
            "metric": "log_loss",
            "problem_type": "binary",
            "ta_name": method,
            "ta_suite": "suite",
            "imputed": False,
        }
        for dataset, fold in splits
    ]


def _context() -> SimpleNamespace:
    trajectory = pd.DataFrame(_rows("HPO-N1-CAT", SUITE, 0.2)).assign(
        config_type="CAT", seed=0, n_configs=1, n_iterations=1
    )
    cat = SimpleNamespace(
        method="CAT",
        method_type="config",
        method_class="model",
        tags=(),
        has_hpo_trajectories=True,
        display_name="CatBoost",
        load_hpo_trajectories=trajectory.copy,
    )
    split = SimpleNamespace(num_instances_train=100, num_instances_test=50)
    return SimpleNamespace(
        method_metadata_collection=SimpleNamespace(
            method_metadata_lst=[cat],
            info=lambda: pd.DataFrame(columns=["method", "suite", "method_class", "tags"]),
        ),
        load_results=lambda: pd.DataFrame(_rows("RF (default)", SUITE, 0.3)),
        task_metadata_collection=[
            SimpleNamespace(tabarena_task_name="alpha", splits_metadata={0: split, 1: split, 2: split}),
            SimpleNamespace(tabarena_task_name="beta", splits_metadata={0: split}),
        ],
        _task_metadata_results_filter=lambda: pd.DataFrame(GRID, columns=["dataset", "fold"]),
        fillna_metrics=AbstractArenaContext.fillna_metrics,
    )


def test_trajectory_data_keeps_only_the_context_tasks():
    """A method that ran only the context's tasks is not imputed elsewhere, so it is kept."""
    context = _context()
    combined, _ = _prepare_tuning_trajectories_data(
        tabarena_context=context,
        extra_results=pd.DataFrame(_rows("My system", GRID, 0.1)),
        methods_to_display=None,
        include_baselines=False,
    )

    assert set(zip(combined["dataset"], combined["fold"], strict=True)) == set(GRID)
    scoped = _subset_trajectory_data(
        combined,
        tabarena_context=context,
        fillna_method="RF (default)",
        exclude_imputed=True,
        subset=None,
        folds=None,
    )
    assert set(scoped["method"]) == {"RF (default)", "HPO-N1-CAT", "My system-1.0"}
