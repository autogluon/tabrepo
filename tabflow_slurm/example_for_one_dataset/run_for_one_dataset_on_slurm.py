"""Setup slurm jobs for a local task for the collaborations."""

from __future__ import annotations

import pandas as pd
from get_local_task import get_tasks_for_tabarena

from tabflow_slurm.setup_slurm_base import BenchmarkSetup


def _get_metadata(task_id_str) -> pd.DataFrame:
    return pd.DataFrame(
        [[10, 3, task_id_str, 3000, 50, 2, "binary"]],
        columns=[
            "tabarena_num_repeats",
            "num_folds",
            "task_id",
            "num_instances",
            "num_features",
            "num_classes",
            "problem_type",
        ],
    )


# --- Run Local Dataset Benchmark for Biopsy collaboration
user_task = get_tasks_for_tabarena(dataset_file="biopsie_preprocessed_full_cohort.csv")
BenchmarkSetup(
    custom_metadata=_get_metadata(task_id_str=user_task.task_id_str),
    benchmark_name="biopsie_preprocessed_full_cohort",
    n_random_configs=25,
    models=[
        # -- TFMs
        ("TabPFNv2", "all"),
        # -- Neural networks
        ("RealMLP", "all"),
        ("TabM", "all"),
        # -- Tree-based models
        ("CatBoost", "all"),
        ("EBM", "all"),
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        # -- Baselines
        ("KNN", "all"),
        ("Linear", "all"),
    ],
    num_gpus=1,
).setup_jobs()