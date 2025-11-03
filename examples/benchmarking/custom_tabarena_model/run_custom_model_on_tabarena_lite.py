"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

from pathlib import Path

import openml
from custom_random_forest_model import get_configs_for_custom_rf

from tabarena.benchmark.experiment import run_experiments_new

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "custom_model")
"""Output directory for saving the results and result artifacts from TabArena."""


def run_tabarena_lite_for_custom_rf():
    """Put all the code together to run a TabArenaLite experiment for
    the custom random forest model.
    """
    # Get all tasks from TabArena-v0.1
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks

    # Gets 1 default and 1 random config = 2 configs
    model_experiments = get_configs_for_custom_rf(num_random_configs=1)

    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )


if __name__ == "__main__":
    run_tabarena_lite_for_custom_rf()
