"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# --- Example of RealMLP on GPU
BenchmarkSetup(
    benchmark_name="realmlp_output_dir",
    models=[
        # Model name, number of configs / "all" for the maximum
        ("RealMLP", "all"),
    ],
    num_gpus=1,
).setup_jobs()