"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# --- Benchmark TabPFN-2.5 (07/11/2025)
BenchmarkSetup(
    benchmark_name="tabpfnv25_output_07112025",
    models=[
        # Model name, number of configs / "all" for the maximum
        ("RealTabPFN-v2.5", 0),
    ],
    num_gpus=1,
    custom_model_constraints={
        "REALTABPFN-V2.5": {
            "max_n_samples_train_per_fold": 50_000,
            "max_n_features": 2000,
            "max_n_classes": 10,
        }
    },
).setup_jobs()
