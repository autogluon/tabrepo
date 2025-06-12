from __future__ import annotations

from setuptools import find_packages, setup

requirements = [
    "autogluon.core[all]>=1.3",
    "openml>=0.14.1",  # consider making optional
    "pytest",
    "typing-extensions>=4.11,<5",  # used for `Self` type hint
    "huggingface-hub",
]

extras_require = {
    "autogluon": [
        "autogluon>=1.3",
    ],
    "tabpfn": [
        "tabpfn>=2.0.9",  # We used version 2.0.9
        "kditransform",
    ],
    "tabicl": [
        "tabicl>=0.1.1",
    ],
    "ebm": [
        "interpret-core>=0.6.1",
    ],
    "search_spaces": [
        "configspace",
    ],
    "realmlp": [
        "pytabkit>=1.3.0",
    ],
    "tabdpt": [
        # TODO: pypi package is not available yet
        "tabdpt @ git+https://github.com/layer6ai-labs/TabDPT.git",
        # used hash: 9699d9592b61c5f70fc88f5531cdb87b40cbedf5
    ],
    "tabm": [
        "pytabkit>=1.3.0",
    ],
    "modernnca": [
        "category_encoders",
    ],
    "mitra": [
        "loguru",
        "einx",

        # FIXME: flash-attn seems to be an involved process to install... requires running with:
        #  pip install flash-attn --no-build-isolation
        #  Ref here: https://github.com/Dao-AILab/flash-attention
        #  pip install ninja for speedup? (has to be done beforehand?)
        "flash-attn==2.6.3",
        "autogluon>=1.3",
    ],
}

benchmark_requires = []
for extra_package in [
    "autogluon",
    "tabpfn",
    "tabicl",
    "ebm",
    "search_spaces",
    "realmlp",
    "tabdpt",
    "tabm",
    "modernnca",
]:
    benchmark_requires += extras_require[extra_package]
benchmark_requires = list(set(benchmark_requires))
extras_require["benchmark"] = benchmark_requires

# FIXME: For 2025 paper, cleanup after
extras_require["benchmark"] += [
    "seaborn==0.13.2",
    "matplotlib==3.9.2",
    "autorank==1.2.1",
    "fastparquet",  # FIXME: Without this, parquet loading is inconsistent for list columns
    "tueplots",
]

setup(
    name="tabrepo",
    version="0.0.1",
    packages=find_packages(exclude=("docs", "tst", "data")),
    package_data={
        "tabrepo": [
            "metrics/_roc_auc_cpp/compile.sh",
            "metrics/_roc_auc_cpp/cpp_auc.cpp",
            "nips2025_utils/metadata/task_metadata_tabarena51.csv",
            "nips2025_utils/metadata/task_metadata_tabarena60.csv",
            "nips2025_utils/metadata/task_metadata_tabarena61.csv",
        ],
    },
    url="https://github.com/autogluon/tabrepo",
    license="Apache-2.0",
    author="AutoGluon Community",
    install_requires=requirements,
    extras_require=extras_require,
    description="Dataset of model evaluations supporting ensembling and simulation",
)
