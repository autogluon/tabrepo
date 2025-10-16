from __future__ import annotations

from setuptools import find_packages, setup

requirements = [
    "autogluon>=1.4.1b20250910,<1.5",  # TODO: Remove after moving `benchmark` code elsewhere
    "openml>=0.14.1",  # consider making optional
    "pyyaml",
    "pytest",
    "tqdm",
    "typing-extensions>=4.11,<5",  # used for `Self` type hint
    "huggingface-hub",
    "numpy",
    "pandas",

    # TODO: For 2025 paper, consider making optional
    "tueplots",
    "autorank==1.2.1",
    "seaborn==0.13.2",
    "matplotlib==3.9.2",
]

extras_require = {
    "tabpfn": [
        "tabpfn>=2.0.9",  # We used version 2.0.9
        "numba>=0.57,<1.0",  # Required by kditransform
    ],
    "tabicl": [
        "tabicl>=0.1.1",
    ],
    "ebm": [
        "interpret-core>=0.7.2",
    ],
    "search_spaces": [
        "configspace>=1.2,<2.0",
    ],
    "realmlp": [
        "pytabkit>=1.5.0,<2.0",
    ],
    "tabdpt": [
        "tabdpt>=1.1.6"
    ],
    "tabm": [
        "torch",
    ],
    "modernnca": [
        "category_encoders",
    ],
    "xrfm": [
        "xrfm[cu12]",
    ]
}

benchmark_requires = []
for extra_package in [
    "tabpfn",
    "tabicl",
    "ebm",
    # "search_spaces",
    "realmlp",
    "tabdpt",
    "tabm",
    "modernnca",
    "xrfm",
]:
    benchmark_requires += extras_require[extra_package]
benchmark_requires = list(set(benchmark_requires))
extras_require["benchmark"] = benchmark_requires

setup(
    name="tabrepo",
    version="0.0.1",
    packages=find_packages(exclude=("docs", "tst", "data")),
    package_data={
        "tabrepo": [
            "metrics/_roc_auc_cpp/compile.sh",
            "metrics/_roc_auc_cpp/cpp_auc.cpp",
            "nips2025_utils/metadata/task_metadata_tabarena51.csv",
        ],
    },
    url="https://github.com/autogluon/tabrepo",
    license="Apache-2.0",
    author="AutoGluon Community",
    install_requires=requirements,
    extras_require=extras_require,
    description="Dataset of model evaluations supporting ensembling and simulation",
)
