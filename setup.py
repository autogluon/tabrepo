from __future__ import annotations

from setuptools import find_packages, setup

requirements = [
    "autogluon>=1.4,<1.5",  # TODO: Remove after moving `benchmark` code elsewhere
    "openml>=0.14.1",  # consider making optional
    "pyyaml",
    "pytest",
    "typing-extensions>=4.11,<5",  # used for `Self` type hint
    "huggingface-hub",
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
        "interpret-core>=0.6.1",
    ],
    "search_spaces": [
        "configspace>=1.2,<2.0",
    ],
    "realmlp": [
        "pytabkit>=1.5.0,<2.0",
    ],
    "tabdpt": [
        # TODO: pypi package is not available yet
        # FIXME: newest version (1.1) has (unnecessary) strict version requirements
        #  that are not compatible with autogluon, so we stick to the old hash
        "tabdpt @ git+https://github.com/layer6ai-labs/TabDPT.git@9699d9592b61c5f70fc88f5531cdb87b40cbedf5",
        # used hash: 9699d9592b61c5f70fc88f5531cdb87b40cbedf5
    ],
    "tabm": [
        "torch",
    ],
    "modernnca": [
        "category_encoders",
    ],
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
]:
    benchmark_requires += extras_require[extra_package]
benchmark_requires = list(set(benchmark_requires))
extras_require["benchmark"] = benchmark_requires

# FIXME: For 2025 paper, cleanup after
extras_require["benchmark"] += [
    "seaborn==0.13.2",
    "matplotlib==3.9.2",
    "autorank==1.2.1",
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
