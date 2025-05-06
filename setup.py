from setuptools import setup, find_packages

requirements = [
    'autogluon.core[all]>=1.3',
    'openml>=0.14.1',  # consider making optional
    'pytest',
    'typing-extensions>=4.11,<5',  # used for `Self` type hint
    'huggingface-hub',
]

extras_require = {
    "autogluon": [
        "autogluon.tabular[all]>=1.3",
    ],
    "tabpfn": [
        "tabpfn>=2",
    ],
    "tabicl": [
        "tabicl",
    ],
    "ebm": [
        "interpret-core>=0.6.1",
    ],
    "search_spaces": [
        "configspace",
    ],
    "realmlp": [
        "pytabkit"
    ]
}

benchmark_requires = []
for extra_package in ["autogluon", "tabpfn", "tabicl", "ebm", "search_spaces", "realmlp"]:
    benchmark_requires += extras_require[extra_package]
benchmark_requires = list(set(benchmark_requires))
extras_require["benchmark"] = benchmark_requires

setup(
    name='tabrepo',
    version='0.0.1',
    packages=find_packages(exclude=('docs', 'tst', 'data')),
    package_data={"tabrepo": [
        "metrics/_roc_auc_cpp/compile.sh",
        "metrics/_roc_auc_cpp/cpp_auc.cpp",
    ]},
    url='https://github.com/autogluon/tabrepo',
    license='Apache-2.0',
    author='AutoGluon Community',
    install_requires=requirements,
    extras_require=extras_require,
    description='Dataset of model evaluations supporting ensembling and simulation'
)
