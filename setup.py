from setuptools import setup, find_packages

requirements = [
    'autogluon.core[all]>=1.0',
    'openml>=0.14.1',  # consider making optional
    'pytest',
    'typing-extensions>=4.11,<5',  # used for `Self` type hint
    'huggingface-hub',
]

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
    description='Dataset of model evaluations supporting ensembling and simulation'
)
