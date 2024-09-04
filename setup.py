from setuptools import setup, find_packages

requirements = [
    'autogluon.core[all]',
    'pytest',
    'typing-extensions>=4.11,<5',  # used for `Self` type hint
]

setup(
    name='tabrepo',
    version='0.0.1',
    packages=find_packages(exclude=('docs', 'tst', 'data')),
    url='https://github.com/autogluon/tabrepo',
    license='Apache-2.0',
    author='AutoGluon Community',
    install_requires=requirements,
    description='Dataset of model evaluations supporting ensembling and simulation'
)
