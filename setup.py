from setuptools import setup, find_packages

requirements = [
    'autogluon',
]

setup(
    name='autogluon-zeroshot',
    version='0.0.1',
    packages=find_packages(exclude=('docs', 'tst', 'scripts', 'data')),
    url='https://github.com/Innixma/autogluon-zeroshot-private',
    license='Apache-2.0',
    author='AutoGluon Community',
    install_requires=requirements,
    description='Zeroshot-HPO Simulation for AutoGluon'
)
