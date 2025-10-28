# TabRepo

## Note

The below is a description from the 2024 TabRepo paper, and refactoring has occurred since then.
To ensure the code works exactly as described below, please refer to the instructions in the AutoML2024 branch of tabrepo: https://github.com/autogluon/tabrepo/tree/AutoML2024

## Introduction

TabRepo contains the predictions and metrics of 1530 models evaluated on 211 classification and regression datasets. 
This allows to compare against state-of-the-art AutoML systems or random configurations by querying 
precomputed results. We also store and expose model predictions so any ensembling strategy can also be benchmarked 
cheaply by just querying precomputed results.

We give scripts from our paper, [TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications](https://arxiv.org/abs/2311.02971), so that one can reproduce all experiments that compare different models and portfolio
strategies against state-of-the-art AutoML systems.

The key features of the repo are:
* 1530 models densely evaluated on 211 datasets with 3 distinct folds using 8-fold bagged CV
  * `1530*211*3*8` = 7.75 million models
* 330 GB of model predictions
* code to compare methods against state-of-the-art AutoML systems and random model configurations
* fast evaluations of any ensemble of models from table lookups with a few engineering tricks:
  * fast metric evaluations with specific optimized cpp code (for instance to compute roc auc)
  * efficient format that loads model evaluation on the fly with low memory footprint

![tuning-impact.png](https://raw.githubusercontent.com/autogluon/tabrepo/refs/heads/AutoML2024/data/tuning-impact.png)
![sensitivity.png](https://raw.githubusercontent.com/autogluon/tabrepo/refs/heads/AutoML2024/data/sensitivity.png)
![paper-figure.png](https://raw.githubusercontent.com/autogluon/tabrepo/refs/heads/AutoML2024/data/paper-figure.png)

 
## Installation

To install the repository, ensure you are using Python 3.9-3.11. Other Python versions are not supported. Then, run the following:

```bash
git clone https://github.com/autogluon/tabrepo.git
pip install -e tabrepo/
```

To install the dependencies needed for the upcoming TabRepo 2.0, you must instead do the following:

```bash
# Requires latest mainline AutoGluon (or AutoGluon 1.3+)
git clone https://github.com/autogluon/autogluon
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabrepo.git
pip install -e tabrepo/[benchmark]
```

Only Linux support has been tested. Support for Windows and MacOS is not confirmed, and you may run into bugs or a suboptimal experience (especially if you are unable to install ray).

### Reproducing AutoML Conf 2024 Paper

If you are interested in reproducing the experiments of the paper, you will need these extra dependencies:

```bash
# Install AG benchmark, required only to reproduce results showing win-rate tables

git clone https://github.com/autogluon/autogluon-bench.git
pip install -e autogluon-bench/

git clone https://github.com/Innixma/autogluon-benchmark.git
pip install -e autogluon-benchmark/

# Install extra dependencies used for results scripts
pip install autorank seaborn
```

You are all set!

## Quick-start

**Recommended: Refer to [examples/tabrepo/run_quickstart.py](examples/tabrepo/run_quickstart.py) for a full runnable tutorial.**

Now lets see how to do basic things with TabRepo.

**Accessing model evaluations.** To access model evaluations, you can do the following:

```python
from tabarena import load_repository

repo = load_repository("D244_F3_C1530_30")
repo.metrics(datasets=["Australian"], configs=["CatBoost_r22_BAG_L1", "RandomForest_r12_BAG_L1"])
```

The code will return the metrics available for the configuration and dataset chosen. 

The example loads a smaller version of TabRepo with only a few datasets for illustrative purpose and shows
the evaluations of one ensemble and how to query the stored predictions of a given model.
When calling `load_repository` models predictions and TabRepo metadata will be fetched from the internet. We use a smaller version 
here as it can take a long time to download all predictions, in case you want to query all datasets, replace the context
with `D244_F3_C1530`.


**Querying model predictions.**
To query model predictions, run the following code:

```python
from tabarena import load_repository

repo = load_repository("D244_F3_C1530_30")
print(repo.predict_val_multi(dataset="Australian", fold=0, configs=["CatBoost_r22_BAG_L1", "RandomForest_r12_BAG_L1"]))
```

This will return predictions on the validation set. 
You can also use `predict_test` to get the predictions on the test set.

**Simulating ensembles.**
To evaluate an ensemble of any list of configuration, you can run the following:

```python
from tabarena import load_repository

repo = load_repository("D244_F3_C1530_30")
print(repo.evaluate_ensemble(dataset="Australian", fold=0, configs=["CatBoost_r22_BAG_L1", "RandomForest_r12_BAG_L1"]))
```

this code will return the error of an ensemble whose weights are computed with the Caruana procedure after loading model
predictions and validation groundtruth.

## Available Contexts

Context's are used to load a repository and are downloaded with the following code:

```python
from tabarena import load_repository

repo = load_repository(context_name)
```

Below is a list of the available contexts in TabRepo.

| Context Name      | # Datasets | # Folds | # Configs | Disk Size | Notes                                                                                 |
|:------------------|-----------:|--------:|----------:|----------:|:--------------------------------------------------------------------------------------|
| D244_F3_C1530     |        211 |       3 |      1530 |    330 GB | All successful datasets. 64 GB+ memory recommended. May take a few hours to download. |
| D244_F3_C1530_200 |        200 |       3 |      1530 |    120 GB | Used for results in paper. 32 GB memory recommended                                   |
| D244_F3_C1530_175 |        175 |       3 |      1530 |     57 GB | 16 GB memory recommended                                                              |
| D244_F3_C1530_100 |        100 |       3 |      1530 |    9.5 GB | Ideal for fast prototyping                                                            |
| D244_F3_C1530_30  |         30 |       3 |      1530 |    1.1 GB | Toy context                                                                           |
| D244_F3_C1530_10  |         10 |       3 |      1530 |    220 MB | Toy context                                                                           |
| D244_F3_C1530_3   |          3 |       3 |      1530 |     33 MB | Toy context                                                                           |


The files will be downloaded into `~/.cache/tabrepo/data" by default, you can change this location 
by specifying the environment variable "TABREPO_CACHE".

## Reproducing paper experiments

To ensure reproducibility, you can use the `AutoML2024` [branch](https://github.com/autogluon/tabrepo/tree/AutoML2024) which provides a snapshot of the code that is able to reproduce the results.

To reproduce the experiments from the paper, run:

```bash
python scripts/baseline_comparison/evaluate_baselines.py
```

Note: This file will only exist in the AutoML2024 branch of TabRepo.

The experiment will require ~200GB of disk storage and 32GB of memory (although we use memmap to load model predictions
on the fly, large datasets still have a significant memory footprint even for a couple of models). In particular, we
used a `m6i.4xlarge` machine for our experiments which took under 24 hrs (less than $7 of compute with spot instance pricing).
Excluding the 10-repeat seeded ablations, the experiments take under 1 hour.

All the table and figures of the paper will be generated under `scripts/output/{expname}`.

### Colab Notebook

To run a subset of experiments on a Colab notebook, refer to [https://colab.research.google.com/github/autogluon/tabrepo/blob/AutoML2024/examples/TabRepo_Reproducibility.ipynb](https://colab.research.google.com/github/autogluon/tabrepo/blob/AutoML2024/examples/TabRepo_Reproducibility.ipynb)

### Reproducing the raw TabRepo dataset

To reproduce the entire TabRepo dataset (context `"D244_F3_C1530"`) from scratch, refer to the [benchmark execution README](https://github.com/autogluon/tabrepo/tree/AutoML2024/scripts/execute_benchmarks/README.md).

To instead reproduce a small subset of the TabRepo dataset in a few minutes, run [examples/tabrepo/run_quickstart_from_scratch.py](examples/tabrepo/run_quickstart_from_scratch.py).

## Running HPO experiments with TabRepo

Syne Tune proposes a wrapper to simulate methods such as random-search, bayesian-optimization or others on TabRepo search-space. 
You can see [this example](https://github.com/syne-tune/syne-tune/blob/main/examples/launch_tabrepo.py) to get started.

## Future work

We have been using TabRepo to study potential data leakage in AutoGluon stacking, we believe the following are also 
potential interesting future work directions:

* finding good dataset features to enable strategies that condition on the dataset at hand
* adding evaluations from other domains such as time-series, CV or NLP
* improve the portfolio selection which currently use a greedy approach unaware of the ensembling effect

## Citation

If you find this work useful for you research, please cite the following:
```
@inproceedings{
  tabrepo,
  title={TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications},
  author={David Salinas and Nick Erickson},
  booktitle={AutoML Conference 2024 (ABCD Track)},
  year={2024},
  url={https://openreview.net/forum?id=03V2bjfsFC}
}
```
