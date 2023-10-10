# TabRepo

This repo is made to faciliate the discovery of stronger AutoML system defaults via extensive analysis of large-scale benchmarking results and simulation-based zeroshot portfolio construction.

This repo is WIP and is an actively on-going research effort.

## Installation

Requires the latest `autogluon` installed (can be installed from source).

First, clone the repository:

```
git clone https://github.com/Innixma/autogluon-zeroshot-private.git
cd autogluon-zeroshot-private
```

Create a virtual environments to install the dependencies in:

```
# Optionally create a new venv
python -m venv venv
source venv/bin/activate
```

Install locally:

```
python -m pip install -e .
```

## Quick-start

You can try the code out to generate a zeroshot portfolio immediately by running `scripts/simulate/bagged/run_simulate_zs_single_best_bagged.py`.
Note: You will need `ray` installed to run without edits.

## Related Repositories

https://github.com/ISG-Siegen/assembled - Focused on finding strong ensemble algorithms given a fixed set of models using similar simulation-based approaches. Could be useful to combine the zeroshot simulation logic of this repo with the ensemble simulation logic of `assembled`.

# In-Depth

## Config Generation

To generate model configs, run `scripts/run_generate_all_configs`.

Existing configs are stored in `data/configs/`.

Configs created in this way can then be run on the benchmark to generate results needed for conducting zeroshot simulation.

Next, to generate an AutoMLBenchmark compatible framework yaml file to run the model configs on the benchmark, run `scripts/run_generate_amlb_config.py`.

This will result in the creation of the file `data/configs/frameworks_zeroshot.yaml`.

Refer to `scripts/run_generate_amlb_config.py` for more details.

## Generating hyperparameter sweep results (Running the benchmark on all configs)

WIP. This repo contains results that are already generated inside `data/results/`.

### Prepare the AutoMLBenchmark fork

WIP.

You must copy the `data/configs/frameworks_zeroshot.yaml` file into an AutoMLBenchmark custom_configs folder and execute AutoMLBenchmark.

TODO: Document the full process of creating the AutoMLBenchmark fork and running the benchmark

### Run the benchmark on all configs

WIP.

#### Handling server errors / transient failures

WIP.

## Generating all required input files for simulation

WIP.

### Aggregating raw results

WIP. This repo contains results that are already aggregated and preprocessed in S3 (not yet public)

### Preprocessing aggregated results

WIP.

### Run evaluation code to generating the remaining required inputs

WIP.

## Downloading results

WIP. This repo contains results that are already generated inside `data/results/`.

For size related reasons, the following files are not included in this repo and must be downloaded separately:
- zeroshot_gt_2022_10_13_zs.pkl : 200 MB, contains ground truth labels for 610 dataset folds on test and validation, plus metadata information.
- zeroshot_pred_proba_2022_10_13_zs.pkl : 17 GB, contains prediction probabilities of 608 model configs for 610 dataset folds.

Access to downloading these files is currently WIP. The code to do so if you have permissions is in `scripts/simulate/*/run_download_metadata*.py`.
You can run SingleBest simulations without these files, but Ensemble simulations require these files.

Note: zeroshot_pred_proba is actually 260 GB, but has been shrunk to 17 GB by removing datasets that use a lot of space to store prediction probabilities.
In future, better results could be achieved by using more of the datasets, at the cost of more memory usage and slower simulation speed.

## Zeroshot Simulation

Now that we have results for every config on every dataset, we can simulate config combinations or portfolios and test how well they perform.

Two simulation strategies are implemented:

1. SingleBest - Create a portfolio that scores by choosing the model with the best validation score as the final solution (no ensembling)
    - Run via `scripts/simulate/all_v3/run_simulate_zs_single_best.py`
2. Ensemble - Create a portfolio that scores by creating a GreedyWeightedEnsemble via the validation data as the final solution.
    - Run via `scripts/simulate/all_v3/run_simulate_zs_ensemble.py` (Requires zeroshot_pred_proba file downloaded)

### Scoring function

To determine how well a portfolio does, we calculate average rank across all datasets (lower is better).
Rank is calculated by comparing the evaluation metric score of the portfolio on a given dataset to other potential solutions.
We currently compare against the results of different AutoML systems stored in `data/results/automl/`.

### Avoiding overfitting

To ensure overfitting is not occurring, we run cross-validation during simulation and return the average rank on the holdout fold as the portfolio score.
Note that this average score is not for a particular portfolio, but rather for the entire simulation process, since every fold will generate a different portfolio.

## Validating a portfolio

WIP. To check if a portfolio is indeed strong, we can generate a results file as if we had run that portfolio on the benchmark, but instead using the simulation results directly.
With this results file, we can send this to analysis code that can compare it against other portfolios and AutoML systems.
Note that this analysis code is currently not publicly available, but is planned to be available in future.

To generate a results file from the cross-validated portfolio results, edit and run `scripts/evaluate_simulation.py`.

## Future Work

### Expand the datasets used

Currently we are using the 104 tabular datasets from the AutoMLBenchmark,
with 10-fold cross-val, leading to a total of 1040 task results per config.

It should be noted that 10-fold cross-val helps significantly
to improve the strength of zero-shot by reducing overfitting during simulation and is recommended to keep doing.

I suspect that achieving a benchmark size of 500 datasets with 10-fold cross-val is enough
to nearly completely avoid overfitting via zero-shot even without using cross-val during simulation.

With 1000 datasets, I suspect that zeroshot will simply not have enough degrees of freedom to meaningfully overfit.

### Identify a good scoring mechanism

The current scoring mechanism of average rank compared to AutoML systems is usable but probably not optimal.
The problem with using eval_metric results directly is that they would not weigh the datasets equally.

### Improve config search space

Currently no work has been done to validate or improve the search spaces used to generate the model configs.

### Re-run configs with AutoGluon's bagging mode

By running in bagging mode, we will get 5x+ more validation data due to out-of-fold predictions.
This will help in avoiding over-fitting of the validation data when using a simulation strategy,
leading to larger ensembles working better.

Note that this will vastly increase the size of zeroshot_pred_proba, and is likely impractical for
large multiclass datasets where the size could become greater than 1 TB (eg. Dionis).

### Identify ways to prune model configs from the simulation process

Some configs are obviously bad and/or pareto dominated by other configs. To speed up simulation, we could identify
a way to remove them from consideration quickly.

Note that the strategy used to prune must also avoid any over-fitting on the CV result.

One candidate pruning technique: Run SingleBest to select the best portfolio of N*5 models.
Then use those N*5 models as candidates to select an ensemble portfolio of N models.
Because SingleBest is 100x faster to simulate than Ensemble, this is a feasible pruning strategy,
and would not over-fit if done separately on every fold.

### Speed up eval_metrics

The majority of simulation time is spent in computing scores. By optimizing the implementations
of metrics such as `roc_auc` and `log_loss`, this could significantly speed up simulations.

### Add support for configs that are supported on only a subset of datasets

For example, K-Nearest-Neighbors does not work on datasets with only categorical features in AutoGluon's implementation.
However, KNN still might be useful to include overall, and simply has no impact if it doesn't work on a given dataset.

Another example is Tabular transformer models that might not be worth using when many features are present.

### Multi-objective Support (Training time, inference speed, memory usage, disk usage, deployment complexity)

We may care about more than simply test score. For example, training time is often specified by the user and
we need to ensure a config works well under different time limit constraints.

### Integration with HPO

We could use a zeroshot portfolio to warm-start a HPO process such as Bayesian Optimization.
This would be expensive to evaluate the strength of, so would need to be thoughtfully designed.

### Zeroshot tuning beyond models

Once we identify a strong portfolio, we can further improve the score
of the portfolio by zeroshot tuning other aspects of AutoGluon around it, such as:

- Data Preprocessing
- Number of folds in bagging
- Percentage of holdout validation data
- Order of model training (Meaningful for multi-objective such as training time)
- Integration with HPO
- Integration with Hyperband style methods
- Distillation

Once we have identified improved defaults for the above,
we can then go back and find a new zeroshot portfolio that works best with the new defaults.
This process can be repeated iteratively until convergence.

### Stacking support

Theoretically we could devise a strategy to create a zeroshot portfolio that optimizes the score of a multi-layer stack ensemble.
This would likely be a multi-stage process where we do a greedy portfolio selection layer by layer.
In order to be computationally feasible, code edits would have to be made to AutoGluon to support mock models for prior layers.

### Metalearning

A natural extension to zeroshot is metalearning.
Before starting metalearning, we should be mindful of the risks to overfitting and increased complexity.

While full metalearning has many risks,
a limited scope metalearning could be useful by only considering a few meta-features.

Very low risk meta-features:
- problem_type : ['binary', 'multiclass', 'regression']

Low risk meta-features:
- num_classes (Bucketed by magnitude)
- data_size_in_bytes (Bucketed by magnitude)
- contains_missing_values : [True, False]
- has_text : [True, False]
- has_image : [True, False]
- has_numeric : [True, False]
- has_categorical : [True, False]

Medium risk meta-features:
- num_rows (Bucketed by magnitude)
- num_features (Bucketed by magnitude)
- num_classes
- percentage_missing_values
- percentage_categorical
- percentage_numeric

High risk meta-features:
- data_size_in_bytes (Exact)
- num_rows (Exact)
- num_features (Exact)

Very high risk meta-features:
- num_features_* (numeric, categorical, etc.)
- feature column names
- label column name

## Old

To view initial experiment results, refer to `scripts/zeroshot_all_analysis.ipynb`
