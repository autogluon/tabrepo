from __future__ import annotations

import argparse
import yaml
import pandas as pd
import json

from tabrepo.benchmark.experiment.experiment_utils import ExperimentBatchRunner
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGWrapper
from tabrepo.benchmark.models.ag import RealMLPModel

# If the artifact is present, it will be used and the models will not be re-run.
if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, required=True, help="Name of the context")
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--method_name', type=str, required=True, help="Name of the method")
    parser.add_argument('--wrapper_class', type=str, required=True, help="Wrapper class for the method")
    parser.add_argument('--fit_kwargs', type=str, required=True, help="Fit kwargs for the method in JSON format")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")

    args = parser.parse_args()

    # Load Context
    context_name = args.context_name #"D244_F3_C1530_30"  # 30 Datasets. To run larger, set to "D244_F3_C1530_200"
    expname = args.experiment_name  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    #TODO: Download the repo without pred-proba
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    datasets = args.datasets
    folds = args.folds

    fit_kwargs = json.loads(args.fit_kwargs)
    # NOTE: Cannot change dictionary keys during iteration. Hence iterate over a copy of the keys.
    for key in list(fit_kwargs['hyperparameters'].keys()):
        # Handle model class passed as strings - like GBM in AGWrapper
        if key == "GBM":
            fit_kwargs['hyperparameters'][key] = fit_kwargs['hyperparameters'].pop(key)
        else:
            fit_kwargs['hyperparameters'][eval(key)] = fit_kwargs['hyperparameters'].pop(key)


    methods = [(args.method_name, eval(args.wrapper_class), {'fit_kwargs': fit_kwargs})]

    print(f"\nWrapper class: {args.wrapper_class}\n")
    print(f"Fit kwargs: {fit_kwargs}\n")

    repo: EvaluationRepository = ExperimentBatchRunner(expname=expname, task_metadata=repo_og.task_metadata).generate_repo_from_experiments(
        datasets=datasets, 
        folds=folds,
        methods=methods, 
        ignore_cache=ignore_cache,
        convert_time_infer_s_from_batch_to_sample=True,
        mode="aws",
        s3_bucket=args.s3_bucket,
    )

    repo.print_info()

    save_path = "repo_new"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    print(f"New Configs   : {repo.configs()}")