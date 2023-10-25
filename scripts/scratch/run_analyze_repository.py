from __future__ import annotations

from typing import List

from tabrepo.utils.cache import cache_function
from tabrepo.contexts import get_subcontext
from tabrepo.repository.evaluation_repository_zeroshot import EvaluationRepositoryZeroshot
from autogluon.core.metrics import get_metric


def analyze(repo: EvaluationRepositoryZeroshot, models: List[str] | None = None):
    if models is None:
        models = repo.get_configs()

    datasets = repo.datasets()
    fold = 0
    n_datasets = len(datasets)
    print(f'n_models={len(models)} | models={models}')
    print(f'n_datasets={n_datasets}')
    for dataset_num, dataset in enumerate(datasets):
        task = repo.task_name(dataset=dataset, fold=fold)

        zsc = repo._zeroshot_context

        task_info = repo.dataset_info(dataset=dataset)

        problem_type = task_info['problem_type']
        metric_name = task_info['metric']
        eval_metric = get_metric(metric=metric_name, problem_type=problem_type)
        y_val = repo.labels_val(dataset=dataset, fold=fold)
        y_test = repo.labels_test(dataset=dataset, fold=fold)

        print(f'({dataset_num + 1}/{n_datasets}) task: {task}\n'
              f'\tdataset: {dataset} | fold: {fold}\n'
              f'\tproblem_type={problem_type} | metric_name={metric_name}\n'
              f'\ttrain_rows={len(y_val)} | test_rows={len(y_test)}')

        pred_test = repo.predict_test(dataset=dataset, fold=fold, configs=models)
        pred_val = repo.predict_val(dataset=dataset, fold=fold, configs=models)

        if problem_type == 'binary':
            # Force binary prediction probabilities to 1 dimensional prediction probabilities of the positive class
            # if it is in multiclass format
            if len(pred_val.shape) == 3:
                pred_val = pred_val[:, :, 1]
            if len(pred_test.shape) == 3:
                pred_test = pred_test[:, :, 1]

        for i, m in enumerate(models):
            print(f'\tMODEL {i}: {m}')

            pred_val_m = pred_val[i]
            pred_test_m = pred_test[i]

            row = zsc.df_results_by_dataset_vs_automl[zsc.df_results_by_dataset_vs_automl['task'] == task]
            row = row[row['framework'] == m]

            test_error_row = row['metric_error'].iloc[0]
            val_error_row = row['metric_error_val'].iloc[0]
            test_error_zs = eval_metric.error(y_test, pred_test_m)
            val_error_zs = eval_metric.error(y_val, pred_val_m)

            test_error_diff = test_error_zs - test_error_row
            val_error_diff = val_error_zs - val_error_row
            print(f'\t\tTest Error: {test_error_zs:.4f}\t| {test_error_row:.4f}\t | DIFF: {test_error_diff:.4f}')
            print(f'\t\tVal  Error: {val_error_zs:.4f}\t| {val_error_row:.4f}\t | DIFF: {val_error_diff:.4f}')

    print('Done...')


if __name__ == '__main__':
    # Download repository from S3 and cache it locally for re-use in future calls
    repository: EvaluationRepositoryZeroshot = cache_function(
        fun=lambda: get_subcontext("D244_F3_C1416_30").load_from_parent(),
        cache_name=f"repo_micro",
    ).to_zeroshot()

    analyze(repo=repository)
