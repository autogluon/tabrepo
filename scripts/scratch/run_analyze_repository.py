from __future__ import annotations

from typing import List

from tabrepo.utils.cache import cache_function
from tabrepo.repository import EvaluationRepository, EvaluationRepositoryZeroshot
from autogluon.core.metrics import get_metric


def analyze(repo: EvaluationRepositoryZeroshot, models: List[str] | None = None):
    if models is None:
        models = repo.list_models()

    # models = ["CatBoost_r35_BAG_L1"]

    datasets = repo.datasets()
    fold = 0
    n_datasets = len(datasets)
    print(f'n_models={len(models)} | models={models}')
    print(f'n_datasets={n_datasets}')
    for dataset_num, dataset in enumerate(datasets):
        tid = repo.dataset_to_tid(dataset)
        task = repo.task_name(tid=tid, fold=fold)

        zsc = repo._zeroshot_context
        tid = zsc.dataset_name_to_tid_dict[task]

        # Note: This contains a lot of information beyond what is used here, use a debugger to view
        task_ground_truth_metadata: dict = repo._ground_truth[tid][fold]

        problem_type = task_ground_truth_metadata['problem_type']
        metric_name = task_ground_truth_metadata['eval_metric']
        eval_metric = get_metric(metric=metric_name, problem_type=problem_type)
        y_val = task_ground_truth_metadata['y_val']
        y_test = task_ground_truth_metadata['y_test']

        print(f'({dataset_num + 1}/{n_datasets}) task: {task}\n'
              f'\tname: {dataset} | fold: {fold} | tid: {tid}\n'
              f'\tproblem_type={problem_type} | metric_name={metric_name}\n'
              f'\ttrain_rows={len(y_val)} | test_rows={len(y_test)}')

        pred_val, pred_test = repo._tabular_predictions.predict(dataset=tid, fold=fold, splits=['val', 'test'], models=models)

        if problem_type == 'binary':
            # Force binary prediction probabilities to 1 dimensional prediction probabilities of the positive class
            # if it is in multiclass format
            if len(pred_val.shape) == 3:
                pred_val = pred_val[:, :, 1]
            if len(pred_test.shape) == 3:
                pred_test = pred_test[:, :, 1]

        # Optional if you want them to be in numpy form
        # Note: Both y_val and y_test are in the internal AutoGluon representation, not the external representation.
        y_val = y_val.to_numpy()
        y_test = y_test.fillna(-1).to_numpy()

        for i, m in enumerate(models):
            print(f'\tMODEL {i}: {m}')

            pred_val_m = pred_val[i]
            pred_test_m = pred_test[i]

            row = zsc.df_results_by_dataset_vs_automl[zsc.df_results_by_dataset_vs_automl['dataset'] == task]
            row = row[row['framework'] == m]

            test_error_row = row['metric_error'].iloc[0]
            test_error_zs = eval_metric.error(y_test, pred_test_m)
            val_error_zs = eval_metric.error(y_val, pred_val_m)

            test_error_diff = test_error_zs - test_error_row
            print(f'\t\tTest Error: {test_error_zs:.4f}\t| {test_error_row:.4f}\t | DIFF: {test_error_diff:.4f}')
            print(f'\t\tVal  Error: {val_error_zs:.4f}')

    print('Done...')


if __name__ == '__main__':
    # Download repository from S3 and cache it locally for re-use in future calls
    repository: EvaluationRepositoryZeroshot = cache_function(
        fun=lambda: EvaluationRepository.load('s3://autogluon-zeroshot/repository/BAG_D244_F1_C16_micro.pkl'),
        cache_name=f"repo_micro",
    ).to_zeroshot()

    analyze(repo=repository)
