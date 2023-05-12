import pandas as pd
import numpy as np

from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext
from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions

np.random.seed(0)

def make_random_metric():
    output_cols = ['time_train_s', 'time_infer_s', 'bestdiff', 'loss_rescaled', 'time_train_s_rescaled',
                   'time_infer_s_rescaled', 'metric_error', 'score_val']
    return {output_col: np.random.rand() for output_col in output_cols}


def load_context_artificial(**kwargs):
    # TODO write specification of dataframes schema, this code produces a minimal example that enables
    #  to use all the features required in evaluation such as listing datasets, evaluating ensembles or
    #  comparing to baselines
    dataset_names = ["ada", "abalone"]
    dataset_ids = [359944, 359946]
    n_folds = 3
    models = ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    baselines = ["b1", "b2"]

    configs_full = {model: {} for model in models}

    df_metadata = pd.DataFrame([{
        'tid': dataset_id,
        'name': dataset_name,
        'task_type': "TaskType.SUPERVISED_CLASSIFICATION",
    }
        for dataset_id, dataset_name in zip(dataset_ids, dataset_names)
    ])
    df_results_by_dataset = pd.DataFrame({
         "dataset": f"{dataset_id}_{fold}",
         "framework": model,
         "problem_type": "regression",
         "fold": fold,
         "tid": dataset_id,
         **make_random_metric()
     } for fold in range(n_folds) for model in models for (dataset_id, dataset_name) in zip(dataset_ids, dataset_names)
     )

    df_results_by_dataset_automl = pd.DataFrame({
         "dataset": f"{dataset_id}_{fold}",
         "framework": baseline,
         "problem_type": "regression",
         "fold": fold,
         "tid": dataset_id,
        **make_random_metric()
     } for fold in range(n_folds) for baseline in baselines for (dataset_id, dataset_name) in zip(dataset_ids, dataset_names)
     )
    df_raw = pd.DataFrame({
         "dataset": dataset_name,
         "framework": baseline,
         "problem_type": "regression",
         "fold": fold,
         "tid": dataset_id,
         "tid_new": f"{dataset_id}_{fold}",
          **make_random_metric()
     } for fold in range(n_folds) for baseline in baselines for (dataset_id, dataset_name) in zip(dataset_ids, dataset_names)
     )
    zsc = ZeroshotSimulatorContext(
        df_results_by_dataset=df_results_by_dataset,
        df_results_by_dataset_automl=df_results_by_dataset_automl,
        df_raw=df_raw,
        folds=list(range(n_folds)),
        df_metadata=df_metadata,
    )
    pred_dict = {
        dataset_id: {
            fold: {
                "pred_proba_dict_val": {
                    m: np.random.rand(123, 25)
                    for m in models
                },
                "pred_proba_dict_test": {
                    m: np.random.rand(13, 25)
                    for m in models
                }
            }
            for fold in range(n_folds)
        }
        for dataset_id in dataset_ids
    }
    zeroshot_pred_proba = TabularPicklePredictions.from_dict(pred_dict)

    zeroshot_gt = {
        dataset_id: {
            fold: {
                'y_val': pd.Series(np.random.randint(low=0, high=25, size=123)),
                'y_test': pd.Series(np.random.randint(low=0, high=25, size=13)),
                'eval_metric': 'root_mean_squared_error',
                'problem_type': 'regression',
                'problem_type_transform': 'regression',
                'task': 'abalone',
            }
            for fold in range(n_folds)
        }
        for dataset_id in dataset_ids
    }

    return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt


if __name__ == '__main__':
    load_context_artificial()
