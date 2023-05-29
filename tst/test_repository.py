import numpy as np

from autogluon_zeroshot.contexts.context_artificial import load_context_artificial
from autogluon_zeroshot.repository import EvaluationRepository


def test_repository():

    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
    repo = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )
    dataset_name = repo.dataset_names()[0]
    config_name = "NeuralNetFastAI_r1"  # TODO accessor


    assert repo.dataset_names() == ['abalone', 'ada']
    assert repo.task_ids() == [359944, 359946]
    assert repo.n_folds() == 3
    assert repo.dataset_to_taskid(dataset_name) == 359946
    assert repo.list_models_available(dataset_name) == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.eval_metrics(dataset_name=dataset_name, config_names=[config_name], fold=2))
    assert repo.val_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (123, 25)
    assert repo.test_predictions(dataset_name=dataset_name, config_name=config_name, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset_name=dataset_name) == {'tid': 359946, 'name': dataset_name, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name], ensemble_size=5, backend="native"),
        [[2.5, 2.5, 2.5]]
    )
    assert np.allclose(
        repo.evaluate_ensemble(dataset_names=[dataset_name], config_names=[config_name, config_name],
                                 ensemble_size=5, folds=[2], backend="native"),
        [[2.5]]
    )