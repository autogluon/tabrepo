import warnings
from tabrepo import load_repository, EvaluationRepository
from context_dl import ContextDataLoader
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, log_loss, \
    precision_score, cohen_kappa_score, average_precision_score
from autogluon.core.metrics import get_metric
import pandas as pd
from autogluon_benchmark.utils.time_utils import Timer
from autogluon.tabular import TabularPredictor
from autogluon.core.data import LabelCleaner
import copy

warnings.simplefilter("ignore")

eval_metric_map = {
    'binary': 'roc_auc',
    'multiclass': 'log_loss',
    'regression': 'rmse',
}


def calculate_classification_metrics(y_true, y_pred, probabilities=None, problem_type=None):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)

    # Log Loss (only if you have probabilities)
    logloss = None
    if probabilities is not None:
        logloss = log_loss(y_true, probabilities)

    quadratic_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    pac = precision_score(y_true, y_pred, average='weighted')  # weighted precision as a placeholder

    # ROC AUC (applicable for binary or multi-class classification)
    roc_auc = None
    if problem_type in ['binary'] and probabilities is not None:
        roc_auc = roc_auc_score(y_true, probabilities)

    # Average Precision (applicable for binary classification or multi-label classification)
    # avg_precision = None
    # if problem_type in ['binary', 'multiclass'] and probabilities is not None:
    #     avg_precision = average_precision_score(y_true, probabilities)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'log_loss': logloss,
        'quadratic_kappa': quadratic_kappa,
        'pac': pac,
        'roc_auc': roc_auc
        # 'avg_precision': avg_precision
    }


def run():
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)
    # ToDo: For testing 146818 - Australian, 359955 - blood-transfusion-service-centre
    tids = [359955]
    folds = [0]

    # Loading the dataset (what we call task) is done by ContextDataLoader
    # ToDo: To Add Random State you can use the train test combined and call the train test split of sklearn?
    results_lst = []
    for tid in tids:
        task = ContextDataLoader.from_task_id(tid)
        task_name = repo.tid_to_dataset(tid=tid)
        eval_metric = repo.dataset_info(task_name)['metric']
        label = task.task.target_name
        for fold in folds:
            X_train, y_train, X_test, y_test = task.get_context_train_test_split(repo=repo, task_id=tid,
                                                                                 fold=fold)
            out = fit_custom(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                             problem_type=task.problem_type, eval_metric=eval_metric, label=label)
            results_lst.append(out)

    print("Results: ", results_lst)


# Separate Func - input and output for this func - output should have all info to get every single columns
# Get a working example of this
# Run.py stuff goes here to add problem type and the whole map
def fit_custom(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
               init_args: dict = None, problem_type: str = None, eval_metric: str = None,
               label: str = None) -> dict:
    if init_args is None:
        init_args = {}
    if problem_type is not None:
        init_args['problem_type'] = problem_type
    if 'eval_metric' not in init_args:
        if init_args.get('problem_type', None):
            init_args['eval_metric'] = eval_metric_map[init_args['problem_type']]

    with Timer() as timer_fit:
        model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32).fit(X_train, y_train)

    is_classification = problem_type in ['binary', 'multiclass']
    if is_classification:
        with Timer() as timer_predict:
            y_pred, y_proba = model.predict(X_test, return_winning_probability=True)
    else:
        with Timer() as timer_predict:
            y_pred = model.predict(X_test, as_pandas=False)
        y_proba = None

    metrics = {}
    scorer = get_metric(metric=eval_metric, problem_type=problem_type)

    # Label Cleaning Logic in-order to apply scorer.error() - START
    predictor = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric)
    X_test = pd.concat([X_test, y_test.to_frame(name=label)], axis=1)
    X_index = copy.deepcopy(X_test.index)
    y = pd.concat([y_train, y_test])
    predictor._learner.label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y, y_uncleaned=y,
                           positive_class=None)
    y_proba = predictor._learner._post_process_predict_proba(y_pred_proba=y_proba, index=X_index)
    # Label Cleaning Logic - END

    if problem_type == 'binary':
        metrics = calculate_classification_metrics(y_true=y_test, y_pred=y_pred, probabilities=y_proba,
                                                   problem_type=problem_type)
    elif problem_type == 'multiclass':
        metrics = calculate_classification_metrics(y_true=y_test, y_pred=y_pred, probabilities=y_proba,
                                                   problem_type=problem_type)

    # ToDo: Regression metrics - metric = getMetric(), and pass the repo.dataset_info() metric.error(y_true, y_pred)
    # ToDo: WHAT ABOUT VALIDATION - We do not need it - leave as empty
    # ToDo: reach out to Xiyuen for TabPFNv2, run lightAutoML xy series

    scorer = get_metric(eval_metric, problem_type)
    scorer.error(y_test, y_proba)

    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'timer_fit': timer_fit,
        'timer_predict': timer_predict,
        'metrics': metrics
    }


if __name__ == '__main__':
    run()
