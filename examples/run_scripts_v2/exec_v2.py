from tabpfn import TabPFNClassifier
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.core.metrics import get_metric, Scorer
import pandas as pd
from autogluon_benchmark.utils.time_utils import Timer
from autogluon_benchmark.frameworks.autogluon.run import ag_eval_metric_map
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import generate_train_test_split


def fit_outer(task, fold: int, task_name: str, method: str, init_args: dict = None, **kwargs):
    if init_args is None:
        init_args = {}
    if 'eval_metric' not in init_args:
        init_args['eval_metric'] = ag_eval_metric_map[task.problem_type]

    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold)

    out = fit_custom_clean(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                           problem_type=task.problem_type, eval_metric=init_args['eval_metric'], label=task.label)

    out["framework"] = method
    out["dataset"] = task_name
    out["tid"] = task.task_id
    out["fold"] = fold
    out["problem_type"] = task.problem_type
    print(f"Task  Name: {out['dataset']}")
    print(f"Task    ID: {out['tid']}")
    print(f"Metric    : {out['eval_metric']}")
    print(f"Test Error: {out['test_error']:.4f}")
    print(f"Fit   Time: {out['time_fit']:.3f}s")
    print(f"Infer Time: {out['time_predict']:.3f}s")

    out.pop("predictions")
    out.pop("probabilities")
    out.pop("truth")

    df_results = pd.DataFrame([out])
    ordered_columns = ["dataset", "fold", "framework", "test_error", "eval_metric", "time_fit"]
    columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
    df_results = df_results[columns_reorder]
    return df_results


# TODO: Nick: This works for 99.99% of cases, but to handle all possible edge-cases,
#  we probably want to use Tabular's LabelCleaner during metric calculation to avoid any oddities.
#  This can be done as a follow-up
#  We also need to track positive_class for binary classification
def calc_error(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.DataFrame,
    problem_type: str,
    scorer: Scorer,
) -> float:
    if scorer.needs_pred:  # use y_pred
        error = scorer.error(y_true=y_true, y_pred=y_pred)
    elif problem_type == "binary":  # use y_pred_proba
        error = scorer.error(y_true=y_true, y_pred=y_pred_proba.iloc[:, 1], labels=y_pred_proba.columns)
    else:
        error = scorer.error(y_true=y_true, y_pred=y_pred_proba, labels=y_pred_proba.columns)
    return error


def fit_custom_clean(X_train, y_train, X_test, y_test, problem_type: str = None, eval_metric: str = None, **kwargs):
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train, y_uncleaned=y_train)
    y_train_clean = label_cleaner.transform(y_train)
    y_test_clean = label_cleaner.transform(y_test)

    # TODO: Nick: For now, I'm preprocessing via AutoGluon's feature generator because otherwise TabPFN crashes on some datasets.
    feature_generator = AutoMLPipelineFeatureGenerator()
    X_train_clean = feature_generator.fit_transform(X=X_train, y=y_train)
    X_test_clean = feature_generator.transform(X=X_test)

    out = fit_custom(
        X_train=X_train_clean,
        y_train=y_train_clean,
        X_test=X_test_clean,
        y_test=y_test_clean,
        problem_type=problem_type,
        **kwargs,
    )

    y_pred_test_clean = out["predictions"]
    y_pred_proba_test_clean = out["probabilities"]

    scorer: Scorer = get_metric(metric=eval_metric, problem_type=problem_type)

    test_error = calc_error(
        y_true=y_test_clean,
        y_pred=y_pred_test_clean,
        y_pred_proba=y_pred_proba_test_clean,
        problem_type=problem_type,
        scorer=scorer,
    )

    y_pred_test = label_cleaner.inverse_transform(y_pred_test_clean)
    out["predictions"] = y_pred_test

    if y_pred_proba_test_clean is not None:
        y_pred_proba_test = label_cleaner.inverse_transform_proba(y_pred_proba_test_clean, as_pandas=True)
        out["probabilities"] = y_pred_proba_test

    out["test_error"] = test_error
    out["eval_metric"] = scorer.name
    out["truth"] = y_test

    return out


def fit_custom(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = None,
    label: str = None,
) -> dict:

    # FIXME: Nick: This is a hack specific to TabPFN, since it doesn't handle large data, parameterize later
    sample_limit = 4096
    if len(X_train) > sample_limit:
        X_train, _, y_train, _ = generate_train_test_split(
            X=X_train,
            y=y_train,
            problem_type=problem_type,
            train_size=sample_limit,
            random_state=0,
            min_cls_count_train=1,
        )

    # with Timer() as timer_fit:
    #     model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32).fit(X_train, y_train, overwrite_warning=True)

    from tabpfn_client.estimator import TabPFNClassifier as TabPFNClassifierV2, TabPFNRegressor
    model = TabPFNClassifierV2(model="latest_tabpfn_hosted", n_estimators=32)
    with Timer() as timer_fit:
        model = model.fit(X_train, y_train)

    is_classification = problem_type in ['binary', 'multiclass']
    if is_classification:
        with Timer() as timer_predict:
            y_pred_proba = model.predict_proba(X_test)
            y_pred_proba = pd.DataFrame(y_pred_proba, columns=model.classes_, index=X_test.index)
        y_pred = y_pred_proba.idxmax(axis=1)
    else:
        with Timer() as timer_predict:
            y_pred = model.predict(X_test)
            y_pred = pd.Series(y_pred, name=label, index=X_test.index)
        y_pred_proba = None

    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'time_fit': timer_fit.duration,
        'time_predict': timer_predict.duration,
    }
