from __future__ import annotations

import pandas as pd
import numpy as np

from typing import List, Tuple, Dict

from autogluon.core.metrics import get_metric, Scorer
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.common.features.feature_metadata import FeatureMetadata

from scripts.leakage_benchmark.src.config_and_data_utils import L1_PREFIX
from scripts.leakage_benchmark.src.custom_metadata_funcs import _sub_sample, _find_optimal_threshold, \
    _get_leaf_node_view, _get_leaf_duplicated_view, _all_wrong_count


def obtain_input_data_for_l2(repo: EvaluationRepository, l1_models: List[str], dataset: str, fold: int) \
        -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, Scorer, List[str], pd.DataFrame, FeatureMetadata]:
    """
    Obtain the input data for the next stacking layer from the repository.
    Additionally, return the eval metric for this fold and dataset combination.

    :param repo: EvaluationRepository
    :param l1_models: List of models in layer 1 to use for leakage analysis.
    :param dataset: Dataset to use for leakage analysis.
    :param fold: outer fold used in this analysis
    :return: l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names
    """

    # Simulation Setup
    tid = repo.dataset_to_tid(dataset)
    task = repo.task_name(tid=tid, fold=fold)
    zsc = repo._zeroshot_context
    tid = zsc.dataset_name_to_tid_dict[task]
    task_ground_truth_metadata: dict = repo._ground_truth[tid][fold]

    problem_type = task_ground_truth_metadata['problem_type']
    metric_name = task_ground_truth_metadata['eval_metric']
    eval_metric = get_metric(metric=metric_name, problem_type=problem_type)
    eval_metric.problem_type = problem_type

    # - Obtain X and y
    # - Obtain X
    train_data, test_data = repo.get_data(tid, fold)
    l2_X_train, l2_y_train, l2_X_test, l2_y_test, l1_feature_metadata \
        = repo.preprocess_data(tid, fold, train_data, test_data, reset_index=True)

    # Previous code had `y_test.fillna(-1)` in code. Not sure why, hence see where this happens with the assert.
    assert l2_y_test.hasnans is False
    l2_y_train = l2_y_train.to_numpy()
    l2_y_test = l2_y_test.fillna(-1).to_numpy()

    # - Obtain preds and build stack_X_train, stack_X_test
    pred_val, pred_test = repo._tabular_predictions.predict(dataset=tid, fold=fold, splits=['val', 'test'],
                                                            models=l1_models, force_1d=problem_type == 'binary')
    oof_col_names = []
    for i, m in enumerate(l1_models):

        if problem_type in ['binary', 'regression']:
            pred_val_m = pred_val[i]
            pred_test_m = pred_test[i]
            col_name = f'{L1_PREFIX}{m}'

            l2_X_train[col_name] = pred_val_m
            l2_X_test[col_name] = pred_test_m
            oof_col_names.append(col_name)
        elif problem_type in ['multiclass']:
            classes = task_ground_truth_metadata['ordered_class_labels_transformed']

            preds_val_m = pred_val[i]
            preds_test_m = pred_test[i]
            col_names = [f'{L1_PREFIX}{m}/c{c_name}' for c_name in classes]

            l2_X_train[col_names] = preds_val_m
            l2_X_test[col_names] = preds_test_m
            oof_col_names.extend(col_names)
        else:
            raise NotImplementedError(f'Problem type {problem_type} not supported.')

    # Get L1 scores
    leaderboard = zsc.df_results_by_dataset_vs_automl.loc[(zsc.df_results_by_dataset_vs_automl['dataset'] == task) & (
        zsc.df_results_by_dataset_vs_automl['framework'].isin(l1_models)), ['framework', 'metric_error', 'score_val']]
    leaderboard['metric_error'] = leaderboard['metric_error'].apply(lambda x: eval_metric.optimum - x)
    leaderboard = leaderboard.rename(columns={'framework': 'model', 'metric_error': 'score_test'}).reset_index(
        drop=True)

    return l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names, leaderboard, l1_feature_metadata


def _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata):
    # Follow autogluon/core/src/autogluon/core/models/ensemble/stacker_ensemble_model.py _add_stack_to_feature_metadata
    #   -- No additional preprocessing from our side here. If at all, this happens at the model level.
    from autogluon.common.features.types import R_FLOAT, S_STACK

    type_map_raw = {column: R_FLOAT for column in oof_col_names}
    type_group_map_special = {S_STACK: oof_col_names}
    stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
    l2_feature_metadata = l1_feature_metadata.join_metadata(stacker_feature_metadata)

    return l2_feature_metadata


def _get_meta_data(l2_train_data, l2_test_data, label, oof_col_names, problem_type, eval_metric):
    # Init
    f_dup = oof_col_names + [label]
    f_l_dup = oof_col_names
    train_n_instances = len(l2_train_data)
    n_columns = len(l2_train_data.columns)
    test_n_instances = len(l2_test_data)

    X_train = l2_train_data.drop(columns=[label])
    y_train = l2_train_data[label]
    X_test = l2_test_data.drop(columns=[label])
    y_test = l2_test_data[label]

    # Compute metadata
    custom_meta_data = dict(
        train_l2_duplciates=sum(l2_train_data.duplicated()) / train_n_instances,
        train_feature_duplciates=sum(l2_train_data.drop(columns=f_dup).duplicated()) / train_n_instances,
        train_feature_label_duplicates=sum(l2_train_data.drop(columns=f_l_dup).duplicated()) / train_n_instances,
        test_l2_duplicates=sum(l2_test_data.duplicated()) / test_n_instances,
        test_feature_duplciates=sum(l2_test_data.drop(columns=f_dup).duplicated()) / test_n_instances,
        test_feature_label_duplicates=sum(l2_test_data.drop(columns=f_l_dup).duplicated()) / test_n_instances,

        # Unique
        train_unique_vlaues_per_oof=[(col, len(np.unique(l2_train_data[col])) / train_n_instances) for col in
                                     oof_col_names],
        test_unique_vlaues_per_oof=[(col, len(np.unique(l2_test_data[col])) / test_n_instances) for col in
                                    oof_col_names],
        train_duplicated_columns=sum(l2_train_data.T.duplicated()) / n_columns,
        test_duplicated_columns=sum(l2_test_data.T.duplicated()) / n_columns,

        # Basic properties
        train_n_instances=train_n_instances,
        test_n_instances=test_n_instances,
        n_columns=n_columns,
        problem_type=problem_type,
        eval_metric_name=eval_metric.name,

    )

    if problem_type == 'binary':
        custom_meta_data['oof_col_names_order'] = oof_col_names

        custom_meta_data['optimal_threshold_train_per_oof'] = \
            [_find_optimal_threshold(l2_train_data[label], l2_train_data[col]) for col in oof_col_names]
        custom_meta_data['optimal_threshold_test_per_oof'] = \
            [_find_optimal_threshold(l2_test_data[label], l2_test_data[col]) for col in oof_col_names]
        custom_meta_data['always_wrong_row_ratio_train'] = _all_wrong_count(X_train, y_train, oof_col_names,
                                                                      threshold=np.mean(custom_meta_data['optimal_threshold_train_per_oof']))
        custom_meta_data['always_wrong_row_ratio_test'] = _all_wrong_count(X_test, y_test, oof_col_names,
                                                                            threshold=np.mean(custom_meta_data['optimal_threshold_test_per_oof']))

        custom_meta_data['potential_for_cheat_stats_tree_view'] = \
            _get_leaf_node_view(X_train, y_train, X_test, y_test, min_samples_leaf=1, problem_type=problem_type, oof_col_names=oof_col_names)

        custom_meta_data['potential_for_cheat_stats_duplicates_view'] = \
            _get_leaf_duplicated_view(X_train, y_train, X_test, y_test, oof_col_names=oof_col_names)
        # custom_meta_data['potential_for_cheat_stats_cv'] = \
        #     _cv_wrapper_avg_cheat(X_train, y_train, min_samples_leaf=1, problem_type=problem_type, oof_col_names=oof_col_names )

    return custom_meta_data


def autogluon_l2_runner(l2_models, l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric: Scorer,
                        oof_col_names: List[str], l1_feature_metadata: FeatureMetadata, get_meta_data: bool = True,
                        sub_sample_data: bool = False, problem_type: str | None = None) -> Tuple[pd.DataFrame, Dict]:
    print(f"Start preprocessing L2 data and collect metadata. {l2_X_train.shape}")
    label = "class"
    l2_feature_metadata = _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata)

    # TODO: test performance of something like this as additional feature even without the leak
    #   Following wolpert, the idea is that l2 models learn from knowing the distance to the nearest neighbor
    #   Downside is that it is quite expensive I guess. (add at fit time not here)
    # _compute_nearest_neighbor_distance(l2_X_train.drop(columns=oof_col_names), l2_y_train,
    #                                    l2_X_test.drop(columns=oof_col_names))

    # _plot_problematic_instances(l2_X_train[[f'{L1_PREFIX}RandomForest_c1_BAG_L1']], l2_y_train)

    # Build data
    l2_train_data = l2_X_train
    l2_train_data[label] = l2_y_train
    l2_test_data = l2_X_test
    l2_test_data[label] = l2_y_test

    if get_meta_data:
        custom_meta_data = _get_meta_data(l2_train_data, l2_test_data, label, oof_col_names, problem_type, eval_metric)
    else:
        custom_meta_data = {}

    if sub_sample_data:
        l2_train_data, l2_test_data = _sub_sample(l2_train_data, l2_test_data)

    # Run AutoGluon
    print("Start running AutoGluon on L2 data.")
    predictor = TabularPredictor(eval_metric=eval_metric.name, label=label, verbosity=0, problem_type=problem_type,
                                 learner_kwargs=dict(random_state=1))
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters=l2_models,
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=8,
        feature_generator=IdentityFeatureGenerator(),
        feature_metadata=l2_feature_metadata,
        # ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}
    )

    leaderboard_leak = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']]
    leaderboard_leak['model'] = leaderboard_leak['model'].apply(lambda x: x.replace('L1', 'L2'))

    return leaderboard_leak, custom_meta_data
