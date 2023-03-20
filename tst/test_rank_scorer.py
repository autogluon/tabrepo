import pandas as pd

from autogluon_zeroshot.utils.rank_utils import RankScorer

dataset_col = "dataset"
metric_col = "metric"
framework_col = "framework"

df_results_by_dataset = pd.DataFrame([
    ["dataset1", "xgboost1", 1.0],
    ["dataset1", "xgboost3", 3.0],
    ["dataset1", "xgboost2", 2.0],
    ["dataset2", "xgboost1", 10.0],
    ["dataset2", "xgboost3", 30.0],
    ["dataset2", "xgboost2", 20.0],
],
    columns=[dataset_col, framework_col, metric_col]
)


def test_rank_scorer():
    rank_scorer = RankScorer(
        df_results_by_dataset=df_results_by_dataset,
        datasets=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        dataset_col=dataset_col,
        framework_col=framework_col,
    )
    query_expected = [
        (0.8, 1.0),
        (1.0, 1.5),
        (1.5, 2),
        (2.0, 2.5),
        (4.0, 4),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("dataset1", query) == expected


def test_rank_scorer_pct():
    rank_scorer = RankScorer(
        df_results_by_dataset=df_results_by_dataset,
        datasets=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        dataset_col=dataset_col,
        framework_col=framework_col,
        pct=True,
    )
    query_expected = [
        (0.8, 0.0),
        (1.0, 1/6),
        (1.5, 1/3),
        (2.0, 1/2),
        (2.5, 2/3),
        (3.0, 5/6),
        (4.0, 1.0),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("dataset1", query) == expected


def test_rank_scorer_ties_win():
    rank_scorer = RankScorer(
        df_results_by_dataset=df_results_by_dataset,
        datasets=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        dataset_col=dataset_col,
        framework_col=framework_col,
        ties_win=True,
        pct=False,
    )
    query_expected = [
        (0.8, 1),
        (1.0, 1),
        (1.5, 2),
        (2.0, 2),
        (4.0, 4),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("dataset1", query) == expected


def test_rank_scorer_pct_ties_win():
    rank_scorer = RankScorer(
        df_results_by_dataset=df_results_by_dataset,
        datasets=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        dataset_col=dataset_col,
        framework_col=framework_col,
        ties_win=True,
        pct=True,
    )
    query_expected = [
        (0.8, 0.0),
        (1.0, 0.0),
        (1.5, 1/3),
        (2.0, 1/3),
        (3.0, 2/3),
        (4.0, 1.0),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("dataset1", query) == expected
