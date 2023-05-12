import pandas as pd
import numpy as np

from autogluon_zeroshot.utils.normalized_scorer import NormalizedScorer

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


def test_normalized_scorer():
    scorer = NormalizedScorer(
        df_results_by_dataset=df_results_by_dataset,
        datasets=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        dataset_col=dataset_col,
        framework_col=framework_col,
    )
    query_expected = [
        (1.0, 0.0),
        (2.0, 1.0),
        (1.5, 0.5),
        (3.0, 1.0),
        (0.0, 0.0),
    ]
    for query, expected in query_expected:
        print(scorer.rank("dataset1", query))
        assert np.isclose(scorer.rank("dataset1", query), expected)