from __future__ import annotations

import pandas as pd

from tabarena.plot.interactive.leaderboard_table import leaderboard_table_points


def test_points_flag_a_train_time_that_includes_the_prediction_call():
    df = pd.DataFrame(
        {
            "#": [0, 1],
            "Model": ["A (default)", "B"],
            "TypeName": ["Tree-based", "System"],
            "Elo [⬆️]": [1500.0, 1400.0],
            "FitsAtPredict": [False, True],
        }
    )
    assert leaderboard_table_points(df)["fits_at_predict"].tolist() == [False, True]
    # Tables written before the column existed have no such method.
    assert leaderboard_table_points(df.drop(columns="FitsAtPredict"))["fits_at_predict"].tolist() == [False, False]
