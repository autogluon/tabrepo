import pandas as pd
from autogluon_zeroshot.utils.cache import cache_function, cache_function_dataframe


def test_cache_pickle():
    def f():
        return [1, 2, 3]

    for ignore_cache in [True, False]:
        res = cache_function(f, "f", ignore_cache=ignore_cache)
        assert res == [1, 2, 3]


def test_cache_dataframe():

    def f():
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    for ignore_cache in [True, False]:
        res = cache_function_dataframe(f, "f", ignore_cache=ignore_cache)
        pd.testing.assert_frame_equal(res, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
