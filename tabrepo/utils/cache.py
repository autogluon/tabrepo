from typing import Union

import pandas as pd
import pickle
import sys
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
from dataclasses import dataclass
from typing import Callable, Optional

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl

from tabrepo.utils import catchtime

default_cache_path = Path("~/cache-zeroshot/").expanduser()
default_cache_path.mkdir(parents=True, exist_ok=True)


def cache_function(
        fun: Callable[[], object],
        cache_name: str,
        ignore_cache: bool = False,
        cache_path: Optional[Path] = None
):
    f"""
    :param fun: a function whose result obtained `fun()` will be cached, the output of the function must be serializable.
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.pkl`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = default_cache_path
    cache_file = cache_path / (cache_name + ".pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.loads(f.read())
    else:
        print(f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file")
        with catchtime("Evaluate function."):
            res = fun()
            with open(cache_file, "wb") as f:
                cache = pickle.dumps(res)
                print(f'Writing cache with size {round(sys.getsizeof(cache) / 1e6, 3)} MB')
                f.write(cache)
            return res


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    cache_path: Path | str,
    ignore_cache: bool = False,
) -> pd.DataFrame:
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv`
    :param cache_path: folder where to write cache files
    :param ignore_cache: whether to recompute even if the cache is present
    :return: result of fun()
    """
    cache_file = Path(cache_path) / (cache_name + ".csv")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file")
        with catchtime("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


@contextmanager
def catchtime(name: str, logger=None) -> float:
    start = perf_counter()
    print_fun = print if logger is None else logger.info
    try:
        print_fun(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print_fun(f"Time for {name}: {perf_counter() - start:.4f} secs")


@dataclass
class Experiment:
    expname: str  # name of the parent experiment used to store the file
    name: str  # name of the specific experiment, e.g. "localsearch"
    run_fun: Callable[[], list]  # function to execute to obtain results

    def data(self, ignore_cache: bool = False):
        return cache_function_dataframe(
            lambda: pd.DataFrame(self.run_fun()),
            cache_name=self.name,
            cache_path=self.expname,
            ignore_cache=ignore_cache,
        )


@dataclass
class DummyExperiment(Experiment):
    """
    Dummy Experiment class that doesn't perform caching and simply runs the run_fun and returns the result.
    """

    def data(self, ignore_cache: bool = False):
        return self.run_fun()



class SaveLoadMixin:
    """
    Mixin class to add generic pickle save/load methods.
    """
    def save(self, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        save_pkl.save(path=path, object=self)

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        obj = load_pkl.load(path=path)
        assert isinstance(obj, cls)
        return obj


if __name__ == '__main__':
    def f():
        import time
        time.sleep(0.5)
        return [1, 2, 3]

    res = cache_function(f, "f", ignore_cache=True)
    assert res == [1, 2, 3]

    res = cache_function(f, "f", ignore_cache=False)
    assert res == [1, 2, 3]
