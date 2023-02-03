from contextlib import contextmanager
from time import perf_counter


@contextmanager
def catchtime(name: str, logger = None) -> float:
    start = perf_counter()
    print_fun = lambda x: print if logger is None else logger.info(x)
    try:
        print_fun(f"\nstart: {name}")
        yield lambda: perf_counter() - start
    finally:
        print_fun(f"Time for {name}: {perf_counter() - start:.4f} secs")
