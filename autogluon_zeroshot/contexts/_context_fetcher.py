from typing import Dict, List

from .context import BenchmarkContext
from .context_2023_03_19_bag_289 import context_bag_244_bench_50_mb, context_bag_244_bench_10_mb, \
    context_bag_289_bench_50_mb, context_bag_289_bench_10_mb
from .context_2022_12_11_bag import context_bag_104_bench, context_bag_104_bench_small
from .context_2022_10_13 import context_104_bench


__all__ = [
    'list_contexts',
    'get_context',
]


_context_map: Dict[str, BenchmarkContext] = dict()

for c in [
    context_bag_244_bench_10_mb,
    context_bag_244_bench_50_mb,
    context_bag_289_bench_10_mb,
    context_bag_289_bench_50_mb,
    context_bag_104_bench,
    context_bag_104_bench_small,
    context_104_bench,
]:
    if c.name in _context_map:
        raise AssertionError(f'ERROR: Multiple contexts have the same name: {c.name}')
    _context_map[c.name] = c


def list_contexts() -> List[str]:
    return sorted(list(_context_map.keys()))


def get_context(name: str) -> BenchmarkContext:
    if name not in _context_map:
        available_names = list_contexts()
        raise ValueError(f'Could not find context with name="{name}". '
                         f'Valid names: {available_names}')
    return _context_map[name]
