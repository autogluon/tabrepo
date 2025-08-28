from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def fetch_all_pickles(dir_path: str | Path, suffix: str = ".pkl") -> list[Path]:
    """
    Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns
    -------
    List[Any]
        A list whose elements are the Python objects obtained from each
        successfully un‑pickled file, in depth‑first lexical order.

    Notes
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    root = Path(dir_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    file_paths: list[Path] = []

    # Look for *.pkl, case‑insensitive
    patterns = (f"*{suffix}",)
    i = 0
    for pattern in patterns:
        pattern_suffix = pattern[1:]
        for file_path in root.rglob(pattern):
            if not str(file_path).endswith(pattern_suffix):
                continue
            if file_path.is_file():
                i += 1
                if i % 10000 == 0:
                    print(i, file_path)
                file_paths.append(file_path)

    return file_paths


def load_all_pickles(dir_path: str | Path) -> list[Any]:
    """
    Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns
    -------
    List[Any]
        A list whose elements are the Python objects obtained from each
        successfully un‑pickled file, in depth‑first lexical order.

    Notes
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    file_paths = fetch_all_pickles(dir_path=dir_path)
    loaded_objects = []

    for file_path in file_paths:
        with file_path.open("rb") as f:
            loaded_objects.append(pickle.load(f))
    return loaded_objects
