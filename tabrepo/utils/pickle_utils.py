from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def fetch_all_pickles(dir_path: str | Path, suffix: str = ".pkl") -> list[Path]:
    """Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns:
    -------
    List[Any]
        A list whose elements are the Python objects obtained from each
        successfully un‑pickled file, in depth‑first lexical order.

    Notes:
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


def fetch_all_pickles_fast(
    dir_path: str | Path, suffix: str = ".pkl"
) -> dict[str, list[Path]]:
    """Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns:
    -------
    file_paths : dict[str, list[Paths]]
        List of file paths for each dataset-split-id combination.

    Notes:
    -----
    Never un-pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    import tqdm

    root = Path(dir_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    file_paths: dict[str, list[Path]] = {}

    print("Root dir:", root)
    for file_path in tqdm.tqdm(
        root.glob(f"*/*/*/{suffix}"), desc="Searching for pickles"
    ):
        did_sid = f"{file_path.parts[-3]}/{file_path.parts[-2]}"
        if did_sid not in file_paths:
            file_paths[did_sid] = []
        file_paths[did_sid].append(file_path)

    return file_paths

def load_all_pickles(dir_path: str | Path) -> list[Any]:
    """Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns:
    -------
    List[Any]
        A list whose elements are the Python objects obtained from each
        successfully un‑pickled file, in depth‑first lexical order.

    Notes:
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
