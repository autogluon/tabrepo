from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import tqdm


def fetch_all_pickles(
    dir_path: str | Path | list[str | Path], suffix: str = ".pkl"
) -> list[Path]:
    """Recursively find every file ending in “.pkl” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | Path | list[str | Path]
        Root directory to search.
        If a list of directories, will search over all directories.

    Returns:
    -------
    list[Path]
        A list of paths to .pkl files.

    Notes:
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    if not isinstance(dir_path, list):
        dir_path = [dir_path]

    file_paths: list[Path] = []
    for cur_dir_path in dir_path:
        root = Path(cur_dir_path).expanduser().resolve()
        if not root.is_dir():
            if root.is_file():
                assert str(root).endswith(suffix), f"{root} is a file that does not end in `{suffix}`."
                file_paths.append(root)
            else:
                raise NotADirectoryError(f"{root} is not a directory")
        else:
            # Look for *.pkl
            pattern = f"*{suffix}"
            for file_path in tqdm.tqdm(
                root.rglob(pattern), desc=f"Searching for pickles in {cur_dir_path}"
            ):
                file_paths.append(file_path)

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
