import pickle
from pathlib import Path
from typing import Any


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
    root = Path(dir_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    loaded_objects: list[Any] = []

    # Look for *.pkl and *.pickle, case‑insensitive
    patterns = ("*.pkl", "*.pickle")
    for pattern in patterns:
        for file_path in root.rglob(pattern):
            if file_path.is_file():
                with file_path.open("rb") as f:
                    loaded_objects.append(pickle.load(f))

    return loaded_objects
