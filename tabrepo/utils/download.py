
import pathlib
import os
import urllib.request
from tqdm import tqdm


def download_files(remote_to_local_tuple_list: list, dry_run: bool = False, verbose: bool = False):
    num_files = len(remote_to_local_tuple_list)
    for i in tqdm(range(num_files), disable=(not verbose) or dry_run, desc="Downloading Files"):
        remote_path, local_path = remote_to_local_tuple_list[i]
        if dry_run:
            print(f'Dry Run: Would download file "{remote_path}" to "{local_path}"')
            continue
        directory = os.path.dirname(local_path)
        if directory not in ["", "."]:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(remote_path, local_path)
