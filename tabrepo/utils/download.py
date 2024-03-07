
import pathlib
import os
import urllib.request
from tqdm import tqdm

from tabrepo.utils.parallel_for import parallel_for


def download_files(remote_to_local_tuple_list: list, dry_run: bool = False, verbose: bool = False):
    def download_file(remote_path, local_path, dry_run):
        if dry_run:
            print(f'Dry Run: Would download file "{remote_path}" to "{local_path}"')
            return
        directory = os.path.dirname(local_path)
        if directory not in ["", "."]:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(remote_path, local_path)

    parallel_for(download_file, inputs=remote_to_local_tuple_list, context={"dry_run": dry_run})