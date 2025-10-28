"""
Download specified OpenML tasks into the local cache (if not already present)
and sync the entire OpenML cache directory to an S3 prefix.

Requires write permissions to the s3 location.
"""

from __future__ import annotations

import argparse
from tabarena.benchmark.task.openml import OpenMLTaskWrapper, OpenMLS3TaskWrapper
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def main():
    parser = argparse.ArgumentParser(description="Sync cache to S3.")
    parser.add_argument(
        "--s3_path",
        default="s3://tabarena/openml/openml_cache",
        help="Destination S3 URL (bucket/prefix), e.g., s3://my-bucket/my/prefix",
    )
    args = parser.parse_args()
    task_metadata = TabArenaContext().task_metadata
    tids = list(task_metadata["tid"])
    s3_dataset_cache = args.s3_path

    for task_id in tids:
        task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
        task.get_split_indices()  # force OpenML splits file download
        # Cache OpenML files to S3
        OpenMLS3TaskWrapper.update_s3_cache(
            task_id=task.task_id,
            dataset_id=task.dataset_id,
            s3_dataset_cache=s3_dataset_cache,
        )
        # Can now download the task from s3 in future without downloading from OpenML:
        # task = OpenMLS3TaskWrapper.from_task_id(task_id=task_id, s3_dataset_cache=s3_dataset_cache)


if __name__ == "__main__":
    main()
