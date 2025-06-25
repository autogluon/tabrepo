from __future__ import annotations

import os
import shutil
from pathlib import Path

from autogluon.common.utils.s3_utils import upload_file


class MethodUploader:
    def __init__(
        self,
        method: str,
        bucket: str,
        prefix: str,
        local_path: str | Path,
        upload_as_public: bool = False,
    ):
        self.method = method
        self.bucket = bucket
        self.prefix = prefix
        self.local_path = Path(local_path)
        self.upload_as_public = upload_as_public

    def _local_path_raw(self) -> Path:
        return self.local_path / "raw"

    def _local_path_processed(self) -> Path:
        return self.local_path / "processed"

    def _local_path_results(self) -> Path:
        return self.local_path / "results"

    def upload_raw(self):
        path_data_method = self._local_path_raw()

        tmp_dir = Path("tmp")
        file_prefix = tmp_dir / self.method / "raw"
        shutil.make_archive(file_prefix, 'zip', root_dir=path_data_method)
        file_name = f"{file_prefix}.zip"

        self._upload_file(file_name=file_name, prefix=self.prefix)
        # os.remove(file_name)

    def _upload_file(self, file_name: str | Path, prefix: str):
        kwargs = {}
        if self.upload_as_public:
            kwargs = {"ExtraArgs": {"ACL": "public-read"}}
        upload_file(file_name=file_name, bucket=self.bucket, prefix=prefix, **kwargs)

    def upload_processed(self):
        path_data_method = self._local_path_processed()

        tmp_dir = Path("tmp")
        file_prefix = tmp_dir / self.method / "processed"
        shutil.make_archive(file_prefix, 'zip', root_dir=path_data_method)
        file_name = f"{file_prefix}.zip"

        self._upload_file(file_name=file_name, prefix=self.prefix)
        # os.remove(file_name)

    def upload_results(self):
        path_method_results = self._local_path_results()
        prefix = f"{self.prefix}/results"

        file_names = [
            "model_results.parquet",
            "hpo_results.parquet",
        ]
        for file_name in file_names:
            local_file_path = path_method_results / file_name
            self._upload_file(file_name=local_file_path, prefix=prefix)
