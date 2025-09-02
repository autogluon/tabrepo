from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata


class MethodDownloaderS3:
    """
    Download a method's cached artifacts from S3 and restore the original local layout
    expected by MethodUploaderS3 / MethodMetadata.

    Artifacts handled:
      - metadata YAML
      - raw.zip  -> extracted into `method_metadata.path_raw`
      - processed.zip -> extracted into `method_metadata.path_processed`
      - configs_hyperparameters (standalone YAML/JSON/etc. per your metadata)
      - results files (the set returned by `method_metadata.path_results_files()`)

    Notes
    -----
    - S3 keys are reconstructed from the local target paths via MethodMetadata.to_s3_cache_loc
      so the local <-> S3 mapping stays perfectly symmetric with MethodUploaderS3.
    - If an optional artifact is missing on S3, it is skipped with a warning (no exception).
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        bucket: str,
        s3_prefix_root: str = "cache",
        verbose: bool = True,
        clear_dirs: bool = True,
    ):
        self.method_metadata = method_metadata
        self.method = method_metadata.method
        self.bucket = bucket
        self.s3_prefix_root = s3_prefix_root
        self.prefix = Path(self.s3_prefix_root) / method_metadata.relative_to_cache_root(method_metadata.path)
        self.verbose = verbose
        self.clear_dirs = clear_dirs

    # --------------------
    # Properties / mapping
    # --------------------
    @property
    def s3_cache_root(self) -> str:
        return f"s3://{self.bucket}/{self.s3_prefix_root}"

    def local_to_s3_path(self, path_local: str | Path) -> str:
        s3_path_loc = self.method_metadata.to_s3_cache_loc(path=Path(path_local), s3_cache_root=self.s3_cache_root)
        _, s3_key = s3_path_to_bucket_prefix(s3_path_loc)
        return s3_key

    # ---------------
    # Public entrypoint
    # ---------------
    def download_all(self):
        self.download_metadata()
        self.download_raw()
        self.download_processed()
        self.download_results()

    # --------
    # Downloads
    # --------
    def download_metadata(self):
        path_local = Path(self.method_metadata.path_metadata)
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    def download_raw(self):
        dest_dir = Path(self.method_metadata.path_raw)
        s3_key = (self.prefix / "raw.zip").as_posix()
        self._download_and_unzip_if_exists(s3_key=s3_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_processed(self):
        dest_dir = Path(self.method_metadata.path_processed)
        s3_key = (self.prefix / "processed.zip").as_posix()
        self._download_and_unzip_if_exists(s3_key=s3_key, dest_dir=dest_dir, clear_dir=self.clear_dirs)

    def download_configs_hyperparameters(self, holdout: bool = False):
        path_local = Path(self.method_metadata.path_configs_hyperparameters(holdout=holdout))
        s3_key = self.local_to_s3_path(path_local=path_local)
        self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    def download_results(self, holdout: bool = False):
        file_names: Iterable[Path | str] = self.method_metadata.path_results_files(holdout=holdout)
        for path_local in file_names:
            path_local = Path(path_local)
            s3_key = self.local_to_s3_path(path_local=path_local)
            self._download_to_local_if_exists(s3_key=s3_key, path_local=path_local)

    # --------------
    # Helper methods
    # --------------
    def _download_to_local_if_exists(self, s3_key: str | Path, path_local: Path):
        """
        Attempts to download a single file to `path_local`. Skips quietly if not found.
        """
        import boto3
        from botocore.exceptions import ClientError

        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        s3 = boto3.client("s3")

        # Check existence first (avoids creating empty files/directories on 404)
        try:
            s3.head_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                if self.verbose:
                    print(f"[WARN] Missing on S3, skipping: s3://{self.bucket}/{s3_key}")
                return
            raise

        path_local.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[INFO] Downloading s3://{self.bucket}/{s3_key} -> {path_local}")
        s3.download_file(Bucket=self.bucket, Key=s3_key, Filename=str(path_local))

    def _download_and_unzip_if_exists(self, s3_key: str | Path, dest_dir: Path, clear_dir: bool = True):
        """
        Downloads a zip from S3 into memory and extracts into `dest_dir`.
        Skips if the zip does not exist on S3.
        """
        import boto3
        from botocore.exceptions import ClientError

        if isinstance(s3_key, Path):
            s3_key = s3_key.as_posix()

        s3 = boto3.client("s3")

        # Existence check
        try:
            s3.head_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                if self.verbose:
                    print(f"[WARN] Missing on S3, skipping unzip: s3://{self.bucket}/{s3_key}")
                return
            raise

        if self.verbose:
            print(f"[INFO] Downloading s3://{self.bucket}/{s3_key} -> extracting to {dest_dir}")

        # Stream into memory
        obj = s3.get_object(Bucket=self.bucket, Key=s3_key)
        body = obj["Body"].read()
        buf = io.BytesIO(body)

        # Clear destination (optional)
        if clear_dir and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Extract
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(path=dest_dir)
