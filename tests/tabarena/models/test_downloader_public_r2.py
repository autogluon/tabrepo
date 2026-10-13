"""Single-file downloads from public R2 never leave a partial file at the target path (no network involved)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tabarena.models._artifacts.downloader_public_r2 import MethodDownloaderPublicR2
from tabarena.models._method_metadata import MethodMetadata


class _Response:
    """A streamed GET whose chunks run ``on_chunk`` first, so a test can look at the disk mid-download."""

    def __init__(self, chunks, on_chunk):
        self.status_code = 200
        self.chunks = chunks
        self.on_chunk = on_chunk

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for chunk in self.chunks:
            self.on_chunk()
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _download(tmp_path, chunks, on_chunk=lambda: None):
    session = SimpleNamespace(
        head=lambda url, **kwargs: SimpleNamespace(status_code=200, raise_for_status=lambda: None),
        get=lambda url, **kwargs: _Response(chunks, on_chunk),
    )
    metadata = MethodMetadata.config(method="Dummy", suite="dummy-suite", cache_root=tmp_path)
    downloader = MethodDownloaderPublicR2(metadata, base_url="https://example.org", verbose=False, session=session)
    target = tmp_path / "results" / "hpo_trajectories.parquet"
    downloader._download_to_local_if_exists(key="cache/hpo_trajectories.parquet", path_local=target)
    return target


def test_the_target_appears_only_once_complete(tmp_path):
    target = tmp_path / "results" / "hpo_trajectories.parquet"
    seen_mid_download = []
    _download(tmp_path, [b"ab", b"cd"], on_chunk=lambda: seen_mid_download.append(target.exists()))
    assert seen_mid_download == [False, False]
    assert target.read_bytes() == b"abcd"
    assert [p.name for p in target.parent.iterdir()] == [target.name]  # no temp file left behind


def test_a_failed_download_leaves_nothing(tmp_path):
    with pytest.raises(ConnectionError):
        _download(tmp_path, [b"ab", ConnectionError("dropped")])
    assert list((tmp_path / "results").iterdir()) == []
