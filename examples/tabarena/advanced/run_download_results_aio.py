from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_downloader import MethodDownloaderS3
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.tabarena.website_format import format_leaderboard


if __name__ == '__main__':
    bucket = "tabarena"
    s3_prefix_root = "cache_aio"

    methods = [
        "LightGBM_aio_0808",
        "LightGBM_aio_0812",
    ]

    for method in methods:
        metadata = MethodMetadata.from_s3_cache(
            method=method,
            bucket=bucket,
            s3_prefix_root=s3_prefix_root,
        )
        downloader = MethodDownloaderS3(
            method_metadata=metadata,
            bucket=bucket,
            s3_prefix_root=s3_prefix_root,
        )
        downloader.download_all()
        # Nick: Note that LightGBM_aio_0812 will skip processed and results due to invalid raw contents.

    # Compute the leaderboard
    from tabrepo.nips2025_utils.end_to_end import EndToEndResults
    e2e = EndToEndResults.from_cache(methods=["LightGBM_aio_0808"])
    leaderboard = e2e.compare_on_tabarena(output_dir="aio_figs", only_valid_tasks=True)
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
