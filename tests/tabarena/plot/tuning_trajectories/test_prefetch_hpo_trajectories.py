"""The trajectory plots fetch each missing hosted trajectory once, before fanning out to the workers."""

from __future__ import annotations

from types import SimpleNamespace

from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import _prefetch_hpo_trajectories


def test_each_missing_trajectory_is_downloaded_once(tmp_path):
    downloads = []

    def method(name, *, method_type="config", has_hpo_trajectories=False, remote=True, cached=False):
        path = tmp_path / name / "hpo_trajectories.parquet"
        if cached:
            path.parent.mkdir(parents=True)
            path.touch()
        return SimpleNamespace(
            method=name,
            method_type=method_type,
            has_hpo_trajectories=has_hpo_trajectories,
            has_remote_cache=remote,
            path_results_hpo_trajectories=lambda: path,
            method_downloader=lambda: SimpleNamespace(download_results_hpo_trajectories=lambda: downloads.append(name)),
        )

    methods = [
        method("Missing"),
        method("Cached", cached=True),
        method("LocalOnly", remote=False),
        method("Baseline", method_type="baseline"),
        method("Portfolio", method_type="portfolio", has_hpo_trajectories=True),
        method("NotDisplayed"),
    ]
    _prefetch_hpo_trajectories(methods, methods_to_display=["Missing", "Cached", "LocalOnly", "Baseline", "Portfolio"])
    assert downloads == ["Missing", "Portfolio"]
