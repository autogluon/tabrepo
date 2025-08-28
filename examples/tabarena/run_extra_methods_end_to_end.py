from __future__ import annotations

from pathlib import Path

from tabrepo.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults
from tabrepo.tabarena.website_format import format_leaderboard


def download_results(
    url_prefix,
    local_path_prefix,
    filename,
):
    url = f"{url_prefix}{filename}"
    local_dir_suffix = Path(filename).with_suffix("").as_posix()
    path_raw = local_path_prefix / local_dir_suffix
    download_and_extract_zip(url=url, path_local=path_raw)


def end_to_end_new_results(
    path_raw,
    methods: list[str | tuple[str, str]] | None = None,
    cache: bool = True,
):
    fig_output_dir = Path("tabarena_figs") / "new_results"
    """
    Run logic end-to-end and cache all results:
    1. load raw artifacts
        path_raw should be a directory containing `results.pkl` files for each run.
    2. infer method_metadata
    3. cache method_metadata
    4. cache raw artifacts
    5. infer task_metadata
    5. generate processsed
    6. cache processed
    7. generate results
    8. cache results

    Once this is executed once, it does not need to be ran again.
    """
    if cache:
        end_to_end = EndToEnd.from_path_raw(path_raw=path_raw)
        end_to_end_results = end_to_end.to_results()
    else:
        assert methods is not None
        end_to_end_results = EndToEndResults.from_cache(methods=methods)

    """
    Load cached results and compare on TabArena
    1. Generates figures and leaderboard using the TabArena methods and the user's method
    2. Missing values are imputed to default RandomForest.
    """
    leaderboard = end_to_end_results.compare_on_tabarena(output_dir=fig_output_dir)
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))


if __name__ == '__main__':
    url_prefix = "https://data.lennart-purucker.com/tabarena/"
    local_path_prefix = Path("local_data")
    cache = True

    filenames = [
        "data_BetaTabPFN.zip",
        "data_TabFlex.zip",
        "leaderboard_submissions/data_Mitra_14082025.zip",
    ]
    for filename in filenames:
        download_results(
            url_prefix=url_prefix,
            local_path_prefix=local_path_prefix,
            filename=filename,
        )

    methods = [
        "BetaTabPFN",
        "Mitra",
        "TabFlex",
        # "AutoGluon_ExperimentalV140_4h",
        # "LightGBM_aio_0808",
    ]

    end_to_end_new_results(
        path_raw=local_path_prefix,
        cache=cache,
        # methods=methods,
    )
