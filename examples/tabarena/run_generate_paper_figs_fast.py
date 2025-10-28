from pathlib import Path

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.tabarena.website_format import format_leaderboard
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata import (
    tabarena_method_metadata_2025_06_12_collection_main,
)


"""
This is a temporary script to drive development of figures for NeurIPS 2025 camera ready.

TODO: Figures to refine:
    - winrate_matrix.pdf
        Need to make it fit in the paper nicely
        Perhaps remove poor performing methods?
        Limit y axis length?
        Limit legend length to heatmap y-axis length
    - pareto_front_*_vs_*.pdf
        Pareto Front plotted beneath methods?
        y axis lower = 5%?
        Fix arrow + "Optimal" text to be aligned
        Make small to fit in paper figure space
            Make *_vs_time_infer and *_vs_time_train be one figure with two subplots and a shared legend 
            Move legend to bottom
        Stretch: Incorporate "imputed" information.
        
TODO: Figures to create:
    - improvability_test_vs_val.pdf
        TODO. If we make it elo, we could do `val elo - test elo` for overfitting severity
    - improvability_vs_n_configs.pdf
        Shows the improvement as n_configs increases.
        TODO. Data for this is not yet shared.
"""


if __name__ == '__main__':
    save_path = "output_leaderboard"  # folder to save all figures and tables
    output_path = Path(save_path) / "camera_ready"

    methods = [m for m in tabarena_method_metadata_2025_06_12_collection_main.method_metadata_lst if m.method_type != "portfolio"]

    # Hack to avoid plotting these methods so y-axis is nicer.
    methods = [m for m in methods if m.method not in ["LinearModel", "KNeighbors"]]

    # Only methods in the paper
    tabarena_context = TabArenaContext(
        methods=methods,
        include_ag_140=False,
        include_mitra=False,
    )

    leaderboard = tabarena_context.compare(output_dir=output_path)

    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)

    print(f"Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print("")

    # ==========================================================================================
    # TODO: Can use this to plot elo/improvability on test vs val, showcase overfitting
    leaderboard_val = tabarena_context.compare(output_dir=output_path / "score_on_val", score_on_val=True)

    leaderboard = leaderboard.set_index("method", drop=True)
    leaderboard_val = leaderboard_val.set_index("method", drop=True)
    leaderboard["elo_val"] = leaderboard_val["elo"]
    leaderboard["improvability_val"] = leaderboard_val["improvability"]

    # drop AG since we don't have val info
    leaderboard = leaderboard[~leaderboard["elo_val"].isna()]
    # ==========================================================================================
