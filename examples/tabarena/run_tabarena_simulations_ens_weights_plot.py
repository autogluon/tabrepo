from __future__ import annotations

from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from nips2025_utils.load_final_paper_results import load_paper_results


if __name__ == '__main__':
    context_name = "tabarena_paper_full_51"
    eval_save_path = f"{context_name}/output"
    load_from_s3 = False  # Do this for first run, then make false for speed
    generate_from_repo = False
    with_holdout = False

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, df_results_holdout, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        generate_from_repo=generate_from_repo,
        load_from_s3=load_from_s3,
        # save_local_to_s3=True,
    )

    load_sim_cache = True

    context_name = "tabarena_paper_full_51"
    cache_path = f"./{context_name}/repo_cache/tabarena_all.pkl"

    df_result_save_path = f"./{context_name}/data/df_results_portfolio_ens_weights.parquet"
    eval_save_path = f"{context_name}/output"

    repo = EvaluationRepositoryCollection.load(path=cache_path)

    eval_save_path_tabpfn_datasets = f"{eval_save_path}/tabpfn_datasets"
    eval_save_path_tabicl_datasets = f"{eval_save_path}/tabicl_datasets"
    eval_save_path_full = f"{eval_save_path}/full"
    # assert len(repo.datasets()) == 51

    repo.set_config_fallback(config_fallback="RandomForest_c1_BAG_L1")
    paper_full = PaperRunTabArena(repo=repo, output_dir=eval_save_path_full, backend="ray")
    # paper_run.generate_data_analysis()

    if not load_sim_cache:
        df_results = paper_full.run_only_portfolio_200()
        save_pd.save(df=df_results, path=df_result_save_path)
    else:
        df_results = load_pd.load(path=df_result_save_path)

    method = "Portfolio-N200 (ensemble) (4h)"
    figsize = (24, 20)

    paper_full.get_weights_heatmap(
        df_results=df_results,
        method=method,
        figsize=figsize,
        excluded_families=[
            "DUMMY",
            "TABPFNV2",
            "TABICL",
        ],
    )

    paper_tabpfn_datasets = PaperRunTabArena(
        repo=repo,
        output_dir=eval_save_path_tabpfn_datasets,
        datasets=datasets_tabpfn,
    )
    paper_tabicl_datasets = PaperRunTabArena(
        repo=repo,
        output_dir=eval_save_path_tabicl_datasets,
        datasets=datasets_tabicl,
    )

    paper_tabpfn_datasets.get_weights_heatmap(
        df_results=df_results,
        method=method,
        figsize=figsize,
        excluded_families=[
            "DUMMY",
            "TABICL",
        ],
    )

    paper_tabicl_datasets.get_weights_heatmap(
        df_results=df_results,
        method=method,
        figsize=figsize,
        excluded_families=[
            "DUMMY",
            "TABPFNV2",
        ],
    )
