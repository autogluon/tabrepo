from tabrepo.nips2025_utils.load_final_paper_results import load_paper_results
from autogluon.common.savers import save_pd


banned_methods = [
    # "Portfolio-N200 (ensemble) (4h)",
    "Portfolio-N100 (ensemble) (4h)",
    "Portfolio-N50 (ensemble) (4h)",
    "Portfolio-N20 (ensemble) (4h)",
    "Portfolio-N10 (ensemble) (4h)",
    "Portfolio-N5 (ensemble) (4h)",
    "Portfolio-N200 (4h)",
    "Portfolio-N100 (4h)",
    "Portfolio-N50 (4h)",
    "Portfolio-N20 (4h)",
    "Portfolio-N10 (4h)",
    "Portfolio-N5 (4h)",

    "AutoGluon_bq_1h8c",
    "AutoGluon_bq_5m8c",

    # "Portfolio-N200 (ensemble, holdout) (4h)",
    "Portfolio-N100 (ensemble, holdout) (4h)",
    "Portfolio-N50 (ensemble, holdout) (4h)",
    "Portfolio-N20 (ensemble, holdout) (4h)",
    "Portfolio-N10 (ensemble, holdout) (4h)",
    "Portfolio-N5 (ensemble, holdout) (4h)",
]


def rename_func(name):
    if "(tuned)" in name:
        name = name.rsplit("(tuned)", 1)[0]
        name = f"{name}(tuned, holdout)"
    elif "(tuned + ensemble)" in name:
        name = name.rsplit("(tuned + ensemble)", 1)[0]
        name = f"{name}(tuned + ensemble, holdout)"
    elif "(ensemble) (4h)" in name:
        name = name.rsplit("(ensemble) (4h)", 1)[0]
        name = f"{name}(ensemble, holdout) (4h)"
    return name


# TODO: Also clean holdout results
def clean_results():
    context_name = "tabarena_paper_full_51"
    load_from_s3 = True

    print(f"Loading results... context_name={context_name}, load_from_s3={load_from_s3}")
    df_results, df_results_holdout, datasets_tabpfn, datasets_tabicl = load_paper_results(
        context_name=context_name,
        load_from_s3=load_from_s3,
    )
    df_results = df_results[~df_results["framework"].isin(banned_methods)]

    default_map = {
        "LightGBM_c1_BAG_L1": "GBM (default)",
        "XGBoost_c1_BAG_L1": "XGB (default)",
        "CatBoost_c1_BAG_L1": "CAT (default)",
        "NeuralNetTorch_c1_BAG_L1": "NN_TORCH (default)",
        "NeuralNetFastAI_c1_BAG_L1": "FASTAI (default)",
        "KNeighbors_c1_BAG_L1": "KNN (default)",
        "RandomForest_c1_BAG_L1": "RF (default)",
        "ExtraTrees_c1_BAG_L1": "XT (default)",
        "LinearModel_c1_BAG_L1": "LR (default)",
        "TabPFN_c1_BAG_L1": "TABPFN (default)",
        "RealMLP_c1_BAG_L1": "REALMLP (default)",
        "ExplainableBM_c1_BAG_L1": "EBM (default)",
        "FTTransformer_c1_BAG_L1": "FT_TRANSFORMER (default)",
        "TabPFNv2_c1_BAG_L1": "TABPFNV2 (default)",
        "TabICL_c1_BAG_L1": "TABICL (default)",
        "TabDPT_c1_BAG_L1": "TABDPT (default)",
        "TabM_c1_BAG_L1": "TABM (default)",
        "ModernNCA_c1_BAG_L1": "MNCA (default)",

        "RandomForest_r1_BAG_L1_HOLDOUT": "RF (holdout)",
        "ExtraTrees_r1_BAG_L1_HOLDOUT": "XT (holdout)",
        "LinearModel_c1_BAG_L1_HOLDOUT": "LR (holdout)",

        "LightGBM_c1_BAG_L1_HOLDOUT": "GBM (holdout)",
        "XGBoost_c1_BAG_L1_HOLDOUT": "XGB (holdout)",
        "CatBoost_c1_BAG_L1_HOLDOUT": "CAT (holdout)",
        "NeuralNetTorch_c1_BAG_L1_HOLDOUT": "NN_TORCH (holdout)",
        "NeuralNetFastAI_c1_BAG_L1_HOLDOUT": "FASTAI (holdout)",

        "RealMLP_c1_BAG_L1_HOLDOUT": "REALMLP (holdout)",
        "ExplainableBM_c1_BAG_L1_HOLDOUT": "EBM (holdout)",
        "FTTransformer_c1_BAG_L1_HOLDOUT": "FT_TRANSFORMER (holdout)",
        # "TabPFNv2_c1_BAG_L1_HOLDOUT": "TABPFNV2 (holdout)",
        # "TabICL_c1_BAG_L1_HOLDOUT": "TABICL (holdout)",
        # "TabDPT_c1_BAG_L1_HOLDOUT": "TABDPT (holdout)",
        "TabM_c1_BAG_L1_HOLDOUT": "TABM (holdout)",
        "ModernNCA_c1_BAG_L1_HOLDOUT": "MNCA (holdout)",
    }

    default_map_v2 = {k: "default" for k in default_map.keys()}

    df_results["method_subtype"] = df_results["framework"].map(default_map_v2).fillna(df_results["method_subtype"])

    df_results["framework"] = df_results["framework"].map({
        "AutoGluon_bq_4h8c": "AutoGluon 1.3 (4h)",
        "AutoGluon_bq_1h8c": "AutoGluon 1.3 (1h)",
        "AutoGluon_bq_5m8c": "AutoGluon 1.3 (5m)",
        **default_map,
    }).fillna(df_results["framework"])

    df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"])
    df_results = df_results.rename(columns={"framework": "method"})
    df_results["seed"] = df_results["seed"].fillna(0)
    df_results["seed"] = df_results["seed"].astype(int)
    df_results = df_results.sort_values(by=["dataset", "fold", "method"])
    df_results = df_results.reset_index(drop=True)

    df_results_configs = df_results[(df_results["method_type"] == "config") & (df_results["method_subtype"] != "default")]
    df_results_single_best = df_results[(df_results["method_type"] == "portfolio") & (df_results["method_subtype"] == "best")]
    idx_minimal = df_results.index
    idx_minimal = idx_minimal[~idx_minimal.isin(df_results_configs.index)]
    idx_minimal = idx_minimal[~idx_minimal.isin(df_results_single_best.index)]
    df_results_minimal = df_results.loc[idx_minimal]

    df_map = dict(
        full=df_results,
        leaderboard=df_results_minimal,
        single_best=df_results_single_best,
        configs=df_results_configs,
    )

    return df_map


if __name__ == '__main__':
    # cleans the results from the TabArena paper so it is easy to use downstream
    df_map = clean_results()
    df_map_lite = {k: v[v["fold"] == 0] for k, v in df_map.items()}

    # path_prefix = "s3://tabarena/results/"
    path_prefix = "data/"
    for k, df in df_map.items():
        save_pd.save(path=f"{path_prefix}df_results_{k}.parquet", df=df)

    for k, df in df_map_lite.items():
        save_pd.save(path=f"{path_prefix}df_results_lite_{k}.parquet", df=df)
