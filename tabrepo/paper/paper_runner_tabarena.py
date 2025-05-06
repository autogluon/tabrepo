from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabrepo import EvaluationRepository, Evaluator
from scripts.baseline_comparison.plot_utils import (
    figure_path,
)
from scripts.baseline_comparison.evaluate_utils import plot_family_proportion
from tabrepo.paper.paper_utils import make_scorers, generate_sensitivity_plots, get_framework_type_method_names
from scripts.dataset_analysis import generate_dataset_analysis
from .paper_runner import PaperRun


class PaperRunTabArena(PaperRun):
    def run(self) -> pd.DataFrame:
        df_results_baselines = self.run_baselines()
        df_results_configs = self.run_configs()
        df_results_hpo_all = self.run_hpo_by_family()
        n_portfolios = [200]
        df_results_n_portfolio = []
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=1, n_ensemble_in_name=False))

        df_results_extra = []
        # FIXME: Why does n_max_models_per_type="auto" make things so slow? -> 7 seconds to 107 seconds
        # FIXME: n_max_models_per_type doesn't work correctly atm, need to actually separate the types!
        # df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto"))
        df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto", fix_fillna=True))
        df_results_extra.append(self.run_zs(n_portfolios=50, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        df_results_extra.append(self.run_zs(n_portfolios=20, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=10, n_ensemble=None, n_ensemble_in_name=False))
        # df_results_extra.append(self.run_zs(n_portfolios=5, n_ensemble=None, n_ensemble_in_name=False))
        # df_results_extra.append(self.run_zs(n_portfolios=4, n_ensemble=None, n_ensemble_in_name=False))
        # df_results_extra.append(self.run_zs(n_portfolios=3, n_ensemble=None, n_ensemble_in_name=False))
        # df_results_extra.append(self.run_zs(n_portfolios=2, n_ensemble=None, n_ensemble_in_name=False))
        # df_results_extra.append(self.run_zs(n_portfolios=10, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=5, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=4, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=3, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=2, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))

        df_results_n_portfolio = pd.concat(df_results_n_portfolio + df_results_extra)

        df_results_all = self.evaluator.compare_metrics(results_df=df_results_n_portfolio, configs=[], baselines=[], keep_extra_columns=True)

        df_results_all = pd.concat([
            df_results_all,
            df_results_hpo_all,
            df_results_baselines,
            df_results_configs,
        ])

        df_results_all["seed"] = 0
        df_results_all = df_results_all.set_index("seed", append=True)
        df_results_all = df_results_all[~df_results_all.index.duplicated(keep='first')]
        print(df_results_all)
        # TODO: Verify df_results_all format (index names, etc.)
        return df_results_all

    def eval(self, df_results: pd.DataFrame, use_gmean: bool = False):
        framework_types = [
            "GBM",
            "XGB",
            "CAT",
            "NN_TORCH",
            "FASTAI",
            "KNN",
            "RF",
            "XT",
            "LR",
            "TABPFN",
            "REALMLP",
            "EBM",
            "FT_TRANSFORMER",
        ]

        # for model_key, model_name in [("XGB", "XGBoost"), ("CAT", "CatBoost"), ("GBM", "LightGBM"), ("RF", "RandomForest"),
        #                               ("XT", "ExtraTrees")]:
        #     extra = [f"{model_key}_OG", f"{model_key}_ALT"]
        #     framework_types += extra

        df_results = df_results.copy()
        df_results = df_results.reset_index()
        df_results = df_results.rename(columns={
            "framework": "method",
        })
        df_results["method"] = df_results["method"].map({
            "AutoGluon_bq_4h8c_2023_11_14": "AutoGluon 0.8 (4h)",
            "AutoGluon_bq_4h8c_2024_10_25": "AutoGluon 1.1 (4h)",
            "autosklearn2_4h8c_2023_11_14": "Autosklearn2 (4h)",
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
        }).fillna(df_results["method"])
        print(df_results)
        rank_scorer, normalized_scorer = make_scorers(self.repo)
        df_results["normalized-error"] = [normalized_scorer.rank(task=(dataset, fold), error=error) for (dataset, fold, error) in zip(df_results["dataset"], df_results["fold"], df_results["metric_error"])]
        df_results["seed"] = 0

        import copy
        df_results_rank_compare = copy.deepcopy(df_results)
        df_results_rank_compare = df_results_rank_compare.rename(columns={"method": "framework"})

        self.plot_tuning_impact(df=df_results, framework_types=framework_types, save_prefix="tmp/v2_mini", use_gmean=use_gmean)

        # df_results_realmlp_alt = df_results[df_results["method"].str.contains("RealMLP_r") & df_results["method"].str.contains("_alt_")]
        # df_results_realmlp_og = df_results[df_results["method"].str.contains("RealMLP_r") & ~df_results["method"].str.contains("_alt_")]
        #
        # df_results_only_og = df_results.drop(index=df_results_realmlp_alt.index)
        # df_results_only_alt = df_results.drop(index=df_results_realmlp_og.index)
        # self.plot_tuning_impact(df=df_results_only_alt, framework_types=framework_types, save_prefix="tmp/v2_mini_alt", use_gmean=use_gmean)
        #
        # self.plot_tuning_impact(df=df_results_only_og, framework_types=framework_types, save_prefix="tmp/v2_mini_main", use_gmean=use_gmean)

        df_results_rank_compare2 = df_results_rank_compare[~df_results_rank_compare["framework"].str.contains("_BAG_L1") & ~df_results_rank_compare["framework"].str.contains("_r")]
        self.evaluator.plot_overall_rank_comparison(results_df=df_results_rank_compare2, save_dir="tmp/paper_v2")

        hue_order_family_proportion = [
            "RealMLP",
            "CatBoost",
            "LightGBM",
            "XGBoost",
            "NeuralNetTorch",
            "RandomForest",
            "ExtraTrees",
            "LinearModel",
            "KNeighbors",
            # "TabPFNv2",
            "TabPFN",
            # "TabForestPFN",
            "ExplainableBM",
            "NeuralNetFastAI",
            "FTTransformer",
        ]

        plot_family_proportion(df=df_results, save_prefix="tmp/family_prop", method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)
        plot_family_proportion(df=df_results, save_prefix="tmp/family_prop2", method="Portfolio-fix_fillna-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)
