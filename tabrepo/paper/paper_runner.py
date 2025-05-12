from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabrepo import EvaluationRepository, Evaluator
from scripts.baseline_comparison.evaluate_utils import plot_family_proportion
from tabrepo.paper.paper_utils import make_scorers, generate_sensitivity_plots, get_framework_type_method_names
from scripts.dataset_analysis import generate_dataset_analysis


class PaperRun:
    def __init__(self, repo: EvaluationRepository, output_dir: str = None):
        self.repo = repo
        self.evaluator = Evaluator(repo=self.repo)
        self.output_dir = output_dir

    def get_config_type_groups(self, ban_families=True) -> dict:
        config_type_groups = {}
        configs_type = self.repo.configs_type()
        all_configs = self.repo.configs()
        for c in all_configs:
            if configs_type[c] not in config_type_groups:
                config_type_groups[configs_type[c]] = []
            config_type_groups[configs_type[c]].append(c)

        if ban_families:
            # FIXME: TMP HACK
            banned_families = [
                "DUMMY",
                "TABICL",
                "TABDPT",
                # "TABPFNV2",
                "FT_TRANSFORMER",
            ]
            for b in banned_families:
                if b in config_type_groups:
                    config_type_groups.pop(b)
        return config_type_groups

    def run_hpo_by_family(self, include_uncapped: bool = False) -> pd.DataFrame:
        config_type_groups = self.get_config_type_groups(ban_families=True)

        hpo_results_lst = []
        time_limit = 3600 * 4
        for family in config_type_groups:
            configs_family = config_type_groups[family]
            df_results_family_hpo_ens, _ = self.repo.evaluate_ensembles(
                configs=configs_family, fit_order="random", seed=0, ensemble_size=40, time_limit=time_limit
            )
            df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble) (4h)"
            df_results_family_hpo, _ = self.repo.evaluate_ensembles(
                configs=configs_family, fit_order="random", seed=0, ensemble_size=1, time_limit=time_limit
            )
            df_results_family_hpo["framework"] = f"{family} (tuned) (4h)"
            hpo_results_lst.append(df_results_family_hpo.reset_index())
            hpo_results_lst.append(df_results_family_hpo_ens.reset_index())

        if include_uncapped:
            for family in config_type_groups:
                configs_family = config_type_groups[family]
                df_results_family_hpo_ens, _ = self.repo.evaluate_ensembles(
                    configs=configs_family, fit_order="original", seed=0, ensemble_size=40, time_limit=None
                )
                df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble)"
                df_results_family_hpo, _ = self.repo.evaluate_ensembles(
                    configs=configs_family, fit_order="original", seed=0, ensemble_size=1, time_limit=None
                )
                df_results_family_hpo["framework"] = f"{family} (tuned)"
                hpo_results_lst.append(df_results_family_hpo.reset_index())
                hpo_results_lst.append(df_results_family_hpo_ens.reset_index())

        df_results_hpo_all = pd.concat(hpo_results_lst, ignore_index=True)
        df_results_hpo_all = df_results_hpo_all.set_index(["dataset", "fold", "framework"])
        # df_results_hpo_all = self.evaluator.compare_metrics(results_df=df_results_hpo_all, configs=[], baselines=[])
        return df_results_hpo_all

    def run_zs(
        self,
        n_portfolios: int = 200,
        n_ensemble: int = None,
        n_ensemble_in_name: bool = True,
        n_max_models_per_type: int | str | None = None,
        fix_fillna: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        df_zeroshot_portfolio = self.evaluator.zeroshot_portfolio(
            n_portfolios=n_portfolios,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=n_ensemble_in_name,
            n_max_models_per_type=n_max_models_per_type,
            fix_fillna=fix_fillna,
            **kwargs,
        )
        # df_zeroshot_portfolio = self.evaluator.compare_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    def run_zs_single_best(self) -> pd.DataFrame:
        df_zeroshot_portfolio = self.evaluator.zeroshot_portfolio(
            n_portfolios=200,
            n_ensemble=1,
            n_ensemble_in_name=False,
        )
        # df_zeroshot_portfolio = self.evaluator.compare_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    def run_baselines(self) -> pd.DataFrame:
        df_results_baselines = self.evaluator.compare_metrics(configs=[])
        return df_results_baselines

    def run_configs(self) -> pd.DataFrame:
        df_results_configs = self.evaluator.compare_metrics(baselines=[], include_metric_error_val=True)
        return df_results_configs

    def run(self) -> pd.DataFrame:
        df_results_hpo_all = self.run_hpo_by_family()
        # df_zeroshot_portfolio = self.run_zs()
        df_zeroshot_portfolio_single_best = self.run_zs_single_best()
        df_results_baselines = self.run_baselines()
        df_results_configs = self.run_configs()
        n_portfolios = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        n_ensembles = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        df_results_n_portfolio = []
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=1, n_ensemble_in_name=False))
        df_results_n_portfolio = pd.concat(df_results_n_portfolio)
        df_results_n_ensemble = []
        for n_ensemble in n_ensembles:
            df_results_n_ensemble.append(self.run_zs(n_portfolios=200, n_ensemble=n_ensemble, n_ensemble_in_name=True))
        df_results_n_ensemble = pd.concat(df_results_n_ensemble)

        df_results_all = pd.concat([
            df_results_hpo_all,
            # df_zeroshot_portfolio,
            df_zeroshot_portfolio_single_best,
            df_results_n_portfolio,
            df_results_n_ensemble,
        ])

        df_results_all = self.evaluator.compare_metrics(results_df=df_results_all, configs=[], baselines=[])

        df_results_all = pd.concat([
            df_results_all,
            df_results_baselines,
            df_results_configs,
        ])

        df_results_all["seed"] = 0
        df_results_all = df_results_all.set_index("seed", append=True)
        # df_results_all = df_results_all.drop_duplicates(subset=["framework", "dataset", "fold", "seed"])
        df_results_all = df_results_all[~df_results_all.index.duplicated(keep='first')]
        print(df_results_all)
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
        ]
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
        }).fillna(df_results["method"])
        print(df_results)
        rank_scorer, normalized_scorer = make_scorers(self.repo)
        df_results["normalized-error"] = [normalized_scorer.rank(task=(dataset, fold), error=error) for (dataset, fold, error) in zip(df_results["dataset"], df_results["fold"], df_results["metric_error"])]
        df_results["seed"] = 0
        # normalized_error = normalized_scorer.rank(task, metric_error)
        n_portfolios = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        n_ensembles = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        generate_sensitivity_plots(df=df_results, n_portfolios=n_portfolios, n_ensemble_iterations=n_ensembles, save_prefix="tmp/sens")

        baselines = ["Autosklearn2 (4h)", "AutoGluon 0.8 (4h)", "AutoGluon 1.1 (4h)", ]
        baseline_colors = ["darkgray", "black", "blue"]
        self.plot_tuning_impact(df=df_results, framework_types=framework_types, save_prefix="tmp/v2", use_gmean=use_gmean, baselines=baselines, baseline_colors=baseline_colors)

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
            "TabPFNv2",
            "TabPFN",
            "TabForestPFN",
            "EBM",
            "NeuralNetFastAI",
        ]

        plot_family_proportion(df=df_results, save_prefix="tmp/family_prop", method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)

        # self.evaluator.plot_overall_rank_comparison(results_df=df_results, save_dir="tmp/paper_v2")

    # FIXME: Avoid hardcoding
    def plot_tuning_impact(
        self,
        df: pd.DataFrame,
        framework_types: list,
        save_prefix: str,
        baselines: list[str] = None,
        baseline_colors: list[str] = None,
        show: bool = True,
        use_gmean=False,
    ):
        df = df.copy()

        ylim = [0, 1]
        metric = "normalized-error"
        lower_is_better = True

        f_map, f_map_type, f_map_inverse = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600*4, "_4h"),
                (None, None),
            ]
        )

        df["framework_type"] = df["method"].map(f_map_type).fillna(df["method"])
        df["tune_method"] = df["method"].map(f_map_inverse).fillna("default")

        if baselines is None:
            baselines = []
        if baseline_colors is not None:
            assert len(baselines) == len(baseline_colors), f"A color must be specified for each baseline via the `baseline_colors` argument."

        framework_types = baselines + framework_types

        df_plot = df[df["framework_type"].isin(framework_types)]

        # df_plot_w_mean_2 = df_plot.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()

        df_plot_w_mean_per_dataset = df_plot.groupby(["framework_type", "tune_method", "dataset"])[metric].mean().reset_index()

        if use_gmean:
            # FIXME: Doesn't plot correctly, need to figure out error bars for geometric mean
            df_plot_eps = df_plot.copy(deep=True)
            df_plot_eps[metric] += 0.01
            from scipy.stats import gmean
            df_plot_w_gmean_per_dataset = df_plot.groupby(["framework_type", "tune_method", "dataset"])[metric].apply(gmean).reset_index()
            df_plot_w_mean_per_dataset = df_plot_w_gmean_per_dataset

        df_plot_w_mean_2 = df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()

        baseline_means = {}
        for baseline in baselines:
            baseline_means[baseline] = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == baseline][metric].iloc[0]

        df_plot_w_mean_2 = df_plot_w_mean_2[~df_plot_w_mean_2["framework_type"].isin(baselines)]
        df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=True)

        df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")

        framework_type_order = list(df_plot_mean_dedupe["framework_type"].to_list())
        framework_type_order.reverse()

        # sns.set_color_codes("pastel")
        with sns.axes_style("whitegrid"):
            colors = sns.color_palette("pastel").as_hex()
            errcolors = sns.color_palette("deep").as_hex()
            fig, ax = plt.subplots(1, 1, figsize=(24, 6))
            for baseline, color in zip(baselines, baseline_colors):
                baseline_mean = baseline_means[baseline]
                ax.axhline(y=baseline_mean, label=baseline, color=color, linewidth=2.0, ls="--")

            sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Default",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"], ax=ax,
                order=framework_type_order, color=colors[0],
                width=0.6,
                err_kws={"color": errcolors[0]},
                alpha=1.0,
            )
            sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Best",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "best"], ax=ax,
                order=framework_type_order, color=colors[3],
                width=0.55,
                err_kws={"color": errcolors[3]},
                alpha=1.0,
            )
            sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Tuned (4h)",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_4h"], ax=ax,
                order=framework_type_order,
                color=colors[1],
                width=0.5,
                err_kws={"color": errcolors[1]},
            )
            sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Tuned",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"], ax=ax,
                order=framework_type_order,
                color=colors[4],
                width=0.45,
                err_kws={"color": errcolors[4]},
            )
            sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Tuned + Ensembled (4h)",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled_4h"], ax=ax,
                order=framework_type_order, color=colors[2],
                width=0.4,
                err_kws={"color": errcolors[2]},
            )
            boxplot = sns.barplot(
                x="framework_type", y=metric,
                # hue="tune_method",  # palette=["m", "g", "r],
                label="Tuned + Ensembled",
                data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"], ax=ax,
                order=framework_type_order, color=colors[5],
                width=0.35,
                err_kws={"color": errcolors[5]},
            )
            boxplot.set(xlabel=None)  # remove "Method" in the x-axis
            #boxplot.set_title("Effect of tuning and ensembling")
            if ylim is not None:
                ax.set_ylim(ylim)


            #ax.legend(loc="upper center", ncol=5)
            ax.legend(loc="upper center", ncol=5, bbox_to_anchor=[0.5, 1.2])

            # reordering the labels
            handles, labels = ax.get_legend_handles_labels()

            # specify order
            len_baselines = len(baselines)
            num_other = len(labels) - len_baselines
            order = [n + len_baselines for n in range(num_other)] + [n for n in range(len_baselines)]
            # order = [3, 4, 5, 0, 1, 2]

            # pass handle & labels lists along with order as below
            ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=5, bbox_to_anchor=[0.5, 1.2])

            #ax.legend(bbox_to_anchor=[0.1, 0.5], loc='center left', ncol=5)
            plt.tight_layout()

            if save_prefix:
                fig_path = Path(save_prefix)
                fig_path.mkdir(parents=True, exist_ok=True)
                if use_gmean:
                    fig_name = "tuning-impact-gmean.png"
                else:
                    fig_name = "tuning-impact.png"
                fig_save_path = fig_path / fig_name
                plt.savefig(fig_save_path)
            if show:
                plt.show()

    def generate_data_analysis(self, expname_outdir: str):
        generate_dataset_analysis(repo=self.repo, expname_outdir=expname_outdir)


class PaperRunMini(PaperRun):
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
        df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto"))
        df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto", fix_fillna=True))

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
            "TABPFNV2",
            "TABICL",
            "REALMLP",
            "EBM",
        ]
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
            "TabPFNv2_c1_BAG_L1": "TABPFNV2 (default)",
            "TabICL_c1_BAG_L1": "TABICL (default)",
            "RealMLP_c1_BAG_L1": "REALMLP (default)",
            "ExplainableBM_c1_BAG_L1": "EBM (default)",
        }).fillna(df_results["method"])
        print(df_results)
        rank_scorer, normalized_scorer = make_scorers(self.repo)
        df_results["normalized-error"] = [normalized_scorer.rank(task=(dataset, fold), error=error) for (dataset, fold, error) in zip(df_results["dataset"], df_results["fold"], df_results["metric_error"])]
        df_results["seed"] = 0

        import copy
        df_results_rank_compare = copy.deepcopy(df_results)
        df_results_rank_compare = df_results_rank_compare.rename(columns={"method": "framework"})

        self.plot_tuning_impact(df=df_results, framework_types=framework_types, save_prefix=f"{self.output_dir}/tmp/v2_mini", use_gmean=use_gmean)

        # df_results_realmlp_alt = df_results[df_results["method"].str.contains("RealMLP_r") & df_results["method"].str.contains("_alt_")]
        # df_results_realmlp_og = df_results[df_results["method"].str.contains("RealMLP_r") & ~df_results["method"].str.contains("_alt_")]
        #
        # df_results_only_og = df_results.drop(index=df_results_realmlp_alt.index)
        # df_results_only_alt = df_results.drop(index=df_results_realmlp_og.index)
        # self.plot_tuning_impact(df=df_results_only_alt, framework_types=framework_types, save_prefix="tmp/v2_mini_alt", use_gmean=use_gmean)
        #
        # self.plot_tuning_impact(df=df_results_only_og, framework_types=framework_types, save_prefix="tmp/v2_mini_main", use_gmean=use_gmean)

        df_results_rank_compare2 = df_results_rank_compare[~df_results_rank_compare["framework"].str.contains("_BAG_L1") & ~df_results_rank_compare["framework"].str.contains("_r")]
        self.evaluator.plot_overall_rank_comparison(results_df=df_results_rank_compare2, save_dir=f"{self.output_dir}/tmp/paper_v2")

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

        plot_family_proportion(df=df_results, save_prefix=f"{self.output_dir}/tmp/family_prop", method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)
        plot_family_proportion(df=df_results, save_prefix=f"{self.output_dir}/tmp/family_prop2", method="Portfolio-fix_fillna-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)
