from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from matplotlib import ticker
from matplotlib.container import BarContainer
from tueplots import bundles, fonts, fontsizes, figsizes

from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata

matplotlib.rcParams.update(fontsizes.neurips2024())

matplotlib.rcParams.update({
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
})

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from tabrepo import EvaluationRepository, Evaluator
from scripts.baseline_comparison.evaluate_utils import plot_family_proportion
from tabrepo.paper.paper_utils import make_scorers, generate_sensitivity_plots, get_framework_type_method_names
from scripts.dataset_analysis import generate_dataset_analysis

import matplotlib.colors as mcolors

def darken_color(color_str, amount=0.5):
    # Convert color string to RGB tuple (values between 0 and 1)
    rgb = mcolors.to_rgb(color_str)
    # Interpolate with black (0, 0, 0)
    darker_rgb = tuple((1 - amount) * c for c in rgb)
    return darker_rgb

class PaperRun:
    def __init__(self, repo: EvaluationRepository, output_dir: str = None, backend: Literal["ray", "native"] = "ray"):
        self.repo = repo
        self.evaluator = Evaluator(repo=self.repo)
        self.output_dir = output_dir
        self.backend = backend
        assert self.backend in ["ray", "native"]
        if self.backend == "ray":
            self.engine = "ray"
        else:
            self.engine = "sequential"

    def get_config_type_groups(self) -> dict:
        config_type_groups = {}
        configs_type = self.repo.configs_type()
        all_configs = self.repo.configs()
        for c in all_configs:
            if configs_type[c] not in config_type_groups:
                config_type_groups[configs_type[c]] = []
            config_type_groups[configs_type[c]].append(c)

        return config_type_groups

    def run_hpo_by_family(self, include_uncapped: bool = False, include_4h: bool = True, model_types: list[str] | None = None) -> pd.DataFrame:
        config_type_groups = self.get_config_type_groups()

        hpo_results_lst = []

        # for model_key, model_name in [("REALMLP", "RealMLP"), ("XGB", "XGBoost"), ("CAT", "CatBoost"), ("GBM", "LightGBM"), ("RF", "RandomForest"), ("XT", "ExtraTrees")]:
        #     realmlp_og = [c for c in config_type_groups[model_key] if c == f"{model_name}_c1_BAG_L1" or "_alt_" not in c]
        #     realmlp_alt = [c for c in config_type_groups[model_key] if c == f"{model_name}_c1_BAG_L1" or "_alt_" in c]
        #     config_type_groups[f"{model_key}_OG"] = realmlp_og
        #     config_type_groups[f"{model_key}_ALT"] = realmlp_alt

        if model_types is None:
            model_types = list(config_type_groups.keys())
        for family in model_types:
            assert family in config_type_groups, f"Model family {family} missing from available families: {list(config_type_groups.keys())}"

        if include_4h:
            time_limit = 3600 * 4
            # FIXME: do multiple random seeds and average
            for family in model_types:
                df_results_family_hpo_ens = self.run_ensemble_config_type(
                    config_type=family, fit_order="random", seed=0, n_iterations=40, time_limit=time_limit,
                )
                df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble) (4h)"
                df_results_family_hpo = self.run_ensemble_config_type(
                    config_type=family, fit_order="random", seed=0, n_iterations=1, time_limit=time_limit,
                )
                df_results_family_hpo["framework"] = f"{family} (tuned) (4h)"
                hpo_results_lst.append(df_results_family_hpo)
                hpo_results_lst.append(df_results_family_hpo_ens)

        if include_uncapped:
            for family in model_types:
                df_results_family_hpo_ens = self.run_ensemble_config_type(
                    config_type=family, fit_order="original", seed=0, n_iterations=40, time_limit=None,
                )
                df_results_family_hpo_ens["framework"] = f"{family} (tuned + ensemble)"

                df_results_family_hpo = self.run_ensemble_config_type(
                    config_type=family, fit_order="original", seed=0, n_iterations=1, time_limit=None,
                )
                df_results_family_hpo["framework"] = f"{family} (tuned)"
                hpo_results_lst.append(df_results_family_hpo)
                hpo_results_lst.append(df_results_family_hpo_ens)

        df_results_hpo_all = pd.concat(hpo_results_lst, ignore_index=True)
        return df_results_hpo_all

    def run_ensemble_config_type(
        self,
        config_type: str,
        n_iterations: int,
        n_configs: int = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
    ) -> pd.DataFrame:
        # FIXME: Don't recompute this each call, implement `self.repo.configs(config_types=[config_type])`
        config_type_groups = self.get_config_type_groups()
        configs = config_type_groups[config_type]
        if n_configs is not None:
            configs = configs[:n_configs]
        df_results_family_hpo, _ = self.repo.evaluate_ensembles(
            configs=configs,
            fit_order=fit_order,
            ensemble_size=n_iterations,
            seed=seed,
            time_limit=time_limit,
            backend=self.backend,
        )
        df_results_family_hpo = df_results_family_hpo.reset_index()
        df_results_family_hpo["method_type"] = "hpo"

        if n_iterations == 1:
            method_subtype = "tuned"
        else:
            method_subtype = "tuned_ensemble"
        df_results_family_hpo["method_subtype"] = method_subtype
        df_results_family_hpo["config_type"] = config_type

        method_metadata = dict(
            n_iterations=n_iterations,
            n_configs=n_configs,
            time_limit=time_limit,
            config_type=config_type,
            fit_order=fit_order,
        )

        df_results_family_hpo["method_metadata"] = [method_metadata] * len(df_results_family_hpo)

        return df_results_family_hpo

    def run_zs(
            self,
            n_portfolios: int = 200,
            n_ensemble: int = None,
            n_ensemble_in_name: bool = True,
            n_max_models_per_type: int | str | None = None,
            time_limit: float | None = 14400,
            fix_fillna: bool = True,
            **kwargs,
    ) -> pd.DataFrame:
        df_zeroshot_portfolio = self.evaluator.zeroshot_portfolio(
            n_portfolios=n_portfolios,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=n_ensemble_in_name,
            n_max_models_per_type=n_max_models_per_type,
            time_limit=time_limit,
            fix_fillna=fix_fillna,
            engine=self.engine,
            **kwargs,
        )
        df_zeroshot_portfolio["method_type"] = "portfolio"
        # df_zeroshot_portfolio = self.evaluator.compare_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    def run_zs_single_best(self) -> pd.DataFrame:
        df_zeroshot_portfolio = self.evaluator.zeroshot_portfolio(
            n_portfolios=200,
            n_ensemble=1,
            n_ensemble_in_name=False,
            engine=self.engine,
        )
        # df_zeroshot_portfolio = self.evaluator.compare_metrics(results_df=df_zeroshot_portfolio, configs=[], baselines=[])
        return df_zeroshot_portfolio

    # FIXME: TODO
    def run_zs_sim_advanced(self) -> pd.DataFrame:
        from tabrepo.repository import EvaluationRepositoryZeroshot
        adv = EvaluationRepositoryZeroshot.simulate_zeroshot(self.evaluator.repo)
        print(adv)
        return adv

    def run_baselines(self) -> pd.DataFrame | None:
        if not self.repo.baselines():
            return None
        df_results_baselines = self.evaluator.compare_metrics(configs=[]).reset_index()
        df_results_baselines["method_type"] = "baseline"
        return df_results_baselines

    def run_configs(self, model_types: list[str | None] | None = None) -> pd.DataFrame | None:
        configs = self.repo.configs(config_types=model_types)
        if not configs:
            return None
        df_results_configs = self.evaluator.compare_metrics(configs=configs, baselines=[], include_metric_error_val=True).reset_index()
        df_results_configs["method_type"] = "config"
        configs_types = self.repo.configs_type()
        df_results_configs["config_type"] = df_results_configs["framework"].map(configs_types)
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
            df_results_n_portfolio.append(
                self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
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
        df_results["normalized-error"] = [normalized_scorer.rank(task=(dataset, fold), error=error) for
                                          (dataset, fold, error) in
                                          zip(df_results["dataset"], df_results["fold"], df_results["metric_error"])]
        df_results["seed"] = 0
        # normalized_error = normalized_scorer.rank(task, metric_error)
        n_portfolios = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        n_ensembles = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200]
        generate_sensitivity_plots(df=df_results, n_portfolios=n_portfolios, n_ensemble_iterations=n_ensembles,
                                   save_prefix="tmp/sens")

        baselines = ["Autosklearn2 (4h)", "AutoGluon 0.8 (4h)", "AutoGluon 1.1 (4h)", ]
        baseline_colors = ["darkgray", "black", "blue"]
        self.plot_tuning_impact(df=df_results, framework_types=framework_types, save_prefix="tmp/v2",
                                use_gmean=use_gmean, baselines=baselines, baseline_colors=baseline_colors)

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

        plot_family_proportion(df=df_results, save_prefix="tmp/family_prop", method="Portfolio-N200 (ensemble) (4h)",
                               hue_order=hue_order_family_proportion)

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
        use_score: bool = True,
        df_elo: pd.DataFrame = None,
        name_suffix: str | None = None,
        imputed_names: list[str] | None = None,
        use_y: bool = False,
        metric: str = "normalized-error",
        plot_tune_types: list[str] | None = None,
    ):
        same_width = use_y
        use_lim = True
        use_elo = df_elo is not None
        lower_is_better = True
        lim = None
        xlim = None
        ylim = None
        use_latex = False

        if imputed_names is None:
            imputed_names = []
        # imputed_names = imputed_names or ['TabPFNv2', 'TabICL']

        df = df.copy(deep=True)

        framework_col = "framework_type"
        # framework_col = "framework_name"

        groupby_columns_extra = ["dataset"]

        if use_elo:
            metric = "elo"
            use_lim = True
            lim = [500, None]
            lower_is_better = False
            df = df_elo.copy(deep=True)
            df = df[["method", "elo", "elo+", "elo-"]]
            groupby_columns_extra = []
        elif use_score:
            lower_is_better = False
            df["normalized-score"] = 1 - df[metric]
            # df_plot_w_mean_per_dataset["normalized-score"] = 1 - df_plot_w_mean_per_dataset["normalized-error"]
            metric = "normalized-score"
        else:
            metric = metric

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        df["framework_type"] = df["method"].map(f_map_type).fillna(df["method"])
        df["tune_method"] = df["method"].map(f_map_inverse).fillna("default")

        if baselines is None:
            baselines = []
        if baseline_colors is not None:
            assert len(baselines) == len(
                baseline_colors), f"A color must be specified for each baseline via the `baseline_colors` argument."

        framework_types = baselines + framework_types

        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])
        framework_types = [f_map_type_name[ft] if ft in f_map_type_name else ft for ft in framework_types]

        if plot_tune_types:
            df = df[df["tune_method"].isin(plot_tune_types) | df["method"].isin(baselines)]

        df_plot = df[df["framework_type"].isin(framework_types)]
        # df_plot = df_plot[~df_plot["framework_type"].isin(imputed_names)]

        # pd.set_option('display.max_columns', None)  # todo
        # print(f'{df_plot.head()=}')

        # df_plot_w_mean_2 = df_plot.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()

        df_plot_w_mean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
            metric].mean().reset_index()

        if use_gmean:
            # FIXME: Doesn't plot correctly, need to figure out error bars for geometric mean
            df_plot_eps = df_plot.copy(deep=True)
            df_plot_eps[metric] += 0.01
            from scipy.stats import gmean
            df_plot_w_gmean_per_dataset = df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[
                metric].apply(gmean).reset_index()
            df_plot_w_mean_per_dataset = df_plot_w_gmean_per_dataset

        df_plot_w_mean_2 = df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[
            metric].mean().reset_index()

        df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=lower_is_better)
        baseline_means = {}
        for baseline in baselines:
            baseline_means[baseline] = df_plot_w_mean_2[df_plot_w_mean_2["framework_type"] == baseline][metric].iloc[0]

        df_plot_w_mean_2 = df_plot_w_mean_2[~df_plot_w_mean_2["framework_type"].isin(baselines)]

        df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")

        framework_type_order = list(df_plot_mean_dedupe["framework_type"].to_list())
        framework_type_order.reverse()

        # change to names
        # df_plot_w_mean_per_dataset["framework_type"] = df_plot_w_mean_per_dataset["framework_type"].map(f_map_type_name)

        # sns.set_color_codes("pastel")
        # with sns.plotting_context("notebook", font_scale=0.8, rc={
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'font.size': 10.95,
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        #     # 'legend.framealpha': 0.5,
        #     'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
        # }):

        if use_latex:
            matplotlib.rcParams.update(bundles.neurips2024())
            matplotlib.rcParams.update(fonts.neurips2024_tex())
            rc_context_params = {
                'font.family': 'serif',
                "text.usetex": True,
            } | fontsizes.neurips2024(default_smaller=0)
        else:
            rc_context_params = {}

        with sns.axes_style("whitegrid"):
            # with plt.rc_context({'font.family': 'serif', "text.usetex": True, 'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 12}):
            with plt.rc_context(rc_context_params
                                # | figsizes.neurips2024(height_to_width_ratio=0.8)
                                ):
            # with plt.rc_context(fontsizes.neurips2024() | fonts.neurips2024()):
                # with plt.rc_context(figsizes.neurips2024(height_to_width_ratio=0.8)):
                colors = sns.color_palette("pastel").as_hex()
                errcolors = sns.color_palette("deep").as_hex()

                if use_lim and not lim:
                    lim = [0, None]
                if use_y:
                    pos = metric
                    y = framework_col
                    figsize = (3.5, 3)
                    xlim = lim

                    framework_type_order.reverse()

                else:
                    pos = framework_col
                    y = metric
                    ylim = lim
                    figsize = (7, 2.7)
                    # figsize = None

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                # fig, ax = plt.subplots(1, 1)

                if use_y:
                    baseline_func = ax.axvline
                else:
                    baseline_func = ax.axhline

                linewidth = 0.0 if use_y else 0.3
                err_linewidth = 1.6
                err_linewidths = {
                    'tuned_ensembled': err_linewidth,
                    'tuned': err_linewidth * 0.8,
                    'default': err_linewidth * 0.6,
                    'holdout_tuned_ensembled': err_linewidth * 0.6,
                }
                err_alpha = 0.6

                to_plot = [
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"],
                        ax=ax,
                        order=framework_type_order, color=colors[2],
                        width=0.6, linewidth=linewidth,
                        err_kws={"color": errcolors[2], "linewidth": err_linewidths['tuned_ensembled'], 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Default (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.7, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (Holdout)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[5],
                    #     width=0.65, linewidth=linewidth,
                    #     err_kws={"color": errcolors[5]},
                    # ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"], ax=ax,
                        order=framework_type_order,
                        color=colors[1],
                        width=0.5, linewidth=linewidth,
                        err_kws={"color": errcolors[1], "linewidth": err_linewidths['tuned'], 'alpha': err_alpha},
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Default",
                        data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"], ax=ax,
                        order=framework_type_order, color=colors[0],
                        width=0.4, linewidth=linewidth,
                        err_kws={"color": errcolors[0], "linewidth": err_linewidths['default'], 'alpha': err_alpha},
                        alpha=1.0,
                    ),
                    dict(
                        x=pos, y=y,
                        # hue="tune_method",  # palette=["m", "g", "r],
                        label="Tuned + Ensembled (Holdout)",
                        data=df_plot_w_mean_per_dataset[
                            df_plot_w_mean_per_dataset["tune_method"] == "holdout_tuned_ensembled"], ax=ax,
                        order=framework_type_order,
                        color=colors[3],
                        width=0.3, linewidth=linewidth,
                        err_kws={"color": errcolors[3], "linewidth": err_linewidths['holdout_tuned_ensembled'],
                                 'alpha': err_alpha},
                    ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Best",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "best"], ax=ax,
                    #     order=framework_type_order, color=colors[3],
                    #     width=0.55, linewidth=linewidth,
                    #     err_kws={"color": errcolors[3]},
                    #     alpha=1.0,
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_4h"], ax=ax,
                    #     order=framework_type_order,
                    #     color=colors[4],
                    #     width=0.5, linewidth=linewidth,
                    #     err_kws={"color": errcolors[4]},
                    # ),
                    # dict(
                    #     x=x, y=y,
                    #     # hue="tune_method",  # palette=["m", "g", "r],
                    #     label="Tuned + Ensembled (4h)",
                    #     data=df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled_4h"], ax=ax,
                    #     order=framework_type_order, color=colors[5],
                    #     width=0.4,
                    #     err_kws={"color": errcolors[5]},
                    # ),

                ]

                if use_score:
                    widths = [plot_line["width"] for plot_line in to_plot]
                    colors = [plot_line["color"] for plot_line in to_plot]
                    err_kws_lst = [plot_line["err_kws"] for plot_line in to_plot]

                    # to_plot.reverse()
                    for plot_line, width, color, err_kws in zip(to_plot, widths, colors, err_kws_lst):
                        if same_width:
                            plot_line["width"] = 0.6 * 1.3
                        else:
                            plot_line["width"] = width * 1.3
                        # plot_line["color"] = color
                        # plot_line["err_kws"] = err_kws

                for plot_line in to_plot:
                    boxplot = sns.barplot(**plot_line)

                if use_y:
                    boxplot.set(xlabel='Elo' if metric=='elo' else 'Normalized score', ylabel=None)
                else:
                    boxplot.set(xlabel=None, ylabel='Elo' if metric=='elo' else 'Normalized score')  # remove "Method" in the x-axis
                # boxplot.set_title("Effect of tuning and ensembling")

                # FIXME: (Nick) HACK, otherwise it isn't in the plot, don't know why
                if use_elo:
                    if baseline_means and "Portfolio-N200 (ensemble) (4h)" in baselines:
                        max_baseline_mean = max([v for k, v in baseline_means.items()])
                        if ylim is not None:
                            ylim[1] = max_baseline_mean + 50
                        if xlim is not None:
                            xlim[1] = max_baseline_mean + 50

                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

                ticks = boxplot.get_yticks() if use_y else boxplot.get_xticks()
                ticklabels = [tick.get_text() for tick in
                               (boxplot.get_yticklabels() if use_y else boxplot.get_xticklabels())]

                if use_elo:
                    # ----- add elo error bars -----
                    # Get the bar positions


                    # Add asymmetric error bars manually
                    for pos, framework_type in zip(ticks, ticklabels):
                        for tune_method, errcolor in zip(["default", "tuned", "tuned_ensembled", "holdout_tuned_ensembled"], errcolors):
                            row = df_plot.loc[(df_plot["framework_type"] == framework_type) & (df_plot["tune_method"] == tune_method)]
                            print(f'{row=}')
                            if len(row) == 1:
                                # not all methods have tuned or tuned_ensembled
                                y = row['elo'].values
                                yerr_low = row['elo-'].values
                                yerr_high = row['elo+'].values
                                if use_y:
                                    plotline, caps, barlinecols = plt.errorbar(y, pos, xerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                else:
                                    plotline, caps, barlinecols = plt.errorbar(pos, y, yerr=[yerr_low, yerr_high],
                                                                               fmt='none', color=errcolor,
                                                                               alpha=err_alpha,
                                                                               linewidth=err_linewidths[tune_method])
                                # don't round because it will make the lines longer
                                # plt.setp(barlinecols[0], capstyle="round")


                # ----- highlight bars that contain imputed results -----

                # Map x-tick positions to category labels
                label_lookup = dict(zip(ticks, ticklabels))

                has_imputed = False

                for i, bar in enumerate(boxplot.patches):
                    # Get x-position and convert to category label
                    # todo: this only works for vertical barplots
                    pos = bar.get_y() + bar.get_height() / 2 if use_y else bar.get_x() + bar.get_width() / 2
                    category_index = round(pos)  # x-ticks are usually 0, 1, 2, ...
                    category = label_lookup.get(category_index)

                    # print(f'{category=}')

                    # if category in ['TabPFNv2', 'TabICL', 'TabDPT']:
                    # if category in ['TABPFNV2', 'TABICL', 'TABDPT']:
                    if category in imputed_names:
                        has_imputed = True
                        bar.set_hatch('xx')
                        # bar.set_facecolor('lightgray')
                        # bar.set_edgecolor('black')

                if not use_y:
                    # ----- alternate rows of x tick labels -----
                    # Get current x tick labels
                    labels = [label.get_text() for label in boxplot.get_xticklabels()]

                    # Add newline to every second label
                    new_labels = [label if i % 2 == 0 else r'$\uparrow$' + '\n' + label for i, label in enumerate(labels)]

                    # Apply modified labels
                    boxplot.set_xticklabels(new_labels)

                # remove unnecessary extra space on the sides
                if use_y:
                    plt.ylim(len(boxplot.get_yticklabels()) - 0.35, -0.65)
                else:
                    plt.xlim(-0.5, len(boxplot.get_xticklabels()) - 0.5)


                for baseline_idx, (baseline, color) in enumerate(zip(baselines, baseline_colors)):
                    baseline_mean = baseline_means[baseline]
                    # baseline_func(baseline_mean, label=baseline, color=color, linewidth=2.0, ls="--")
                    baseline_func(baseline_mean, color=color, linewidth=2.0, ls="--", zorder=-10)

                    if baseline == 'Portfolio-N200 (ensemble) (4h)':
                        baseline = 'TabArena ensemble (4h)'

                    if use_y:
                        print(f'{ax.get_ylim()=}')
                        ax.text(y=(1 - 0.035 * (1 + 2*(len(baselines) - 1 - baseline_idx))) * ax.get_ylim()[0],
                                x=baseline_mean * 0.985, s=baseline, ha='right', color=darken_color(color))
                    else:
                        ax.text(x=0.5, y=baseline_mean * 0.97, s=baseline, va='top', color=darken_color(color))


                # ax.legend(loc="upper center", ncol=5)
                # these are not the final legend parameters, see below
                ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.02])

                # reordering the labels
                handles, labels = ax.get_legend_handles_labels()

                # this doesn't work, it also removes the hatch from the actual bars in the plot
                # for handle in handles:
                #     patches = []
                #     if isinstance(handle, Patch):
                #         patches = [handle]
                #     elif isinstance(handle, BarContainer):
                #         patches = handle.patches
                #     for patch in patches:
                #         # remove hatch from existing handles
                #         # It can be present if one of the imputed methods is the best method, e.g., for multiclass
                #         patch.set(hatch=None)

                if has_imputed:
                    # Create a custom legend patch for "imputed"
                    imputed_patch = Patch(facecolor='gray', edgecolor='white', hatch='xx', label='Partially imputed')

                    # Add to existing legend
                    handles.append(imputed_patch)
                    labels.append("Partially imputed")

                # quick fix
                is_holdout_plot = "Tuned + Ensembled (Holdout)" in labels
                if is_holdout_plot:
                    valid_idxs = [i for i, label in enumerate(labels) if label != "Default"]
                    labels = [labels[i] for i in valid_idxs]
                    handles = [handles[i] for i in valid_idxs]

                # specify order
                # len_baselines = len(baselines)
                # len_baselines = 0  # we don't put them in the legend anymore
                # num_other = len(labels) - len_baselines
                # order = [n + len_baselines for n in range(num_other)] + [n for n in range(len_baselines)]
                # order = [3, 4, 5, 0, 1, 2]
                order = list(range(len(labels)))
                order = list(reversed(order))
                # if len(order) == 3:
                #     order = [2, 1, 0]
                print(f'{order=}')

                # pass handle & labels lists along with order as below
                ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
                          ncol=(len(labels)+1)//2 if has_imputed and use_y else len(labels),
                          bbox_to_anchor=[0.35 if use_y else 0.5, 1.05])

                # if use_y:
                #     boxplot.margins(y=0.05)
                # else:
                #     boxplot.margins(x=0.05)

                # ax.legend(bbox_to_anchor=[0.1, 0.5], loc='center left', ncol=5)
                plt.tight_layout()

                if save_prefix:
                    if name_suffix is None:
                        name_suffix = ""
                    fig_path = Path(save_prefix)
                    fig_path.mkdir(parents=True, exist_ok=True)
                    if use_gmean:
                        fig_name = f"tuning-impact-gmean{name_suffix}.pdf"
                    else:
                        fig_name = f"tuning-impact{name_suffix}.pdf"
                    fig_save_path = fig_path / fig_name
                    plt.savefig(fig_save_path)
                if show:
                    plt.show()

    def plot_tabarena_times(self, df: pd.DataFrame, output_dir: str,
                            only_datasets_for_method: dict[str, list[str]] | None = None, show: bool = True):
        # for col in df.columns:
        #     print(df[col])
        # could also use the data from load_task_metadata()
        # df_datasets = pd.read_csv('tabarena_dataset_metadata.csv')
        # df = df.merge(df_datasets[['dataset_name', 'num_instances']],
        #               left_on='dataset',
        #               right_on='dataset_name',
        #               how='left').drop(columns='dataset_name')
        # df_datasets = load_task_metadata()
        # df = df.merge(df_datasets[['name', 'NumberOfInstances']],
        #               left_on='dataset',
        #               right_on='name',
        #               how='left').drop(columns='dataset_name')
        # df = df.rename(columns={"NumberOfInstances": 'num_instances'})
        #
        # df['time_train_s_per_1K'] = df['time_train_s'] * 1000 / (2 / 3 * df['num_instances'])
        # df['time_infer_s_per_1K'] = df['time_infer_s'] * 1000 / (1 / 3 * df['num_instances'])

        # filter to only common datasets
        if only_datasets_for_method is not None:
            for datasets in only_datasets_for_method.values():
                df = df[df["dataset"].isin(datasets)]

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
            "TABDPT",
            "REALMLP",
            "EBM",
            "FT_TRANSFORMER",
            "TABM",
            "MNCA",
        ]

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        df["framework_type"] = df["method"].map(f_map_type).fillna(df["method"])
        df["tune_method"] = df["method"].map(f_map_inverse).fillna("default")
        df = df[df["tune_method"].isin(["default", "tuned_ensembled"])]
        df = df[df['framework_type'].isin(framework_types)]
        df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])

        gpu_methods = ['TabICL', 'TabDPT', 'TabPFNv2']  # todo: add TabM + MNCA once available

        if only_datasets_for_method is not None:
            for method, datasets in only_datasets_for_method.items():
                mask = (df['framework_type'] == method) & (~df['dataset'].isin(datasets))
                # print(f"{df[mask]=}")
                df.loc[mask, 'time_train_s_per_1K'] = np.nan
                df.loc[mask, 'time_infer_s_per_1K'] = np.nan
                # print(f"{df[mask]['time_train_s_per_1K']=}")
                # print(f"{df[mask]['time_infer_s_per_1K']=}")

        # add device name
        framework_types = df["framework_type"].unique()
        device_map = {
            ft: f'{ft} ' + r'(GPU)' if ft in gpu_methods else f'{ft} (CPU)' for ft in framework_types
        }
        df["framework_type"] = df["framework_type"].map(device_map).fillna(df["framework_type"])

        # take mean times
        df = df.groupby(['dataset', 'framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].mean().reset_index()
        df = df.groupby(['framework_type', 'tune_method'])[['time_train_s_per_1K', 'time_infer_s_per_1K']].median().reset_index()

        # ----- ChatGPT plotting code -----

        # Unique values for mapping
        # Sort frameworks by max train time
        sorted_frameworks = (
            df.groupby('framework_type')['time_train_s_per_1K']
            .min()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        frameworks = sorted_frameworks
        y_positions = np.arange(len(frameworks))

        # Maps for tuning method to color and marker
        tune_methods = df['tune_method'].unique()
        # color_map = {tm: c for tm, c in zip(tune_methods, plt.cm.tab10.colors)}
        sns_colors = sns.color_palette("muted").as_hex()
        # sns_colors = sns.color_palette("pastel").as_hex()
        color_map = {'default': sns_colors[0], 'tuned': sns_colors[1], 'tuned_ensembled': sns_colors[2]}
        marker_list = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']
        marker_map = {tm: m for tm, m in zip(tune_methods, marker_list)}

        # Create side-by-side subplots with shared y-axis
        fig, (ax_train, ax_infer) = plt.subplots(
            1, 2, sharey=True, figsize=(5, 4)
        )

        # Alternate row background on both axes
        for i in range(0, len(frameworks), 2):
            for ax in [ax_train, ax_infer]:
                ax.axhspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

        # Plot training and inference times
        for i, fw in enumerate(frameworks):
            df_fw = df[df['framework_type'] == fw]
            for _, row in df_fw.iterrows():
                color = color_map[row['tune_method']]
                marker = marker_map[row['tune_method']]
                ax_train.plot(row['time_train_s_per_1K'], i, marker=marker, color=color, linestyle='None')
                ax_infer.plot(row['time_infer_s_per_1K'], i, marker=marker, color=color, linestyle='None')

        # Train time axis
        ax_train.set_xscale('log')
        ax_train.set_xlabel("Median time per 1K samples [s]")
        ax_train.set_title(r"\textbf{Train+val time}", fontweight='bold')
        ax_train.set_yticks(y_positions)
        ax_train.set_yticklabels(frameworks, fontsize=10)
        ax_train.grid(True, axis='x', alpha=0.5)

        # Inference time axis
        ax_infer.set_xscale('log')
        ax_infer.set_xlabel("Median time per 1K samples [s]")
        ax_infer.set_title(r"\textbf{Inference time}", fontweight='bold')
        ax_infer.set_yticks(y_positions)
        ax_infer.tick_params(labelleft=False)  # Explicitly hide y-tick labels
        ax_infer.grid(True, axis='x', alpha=0.5)

        for ax in [ax_train, ax_infer]:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
            # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        tune_method_display_names = {
            'default': 'Default',
            'tuned': 'Tuned',
            'tuned_ensembled': 'Tuned + Ensembled'
        }

        # Add legend above both plots
        legend_elements = [
            plt.Line2D([0], [0], marker=marker_map[tm], color=color_map[tm],
                       linestyle='None', label=tune_method_display_names[tm], markersize=8)
            for tm in tune_methods
        ]
        fig.legend(handles=legend_elements,  # title='Tuning Method',
                   loc='upper center', bbox_to_anchor=(0.65, 1.01), ncol=3, fontsize=10, title_fontsize=11)

        # Layout adjustment (no clipping)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        path_dir = Path(output_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(path_dir / 'time_plot.pdf')
        if show:
            plt.show()
        plt.close(fig)

    def generate_data_analysis(self):
        generate_dataset_analysis(repo=self.repo, expname_outdir=self.output_dir)

    # FIXME: clean this up
    def generate_runtime_plot(self, df_results: pd.DataFrame):
        from scripts.dataset_analysis import plot_train_time_deep_dive
        df_results_configs = df_results[df_results["method_type"] == "config"]
        df_results_configs = df_results_configs.copy(deep=True)
        # df_results_configs["method"] = df_results_configs["config_type"]

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
            "TABDPT",
            "REALMLP",
            "EBM",
            "FT_TRANSFORMER",
            "TABM",
            "MNCA",
        ]

        df_results_configs = df_results_configs[df_results_configs["config_type"].isin(framework_types)]

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        df_results_configs["method"] = df_results_configs["config_type"].map(f_map_type_name)
        # df["tune_method"] = df["framework"].map(f_map_inverse).fillna("default")

        # FIXME: Hack for tabarena paper
        df_results_configs["is_gpu"] = df_results_configs["framework"].str.endswith("_GPU")
        df_results_configs["method"].loc[df_results_configs["is_gpu"]] = df_results_configs["method"].loc[df_results_configs["is_gpu"]] + " (GPU)"

        plot_train_time_deep_dive(df=df_results_configs, expname_outdir=self.output_dir)


class PaperRunMini(PaperRun):
    def run(self) -> pd.DataFrame:
        df_results_baselines = self.run_baselines()
        df_results_configs = self.run_configs()
        df_results_hpo_all = self.run_hpo_by_family()
        n_portfolios = [200]
        df_results_n_portfolio = []
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=1, n_ensemble_in_name=False))

        df_results_extra = []
        # FIXME: Why does n_max_models_per_type="auto" make things so slow? -> 7 seconds to 107 seconds
        # FIXME: n_max_models_per_type doesn't work correctly atm, need to actually separate the types!
        df_results_extra.append(
            self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto"))
        df_results_extra.append(
            self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        df_results_extra.append(
            self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto",
                        fix_fillna=True))

        df_results_n_portfolio = pd.concat(df_results_n_portfolio + df_results_extra)

        df_results_all = self.evaluator.compare_metrics(results_df=df_results_n_portfolio, configs=[], baselines=[],
                                                        keep_extra_columns=True)

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
        df_results["normalized-error"] = [normalized_scorer.rank(task=(dataset, fold), error=error) for
                                          (dataset, fold, error) in
                                          zip(df_results["dataset"], df_results["fold"], df_results["metric_error"])]
        df_results["seed"] = 0

        import copy
        df_results_rank_compare = copy.deepcopy(df_results)
        df_results_rank_compare = df_results_rank_compare.rename(columns={"method": "framework"})

        self.plot_tuning_impact(df=df_results, framework_types=framework_types,
                                save_prefix=f"{self.output_dir}/tmp/v2_mini", use_gmean=use_gmean)

        # df_results_realmlp_alt = df_results[df_results["method"].str.contains("RealMLP_r") & df_results["method"].str.contains("_alt_")]
        # df_results_realmlp_og = df_results[df_results["method"].str.contains("RealMLP_r") & ~df_results["method"].str.contains("_alt_")]
        #
        # df_results_only_og = df_results.drop(index=df_results_realmlp_alt.index)
        # df_results_only_alt = df_results.drop(index=df_results_realmlp_og.index)
        # self.plot_tuning_impact(df=df_results_only_alt, framework_types=framework_types, save_prefix="tmp/v2_mini_alt", use_gmean=use_gmean)
        #
        # self.plot_tuning_impact(df=df_results_only_og, framework_types=framework_types, save_prefix="tmp/v2_mini_main", use_gmean=use_gmean)

        df_results_rank_compare2 = df_results_rank_compare[
            ~df_results_rank_compare["framework"].str.contains("_BAG_L1") & ~df_results_rank_compare[
                "framework"].str.contains("_r")]
        self.evaluator.plot_overall_rank_comparison(results_df=df_results_rank_compare2,
                                                    save_dir=f"{self.output_dir}/tmp/paper_v2")

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

        plot_family_proportion(df=df_results, save_prefix=f"{self.output_dir}/tmp/family_prop",
                               method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion)
        plot_family_proportion(df=df_results, save_prefix=f"{self.output_dir}/tmp/family_prop2",
                               method="Portfolio-fix_fillna-N200 (ensemble) (4h)",
                               hue_order=hue_order_family_proportion)
