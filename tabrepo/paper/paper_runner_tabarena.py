from __future__ import annotations

import copy
import math
from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from scripts.baseline_comparison.evaluate_utils import plot_family_proportion
from .paper_runner import PaperRun
from tabrepo.tabarena.tabarena import TabArena
from .paper_utils import get_framework_type_method_names


class PaperRunTabArena(PaperRun):
    def __init__(
            self,
            *,
            methods: list[str] | None = None,
            folds: list[int] | None = None,
            datasets: list[str] | None = None,
            problem_types: list[str] | None = None,
            banned_model_types: list[str] | None = None,
            elo_bootstrap_rounds: int = 100,
            keep_best: bool = False,
            **kwargs,
    ):
        """

        Parameters
        ----------
        methods
            filter methods
        folds
            filter folds
        datasets
            filter datasets
        problem_types
            filter problem_types
        elo_bootstrap_rounds
            10 = toy
            100 = paper
        kwargs
        """
        super().__init__(**kwargs)
        self.datasets = datasets
        self.problem_types = problem_types
        self.methods = methods
        self.folds = folds
        self.elo_bootstrap_rounds = elo_bootstrap_rounds
        self.banned_model_types = banned_model_types
        self.keep_best = keep_best

    def run(self) -> pd.DataFrame:
        df_results_all_no_sim = self.run_no_sim()
        df_results_single_best_families = self.run_zs_family()

        n_portfolios = [200, 100, 50, 20, 10, 5]
        df_results_n_portfolio = []
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
            df_results_n_portfolio.append(self.run_zs(n_portfolios=n_portfolio, n_ensemble=1, n_ensemble_in_name=False))

        df_results_extra = []
        # FIXME: Why does n_max_models_per_type="auto" make things so slow? -> 7 seconds to 107 seconds
        # FIXME: n_max_models_per_type doesn't work correctly atm, need to actually separate the types!
        # df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto"))
        # df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # # df_results_extra.append(self.run_zs(n_portfolios=200, n_ensemble=None, n_ensemble_in_name=False, n_max_models_per_type="auto", fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=50, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=50, n_ensemble=None, max_runtime=None, n_ensemble_in_name=False, fix_fillna=True))
        # df_results_extra.append(self.run_zs(n_portfolios=20, n_ensemble=None, n_ensemble_in_name=False, fix_fillna=True))
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

        df_results_all = pd.concat(df_results_n_portfolio + df_results_extra + [df_results_single_best_families])

        df_results_all = pd.concat([
            df_results_all,
            df_results_all_no_sim,
        ], ignore_index=True)

        print(df_results_all)
        return df_results_all

    def run_only_portfolio_200(self) -> pd.DataFrame:
        n_portfolio = 200
        return self.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False)

    def run_no_sim(self, model_types: list[str] | None = None) -> pd.DataFrame:
        """
        Run logic that isn't impacted by other methods or other datasets

        Returns
        -------

        """
        df_results_baselines = self.run_baselines()
        df_results_configs = self.run_configs()
        df_results_hpo_all = self.run_hpo_by_family(
            include_uncapped=True,
            include_4h=False,
            model_types=model_types,
        )

        df_results_all = pd.concat([
            df_results_configs,
            df_results_baselines,
            df_results_hpo_all,
        ], ignore_index=True)

        return df_results_all

    def run_zs_family(self) -> pd.DataFrame:
        config_type_groups = self.get_config_type_groups(ban_families=True)

        df_single_best_portfolio_family_lst = []
        for family, family_configs in config_type_groups.items():
            df_single_best_portfolio_family = self.run_zs(
                n_portfolios=1,
                n_ensemble=1,
                fix_fillna=True,
                configs=family_configs,
                time_limit=None,
            )
            df_single_best_portfolio_family["method_subtype"] = "best"
            df_single_best_portfolio_family["config_type"] = family
            df_single_best_portfolio_family["framework"] = f"{family} (best)"
            df_single_best_portfolio_family_lst.append(df_single_best_portfolio_family)

        df_single_best_portfolio_families = pd.concat(df_single_best_portfolio_family_lst, ignore_index=True)
        return df_single_best_portfolio_families

    def compute_normalized_error(self, df_results: pd.DataFrame, static: bool = False) -> pd.DataFrame:
        """
        Adds normalized-error-task and normalized-error-dataset columns to df_results.

        Parameters
        ----------
        df_results
        static: bool, default False
            If True, calculates normalized error based on the methods in `self.repo`.
                This is less fair and penalizes new methods in `df_results` that frequently beat the best method in `self.repo`.
                For example, portfolios.
                However, it provides a static meaning to any given normalized error value that is tied to `self.repo`
                and doesn't change when adding new experiments to `df_results`.
            If False, calculates normalized error based on the methods in `df_results`.
                This is the fairest and accurately reflects the performance of methods,
                but all methods will have their values change when a new method is added.

        Returns
        -------
        df_results
            With two additional columns:
                "normalized-error-task"
                "normalized-error-dataset"

        """
        if not static:
            return self.compute_normalized_error_dynamic(df_results=df_results)

        method_col = "framework"

        df_results = df_results.copy(deep=True)
        df_results_og = df_results.copy(deep=True)

        df_results_per_dataset = df_results.groupby([method_col, "dataset"])["metric_error"].mean().reset_index(
            drop=False)

        from tabrepo.utils.normalized_scorer import NormalizedScorer
        from tabrepo.paper.paper_utils import make_scorers
        rank_scorer, normalized_scorer_task = make_scorers(self.repo)

        df_results_baselines = pd.concat([
            self.repo._zeroshot_context.df_configs_ranked,
            self.repo._zeroshot_context.df_baselines,
        ], ignore_index=True)

        df_comparison_per_dataset = df_results_baselines.groupby([method_col, "dataset"])["metric_error"].mean()
        df_comparison_per_dataset = df_comparison_per_dataset.reset_index(drop=False)
        # Standard normalized-error, only computed off of real experiments, not impacted by simulation runs.
        # This is biased against very strong simulation results because they can't get better than `0.0` on a dataset.
        normalized_scorer_dataset = NormalizedScorer(df_comparison_per_dataset,
                                                     tasks=list(df_results_baselines["dataset"].unique()),
                                                     baseline=None, task_col="dataset", framework_col=method_col)

        df_results["normalized-error-task"] = [normalized_scorer_task.rank(task=(dataset, fold), error=error) for
                                               (dataset, fold, error) in
                                               zip(df_results["dataset"], df_results["fold"],
                                                   df_results["metric_error"])]

        df_results_per_dataset["normalized-error-dataset"] = [
            normalized_scorer_dataset.rank(task=dataset, error=error) for (dataset, error) in
            zip(df_results_per_dataset["dataset"], df_results_per_dataset["metric_error"])
        ]

        df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
            "normalized-error-dataset"]
        df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

        df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
        df_results_og["normalized-error-task"] = df_results["normalized-error-task"]

        return df_results_og

    @classmethod
    def compute_normalized_error_dynamic(cls, df_results: pd.DataFrame) -> pd.DataFrame:
        df_results = df_results.copy(deep=True)
        df_results_og = df_results.copy(deep=True)

        df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"], errors="ignore")

        method_col = "framework"

        df_results_per_dataset = df_results.groupby([method_col, "dataset"])["metric_error"].mean().reset_index(
            drop=False)

        from tabrepo.utils.normalized_scorer import NormalizedScorer

        # Alternative, this also incorporates Portfolios and HPO into the normalized scoring. This makes normalized-error dependent on what simulations we run.
        # This is unbiased against very strong simulation results because the best method defines what is `0.0` on a dataset.
        normalized_scorer_dataset = NormalizedScorer(
            df_results_per_dataset,
            tasks=list(df_results_per_dataset["dataset"].unique()),
            baseline=None,
            task_col="dataset",
            framework_col=method_col,
        )

        all_tasks = df_results[["dataset", "fold"]].drop_duplicates().values.tolist()
        all_tasks = [tuple(task) for task in all_tasks]

        normalized_scorer_task = NormalizedScorer(
            df_results,
            tasks=all_tasks,
            baseline=None,
            task_col=["dataset", "fold"],
            framework_col=method_col,
        )

        df_results["normalized-error-task"] = [normalized_scorer_task.rank(task=(dataset, fold), error=error) for
                                               (dataset, fold, error) in
                                               zip(df_results["dataset"], df_results["fold"],
                                                   df_results["metric_error"])]

        df_results_per_dataset["normalized-error-dataset"] = [
            normalized_scorer_dataset.rank(task=dataset, error=error) for (dataset, error) in
            zip(df_results_per_dataset["dataset"], df_results_per_dataset["metric_error"])
        ]

        df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
            "normalized-error-dataset"]
        df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

        df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
        df_results_og["normalized-error-task"] = df_results["normalized-error-task"]
        return df_results_og

    def eval(
        self,
        df_results: pd.DataFrame,
        use_gmean: bool = False,
        only_norm_scores: bool = False,
        imputed_names: list[str] | None = None,
        only_datasets_for_method: dict[str, list[str]] | None = None,
        baselines: list[str] | str | None = "auto",
        baseline_colors: list[str] | None = None,
        plot_tune_types: list[str] | None = None,
        plot_times: bool = False,
        plot_extra_barplots: bool = False,
        plot_cdd: bool = True,
        plot_other: bool = False,
        framework_types_extra: list[str] = None,
        calibration_framework: str | None = "auto",
    ):
        if framework_types_extra is None:
            framework_types_extra = []
        if calibration_framework is not None and calibration_framework == "auto":
            calibration_framework = "RF (default)"
        if baselines is "auto":
            baselines = [
                # "AutoGluon 1.3 (5m)",
                # "AutoGluon 1.3 (1h)",
                "AutoGluon 1.3 (4h)",
                # "Portfolio-N200 (ensemble) (4h)",
                # "Portfolio-N200 (ensemble, holdout) (4h)",
            ]
            if baseline_colors is None:
                baseline_colors = [
                    # "darkgray",
                    "black",
                    # "blue",
                    # "red",
                ]
        if baseline_colors is None:
            baseline_colors = []
        if baselines is None:
            baselines = []
        assert len(baselines) == len(baseline_colors)
        method_col = "method"
        df_results = df_results.copy(deep=True)
        if "rank" in df_results:
            df_results = df_results.drop(columns=["rank"])
        if "seed" not in df_results:
            df_results["seed"] = 0
        df_results["seed"] = df_results["seed"].fillna(0).astype(int)
        df_results = df_results.drop_duplicates(subset=[
            "dataset", "fold", "framework", "seed"
        ], keep="first")

        if "normalized-error-dataset" not in df_results:
            df_results = self.compute_normalized_error_dynamic(df_results=df_results)
        assert "normalized-error-dataset" in df_results, f"Run `self.compute_normalized_error_dynamic(df_results)` first to get normalized-error."

        df_results = df_results.rename(columns={
            "framework": method_col,
        })

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
        if framework_types_extra:
            framework_types += framework_types_extra
            framework_types = list(set(framework_types))

        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if self.folds is not None:
            df_results = df_results[df_results["fold"].isin(self.folds)]
        if self.methods is not None:
            df_results = df_results[df_results["method"].isin(self.methods)]
        if self.problem_types is not None:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]

        df_results["normalized-error"] = df_results["normalized-error-dataset"]

        # ----- add times per 1K samples -----
        df_datasets = load_task_metadata(paper=True)
        df_results = df_results.merge(df_datasets[['name', 'NumberOfInstances']],
                      left_on='dataset',
                      right_on='name',
                      how='left').drop(columns='name')
        df_results = df_results.rename(columns={"NumberOfInstances": 'num_instances'})

        df_results['time_train_s_per_1K'] = df_results['time_train_s'] * 1000 / (2 / 3 * df_results['num_instances'])
        df_results['time_infer_s_per_1K'] = df_results['time_infer_s'] * 1000 / (1 / 3 * df_results['num_instances'])

        # df_results = self.evaluator.compare_metrics(results_df=df_results, configs=[], baselines=[], keep_extra_columns=True, fillna=True)

        df_results["method"] = df_results["method"].map({
            "AutoGluon_bq_4h8c": "AutoGluon 1.3 (4h)",
            "AutoGluon_bq_1h8c": "AutoGluon 1.3 (1h)",
            "AutoGluon_bq_5m8c": "AutoGluon 1.3 (5m)",
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

        }).fillna(df_results["method"])
        # print(df_results)

        df_results_rank_compare = copy.deepcopy(df_results)
        df_results_unfiltered = copy.deepcopy(df_results)

        # FIXME: (Nick) Unsure which form of the df should go in here?
        # David H: doesn't matter since results are not relative to other methods in the df
        if only_datasets_for_method is not None and plot_times:
            self.plot_tabarena_times(df=df_results_unfiltered, output_dir=self.output_dir,
                                     only_datasets_for_method=only_datasets_for_method, show=False)

        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        print(f'{df_results["method"].unique().tolist()=}')
        print(f'{df_results["method"].map(f_map_type).unique()=}')

        banned_model_types = self.banned_model_types or []

        assert all(method in df_results_rank_compare["method"].map(f_map_type).unique() for method in banned_model_types)
        df_results_rank_compare = df_results_rank_compare[~df_results_rank_compare["method"].map(f_map_type).isin(banned_model_types)]
        # also remove portfolio baselines except AutoGluon?
        df_results_rank_compare = df_results_rank_compare[(~df_results_rank_compare["method"].map(f_map_type).isna()) | (df_results_rank_compare["method"].isin(baselines))]

        if self.banned_model_types:
            framework_types = [f for f in framework_types if f not in self.banned_model_types]

        if not self.keep_best:
            df_results_rank_compare = df_results_rank_compare[~df_results_rank_compare[method_col].str.contains("(best)", regex=False)]

        print(f'{df_results_rank_compare["method"].unique().tolist()=}')

        # recompute normalized errors (requires temporarily renaming "method" to "framework"
        df_results_rank_compare = self.compute_normalized_error_dynamic(df_results_rank_compare.rename(columns={
            method_col: "framework",
        })).rename(columns={
            "framework": method_col,
        })

        # ----- end removing unused methods -----

        hue_order_family_proportion = [
            "CatBoost",
            "TabPFNv2",
            "TabM",
            "ModernNCA",
            "TabDPT",
            "LightGBM",
            "TabICL",

            "RandomForest",
            "XGBoost",
            "RealMLP",

            "NeuralNetFastAI",
            "ExplainableBM",
            "NeuralNetTorch",
            "ExtraTrees",

            # "LinearModel",
            # "KNeighbors",

            # "TabForestPFN",

            # "FTTransformer",

        ]

        # FIXME: TODO (Nick): Move this to its own class for utility plots, no need to re-plot this in every eval call.
        # plot_family_proportion(df=df_results_unfiltered, save_prefix=f"{self.output_dir}/figures/family_prop",
        #                        method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion,
        #                        show=False)

        self.plot_tuning_impact(
            df=df_results_rank_compare,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            use_score=True,
            name_suffix="-normscore-dataset-horizontal",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            show=False,
            use_y=True
        )

        if plot_extra_barplots:
            self.plot_tuning_impact(
                df=df_results_rank_compare,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                use_score=True,
                name_suffix="-normscore-dataset",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False,
            )

            self.plot_tuning_impact(
                df=df_results_rank_compare,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                use_score=True,
                metric="normalized-error-task",
                name_suffix="-normscore-task",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False,
            )

        if only_norm_scores:
            return

        tabarena = TabArena(
            method_col=method_col,
            task_col="dataset",
            seed_column="fold",
            error_col="metric_error",
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                "time_train_s_per_1K",
                "time_infer_s_per_1K",
                "normalized-error",
                "normalized-error-task",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        # configs_all_success = ["TabPFNv2_c1_BAG_L1"]
        # datasets_tabpfn_valid = self.repo.datasets(configs=configs_all_success, union=False)
        # df_results_rank_compare3 = df_results_rank_compare[df_results_rank_compare["dataset"].isin(datasets_tabpfn_valid)]

        leaderboard = tabarena.leaderboard(
            data=df_results_rank_compare,
            # data=df_results_rank_compare3,
            include_winrate=True,
            include_mrr=True,
            # include_failure_counts=True,
            include_rank_counts=True,
            include_elo=True,
            elo_kwargs=dict(
                calibration_framework=calibration_framework,
                calibration_elo=1000,
                BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
            )
        )
        elo_map = leaderboard["elo"]
        leaderboard = leaderboard.reset_index(drop=False)

        from autogluon.common.savers import save_pd
        save_pd.save(path=f"{self.output_dir}/tabarena_leaderboard.csv", df=leaderboard)

        self.create_leaderboard_latex(leaderboard, framework_types=framework_types, save_dir=self.output_dir)

        print(
            f"Evaluating with {len(df_results_rank_compare[tabarena.task_col].unique())} datasets... | problem_types={self.problem_types}, folds={self.folds}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard)

        self.plot_tuning_impact(
            df=df_results_rank_compare,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo-horizontal",
            imputed_names=imputed_names,
            plot_tune_types=plot_tune_types,
            use_y=True,
            show=False
        )

        if plot_extra_barplots:
            self.plot_tuning_impact(
                df=df_results_rank_compare,
                df_elo=leaderboard,
                framework_types=framework_types,
                save_prefix=f"{self.output_dir}",
                use_gmean=use_gmean,
                baselines=baselines,
                baseline_colors=baseline_colors,
                name_suffix="-elo",
                imputed_names=imputed_names,
                plot_tune_types=plot_tune_types,
                show=False
            )

        results_per_task = tabarena.compute_results_per_task(data=df_results_rank_compare)

        print(f'{results_per_task.columns=}')

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            return name.replace('(tuned + ensemble)', '(tuned + ensembled)')

        # use tuned+ensembled version if available, and default otherwise
        tune_methods = results_per_task["method"].map(f_map_inverse)
        method_types = results_per_task["method"].map(f_map_type).fillna(results_per_task["method"])
        tuned_ens_types = method_types[tune_methods == 'tuned_ensembled']
        results_te_per_task = results_per_task[(tune_methods == 'tuned_ensembled') | ((tune_methods == 'default') & ~method_types.isin(tuned_ens_types))]

        # rename model part
        results_te_per_task.loc[:, "method"] = results_te_per_task["method"].map(rename_model)

        if plot_cdd:
            # tabarena.plot_critical_diagrams(results_per_task=results_te_per_task,
            #                                 save_path=f"{self.output_dir}/figures/critical-diagram.png", show=False)
            tabarena.plot_critical_diagrams(results_per_task=results_te_per_task,
                                            save_path=f"{self.output_dir}/figures/critical-diagram.pdf", show=False)

        if plot_other:
            try:
                import autogluon_benchmark
            except:
                print(f"WARNING: autogluon_benchmark failed to import... skipping extra figure generation")
            else:
                results_per_task_ag_benchmark = results_per_task.rename(columns={
                    "champ_delta": "bestdiff",
                })
                self.run_autogluon_benchmark_logic(
                    results_per_task=results_per_task_ag_benchmark,
                    elo_map=elo_map,
                    tabarena=tabarena,
                    calibration_framework=calibration_framework,
                )

    def run_autogluon_benchmark_logic(self, results_per_task: pd.DataFrame, elo_map: dict, tabarena: TabArena,
                                      calibration_framework: str):
        """
        Requires autogluon_benchmark installed:

        Parameters
        ----------
        results_per_task
        elo_map
        tabarena
        calibration_framework

        Returns
        -------

        """
        results_per_task_rename = results_per_task.rename(columns={
            tabarena.method_col: "framework",
            tabarena.task_col: "dataset",
            tabarena.error_col: "metric_error",
        })

        results_per_task_rename["elo"] = results_per_task_rename["framework"].map(elo_map)

        # Nick: Comment out this code if you don't have autogluon_benchmark
        from autogluon_benchmark.plotting.plotter import Plotter
        plotter = Plotter(
            results_ranked_df=results_per_task_rename,
            results_ranked_fillna_df=results_per_task_rename,
            save_dir=f"{self.output_dir}/figures/plotter"
        )

        # FIXME: Nick: This isn't yet merged, as I haven't made it nice yet
        # plotter.plot_pareto_time_infer_elo(data=results_per_task_rename)
        # plotter.plot_pareto_time_train_elo(data=results_per_task_rename)

        plotter.plot_all(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=self.elo_bootstrap_rounds,
        )

        # self.evaluator.plot_overall_rank_comparison(results_df=df_results_rank_compare2, save_dir=f"{self.output_dir}/paper_v2")

    def plot_ensemble_weights_heatmap(self, df_ensemble_weights: pd.DataFrame, **kwargs):
        # FIXME: if family never present, then this won't work
        p = self.evaluator.plot_ensemble_weights(df_ensemble_weights=df_ensemble_weights, **kwargs)
        fig_path = Path(f"{self.output_dir}/figures")
        fig_path.mkdir(parents=True, exist_ok=True)

        p.savefig(fig_path / "ens-weights-per-dataset.png")
        p.savefig(fig_path / "ens-weights-per-dataset.pdf")

    def get_ensemble_weights(
        self,
        df_results: pd.DataFrame,
        method: str,
        excluded_families: list[str] = None,
        aggregate_folds: bool = False,
    ) -> pd.DataFrame:
        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if excluded_families is None:
            excluded_families = []

        df_results_method = df_results[df_results["framework"] == method]

        df_ensemble_weights = df_results_method[["dataset", "fold", "ensemble_weight"]]

        full_dict = []
        # available_configs = set()
        # for ensemble_weights in df_ensemble_weights["ensemble_weight"].values:
        #     for k in ensemble_weights.keys():
        #         if k not in available_configs:
        #             available_configs.add(k)
        #     ens_weights_w_dataset_fold = ensemble_weights.copy(deep=True)
        #     full_dict.append(ensemble_weights)
        #
        # df_ensemble_weights_2 = pd.DataFrame()

        for d, f, ensemble_weights in zip(df_ensemble_weights["dataset"], df_ensemble_weights["fold"], df_ensemble_weights["ensemble_weight"]):
            ens_weights_w_dataset_fold = dict()
            ens_weights_w_dataset_fold["dataset"] = d
            ens_weights_w_dataset_fold["fold"] = f
            ens_weights_w_dataset_fold.update(ensemble_weights)
            full_dict.append(ens_weights_w_dataset_fold)
            pass

        model_to_families = self.repo.configs_type()

        model_families = set()
        for m, f in model_to_families.items():
            if f not in model_families:
                model_families.add(f)

        weight_per_family_dict = []
        for cur_dict in full_dict:
            new_dict = {}
            for k, v in cur_dict.items():
                if k == "dataset":
                    new_dict["dataset"] = v
                elif k == "fold":
                    new_dict["fold"] = v
                else:
                    model_family = model_to_families[k]
                    if model_family not in new_dict:
                        new_dict[model_family] = 0
                    new_dict[model_family] += v
            weight_per_family_dict.append(new_dict)

        import pandas as pd
        df = pd.DataFrame(weight_per_family_dict)
        df = df.set_index(["dataset", "fold"])
        df = df.fillna(0)

        df_cols = df.columns
        f_to_add = []
        for f in model_families:
            if f not in df_cols:
                f_to_add.append(f)
        df[f_to_add] = 0

        if excluded_families:
            df = df.drop(columns=excluded_families)

        df = self.evaluator.get_ensemble_weights(
            df_ensemble_weights=df,
            aggregate_folds=aggregate_folds,
            sort_by_mean=True,
        )

        return df

    def plot_portfolio_ensemble_weights_barplot(self, df_ensemble_weights: pd.DataFrame):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib.colors as mcolors
        import numpy as np

        fig, ax = plt.subplots(1, 1,
                               figsize=(3.5, 3)
                               )

        df_long = df_ensemble_weights.melt(var_name="Model", value_name="Weight")
        model_order = list(df_ensemble_weights.columns)

        start_color = mcolors.to_rgb('tab:green')  # Color for vanilla MLP
        end_color = mcolors.to_rgb('tab:blue')  # Color for final MLP
        palette = [mcolors.to_hex(c) for c in np.linspace(start_color, end_color, len(model_order))]
        alphas = np.linspace(0.5, 0.6, len(model_order))[::-1]
        # palette = sns.color_palette("light:b", n_colors=len(model_order))[::-1]

        barplot = sns.barplot(
            data=df_long,
            x="Weight",  # This is now the horizontal axis
            y="Model",  # Categories on the vertical axis
            hue="Model",
            legend=False,
            ax=ax,
            order=model_order,  # Optional: control order
            # palettes: 'coolwarm',
            palette=palette,
            err_kws={'color': 'silver'},
        )

        # Set alpha only for bar face colors, not error bars
        for patch, alpha in zip(barplot.patches, alphas):
            r, g, b = patch.get_facecolor()[:3]  # Ignore original alpha
            patch.set_facecolor((r, g, b, alpha))  # Set new alpha

        # TODO: Make horizontal?
        # TODO: Drop TabPFN / TabICL columns if you want
        # TODO: Better formatting / nicer style?
        # TODO: Title, xaxis, yaxis names, figsize
        # barplot = sns.barplot(
        #     data=df_ensemble_weights,
        #     ax=ax,
        #     order=list(df_ensemble_weights.columns),
        # )

        barplot.set_xlabel("Average weight in TabArena ensemble")
        barplot.set_ylabel("")

        fig_name = f"portfolio-weight-barplot.pdf"
        fig_prefix = Path(self.output_dir) / "figures"
        fig_prefix.mkdir(parents=True, exist_ok=True)

        fig_save_path = fig_prefix / fig_name
        plt.savefig(fig_save_path)

    def create_leaderboard_latex(self, df: pd.DataFrame, framework_types, save_dir):
        df = df.copy(deep=True)
        f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
            framework_types=framework_types,
            max_runtimes=[
                (3600 * 4, "_4h"),
                (None, None),
            ]
        )

        def rename_model(name: str):
            parts = name.split(" ")
            if parts[0] in f_map_type_name:
                parts[0] = f_map_type_name[parts[0]]
            name = " ".join(parts)
            name = name.replace('(default)', '(D)')
            name = name.replace('(tuned)', '(T)')
            name = name.replace('(tuned + ensemble)', '(T+E)')
            return name

        df = df.sort_values(by="elo", ascending=False)

        df_new = pd.DataFrame()

        print(f'{df.columns=}')

        df_new[r"Model"] = df["method"].map(rename_model)
        # do the more annoying way {}_{...} so that \textbf{} affects the main number
        df_new[r"Elo ($\uparrow$)"] = [f'{round(elo)}' + r'${}_{' + f'-{math.ceil(elom)},+{math.ceil(elop)}' + r'}$'
                                       for elo, elom, elop in zip(df["elo"], df["elo-"], df["elo+"])]
        df_new[r"Norm." + "\n" + r"score ($\uparrow$)"] = [f'{1-err:5.3f}' for err in df["normalized-error"]]
        df_new[r"Avg." + "\n" + r"rank ($\downarrow$)"] = [f'{rank:.1f}' for rank in df["rank"]]
        df_new["Harm.\nmean\n" + r"rank ($\downarrow$)"] = [f'{1/val:.1f}' for val in df["mrr"]]
        df_new[r"\#wins ($\uparrow$)"] = [str(cnt) for cnt in df["rank=1_count"]]
        df_new[f"Improva-\n" + r"bility ($\downarrow$)"] = [f'{100*val:.1f}\\%' for val in df["champ_delta"]]
        df_new[r"Train time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_train_s_per_1K"]]
        df_new[r"Predict time" + "\n" + r"per 1K [s]"] = [f'{t:.2f}' for t in df["median_time_infer_s_per_1K"]]

        # ----- highlight best and second-best numbers per column -----

        # first, convert the strings back to floats
        import re

        def extract_first_float(s):
            """
            Extracts the first sequence of digits (including decimal point) from the input string
            and returns it as a float. Returns None if no valid number is found.
            """
            match = re.search(r'\d+(\.\d+)?', s)
            if match:
                return float(match.group())
            return None

        def find_smallest_and_second_smallest_indices(numbers):
            if len(numbers) < 2:
                return [], []

            # Find the smallest value
            min_val = min(numbers)
            min_indices = [i for i, x in enumerate(numbers) if x == min_val]

            # Exclude the smallest values and find the second smallest
            remaining = [x for x in numbers if x != min_val]
            if not remaining:
                return min_indices, []  # No second smallest

            second_min_val = min(remaining)
            second_min_indices = [i for i, x in enumerate(numbers) if x == second_min_val]

            return min_indices, second_min_indices

        # then, add textbf or underline to the correct rows
        for col_idx, col in enumerate(df_new.columns):
            if r'\uparrow' in col or r'\downarrow' in col:
                # factor = 1 if r'\downarrow' in col else -1
                # numbers = [factor * extract_first_float(s) for s in df_new[col]]
                ranks = df_new[col].map(extract_first_float).rank(method="min", ascending=r'\downarrow' in col)
                for rank, color in [(1, 'gold'), (2, 'silver'), (3, 'bronze')]:
                    df_new.loc[ranks == rank, col] = df_new.loc[ranks == rank, col].apply(
                        lambda x: f"\\textcolor{{{color}}}{{\\textbf{{{x}}}}}"
                    )

                # min_indices, second_min_indices = find_smallest_and_second_smallest_indices(numbers)
                # for idx in min_indices:
                #     df_new.iloc[idx, col_idx] = r'\textbf{' + df_new.iloc[idx, col_idx] + r'}'
                # for idx in second_min_indices:
                #     df_new.iloc[idx, col_idx] = r'\underline{' + df_new.iloc[idx, col_idx] + r'}'


        # ----- create latex table -----

        rows = []
        rows.append(r'\begin{tabular}{' + 'llcccccrr' + r'}')
        rows.append(r'\toprule')
        # rows.append(' & '.join(df_new.columns) + r' \\')

        col_names_split = [col.split('\n') for col in df_new.columns]
        n_rows_header = max([len(rows) for rows in col_names_split])
        for row_idx in range(n_rows_header):
            rows.append(' & '.join([r'\textbf{' + lst[row_idx] + r'}' if row_idx < len(lst) else ''
                                    for lst in col_names_split]) + r' \\')
        rows.append(r'\midrule')

        for row_index, row in df_new.iterrows():
            rows.append(' & '.join([row[col_name] for col_name in df_new.columns]) + r' \\')

        rows.append(r'\bottomrule')
        rows.append(r'\end{tabular}')

        table = '\n'.join(rows)

        with open(Path(save_dir) / 'leaderboard.tex', 'w') as f:
            f.write(table)




