from __future__ import annotations

import copy
import pandas as pd

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
            elo_bootstrap_rounds: int = 10,
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
            )
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
        remove_methods: list[str] | None = None,
        only_datasets_for_method: dict[str, list[str]] | None = None
    ):
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

        if self.datasets is not None:
            df_results = df_results[df_results["dataset"].isin(self.datasets)]
        if self.folds is not None:
            df_results = df_results[df_results["fold"].isin(self.folds)]
        if self.methods is not None:
            df_results = df_results[df_results["method"].isin(self.methods)]
        if self.problem_types is not None:
            df_results = df_results[df_results["problem_type"].isin(self.problem_types)]

        df_results["normalized-error"] = df_results["normalized-error-dataset"]

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

        baselines = [
            # "AutoGluon 1.3 (5m)",
            # "AutoGluon 1.3 (1h)",
            "AutoGluon 1.3 (4h)",
            # "Portfolio-N200 (ensemble) (4h)",
            # "Portfolio-N200 (ensemble, holdout) (4h)",
        ]
        baseline_colors = [
            # "darkgray",
            "black",
            # "blue",
            # "red",
        ]

        df_results_unfiltered = df_results.copy(deep=True)

        if remove_methods is not None:
            f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
                framework_types=framework_types,
                max_runtimes=[
                    (3600 * 4, "_4h"),
                    (None, None),
                ]
            )

            print(f'{df_results["method"].unique().tolist()=}')
            print(f'{df_results["method"].map(f_map_type).unique()=}')

            assert all(method in df_results["method"].map(f_map_type).unique() for method in remove_methods)
            df_results = df_results[~df_results["method"].map(f_map_type).isin(remove_methods)]
            # also remove portfolio baselines except AutoGluon?
            df_results = df_results[(~df_results["method"].map(f_map_type).isna()) | (df_results["method"].isin(baselines))]



        df_results_rank_compare = copy.deepcopy(df_results)

        self.plot_tabarena_times(df=df_results, output_dir=self.output_dir,
                                 only_datasets_for_method=only_datasets_for_method, show=False)

        self.plot_tuning_impact(
            df=df_results,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            use_score=True,
            name_suffix="-normscore-dataset",
            imputed_names=imputed_names,
            show=False,
        )

        if only_norm_scores:
            return

        df_results_rank_compare2 = df_results_rank_compare[~df_results_rank_compare[method_col].str.contains("_BAG_L1") & ~df_results_rank_compare[method_col].str.contains("_r")]

        self.plot_tuning_impact(
            df=df_results,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            use_score=True,
            metric="normalized-error-task",
            name_suffix="-normscore-task",
            imputed_names=imputed_names,
            show=False,
        )

        tabarena = TabArena(
            method_col=method_col,
            task_col="dataset",
            seed_column="fold",
            error_col="metric_error",
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                "normalized-error",
                "normalized-error-task",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        calibration_framework = "RF (default)"

        # configs_all_success = ["TabPFNv2_c1_BAG_L1"]
        # datasets_tabpfn_valid = self.repo.datasets(configs=configs_all_success, union=False)
        # df_results_rank_compare3 = df_results_rank_compare2[df_results_rank_compare2["dataset"].isin(datasets_tabpfn_valid)]

        leaderboard = tabarena.leaderboard(
            data=df_results_rank_compare2,
            # data=df_results_rank_compare3,
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

        print(
            f"Evaluating with {len(df_results_rank_compare2[tabarena.task_col].unique())} datasets... | problem_types={self.problem_types}, folds={self.folds}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard)

        self.plot_tuning_impact(
            df=df_results,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo",
            imputed_names=imputed_names,
            show=False
        )

        self.plot_tuning_impact(
            df=df_results,
            df_elo=leaderboard,
            framework_types=framework_types,
            save_prefix=f"{self.output_dir}",
            use_gmean=use_gmean,
            baselines=baselines,
            baseline_colors=baseline_colors,
            name_suffix="-elo-horizontal",
            imputed_names=imputed_names,
            use_y=True,
            show=False
        )

        results_per_task = tabarena.compute_results_per_task(data=df_results_rank_compare2)

        tabarena.plot_critical_diagrams(results_per_task=results_per_task,
                                        save_path=f"{self.output_dir}/figures/critical-diagram.png", show=False)
        tabarena.plot_critical_diagrams(results_per_task=results_per_task,
                                        save_path=f"{self.output_dir}/figures/critical-diagram.pdf", show=False)

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
            "TabICL",
            "TabDPT",
            # "TabForestPFN",
            "ExplainableBM",
            "NeuralNetFastAI",
            # "FTTransformer",
            "TabM",
            "ModernNCA",
        ]

        # FIXME: TODO (Nick): Move this to its own class for utility plots, no need to re-plot this in every eval call.
        plot_family_proportion(df=df_results_unfiltered, save_prefix=f"{self.output_dir}/figures/family_prop",
                               method="Portfolio-N200 (ensemble) (4h)", hue_order=hue_order_family_proportion,
                               show=False)

        try:
            import autogluon_benchmark
        except:
            print(f"WARNING: autogluon_benchmark failed to import... skipping extra figure generation")
        else:
            self.run_autogluon_benchmark_logic(
                results_per_task=results_per_task,
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
