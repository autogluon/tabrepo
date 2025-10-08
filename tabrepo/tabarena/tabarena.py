from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.stats import gmean

from tabrepo.tabarena.elo_utils import EloHelper

RANK = "rank"
ERROR_COUNT = "error_count"
RANK_1 = "rank=1_count"
IMPROVABILITY = "improvability"
LOSS_RESCALED = "loss_rescaled"
TIME_TRAIN_S = "time_train_s"
TIME_INFER_S = "time_infer_s"
BEST_ERROR = "BEST_ERROR"
WORST_ERROR = "WORST_ERROR"


# TODO: Should "data" be an init arg? Probably not.
# TODO: "fold" what to do?
# TODO: raise_on_missing?
class TabArena:
    def __init__(
        self,
        method_col: str = "method",
        task_col: str = "task",
        error_col: str = "metric_error",
        columns_to_agg_extra: list[str] | str | None = "auto",
        groupby_columns: list[str] | None = None,
        seed_column: str | None = None,
        negative_error_threshold: float = -1e-15,
    ):
        self.method_col = method_col
        self.task_col = task_col
        self.error_col = error_col
        if columns_to_agg_extra is None:
            columns_to_agg_extra = []
        elif columns_to_agg_extra == "auto":
            columns_to_agg_extra = ["time_train_s", "time_infer_s"]
        self.columns_to_agg_extra = columns_to_agg_extra
        self.columns_to_agg = [self.error_col] + self.columns_to_agg_extra
        if groupby_columns is None:
            groupby_columns = []
        self.groupby_columns = [self.method_col, self.task_col] + groupby_columns
        self.seed_column = seed_column
        self.negative_error_threshold = negative_error_threshold

        for c in self.columns_to_agg:
            assert c not in self.groupby_columns
        if self.seed_column is not None:
            assert self.seed_column not in self.columns_to_agg
            assert self.seed_column not in self.groupby_columns
        # FIXME: Folds

    def leaderboard(
        self,
        data: pd.DataFrame,
        include_error: bool = False,
        include_improvability: bool = True,
        include_rescaled_loss: bool = True,
        include_rank_counts: bool = False,
        include_failure_counts: bool = False,
        include_elo: bool = False,
        include_winrate: bool = False,
        include_mrr: bool = False,
        include_relative_error: bool = False,
        include_skill_score: bool = False,
        baseline_relative_error: str | None = None,
        relative_error_kwargs: dict | None = None,
        elo_kwargs: dict | None = None,
        sort_by: str | list[str] | None = "rank",
    ):
        if elo_kwargs is None:
            elo_kwargs = {}
        if relative_error_kwargs is None:
            relative_error_kwargs = {}
        if baseline_relative_error is None:
            baseline_relative_error = elo_kwargs.get("calibration_framework", None)

        self.verify_data(data=data)
        results_per_task = self.compute_results_per_task(data=data)

        results_agg = self.aggregate(results_by_dataset=results_per_task)
        results_lst = [results_agg]

        if include_rank_counts:
            results_lst.append(self.compute_ranks(results_per_task=results_per_task))
        if include_failure_counts:
            results_lst.append(self.compute_failure_count(results_per_task=results_per_task).to_frame())
        if include_elo:
            results_lst.append(self.compute_elo(results_per_task=results_per_task, **elo_kwargs))
        if include_winrate:
            results_lst.append(self.compute_winrate(results_per_task=results_per_task).to_frame())
        if include_mrr:
            results_lst.append(self.compute_mrr(results_per_task=results_per_task).to_frame())
        if baseline_relative_error is not None:
            if include_relative_error:
                results_lst.append(
                    self.compute_relative_error(
                        results_per_task=results_per_task,
                        method_baseline=baseline_relative_error,
                        **relative_error_kwargs
                    ).to_frame()
                )
            if include_skill_score:
                results_lst.append(
                    self.compute_skill_score(results_per_task=results_per_task, method_baseline=baseline_relative_error)
                )

        if include_improvability:
            tasks = list(results_per_task[self.task_col].unique())
            improvability_bootstrap = get_bootstrap_result_lst(
                data=tasks,
                func_=self._weighted_groupby_mean,
                func_kwargs={"data": results_per_task, "agg_column": IMPROVABILITY},
                num_round=100,
            )
            improvability_quantiles = pd.DataFrame({
                f"{IMPROVABILITY}-": results_agg[IMPROVABILITY] - improvability_bootstrap.quantile(.025),
                f"{IMPROVABILITY}+": improvability_bootstrap.quantile(.975) - results_agg[IMPROVABILITY],
            })
            results_lst.append(improvability_quantiles)

        # FIXME: fillna should occur after failure counts?
        results = pd.concat(results_lst, axis=1)

        if not include_error:
            results = results.drop(columns=[self.error_col])
        if not include_rescaled_loss:
            results = results.drop(columns=[LOSS_RESCALED])
        if not include_improvability:
            results = results.drop(columns=[IMPROVABILITY])
        if sort_by is not None:
            results = results.sort_values(by=sort_by)
        results.index.name = self.method_col

        return results

    def verify_data(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)
        data_columns = list(data.columns)
        data_columns_set = set(data_columns)
        assert len(data_columns) == len(data_columns_set)

        missing_columns = []
        present_columns = []
        for c in self.columns_to_agg:
            if c not in data_columns_set:
                missing_columns.append(c)
            else:
                present_columns.append(c)
        for c in self.groupby_columns:
            if c not in data_columns_set:
                missing_columns.append(c)
            else:
                present_columns.append(c)
        if self.seed_column is not None:
            if self.seed_column not in data_columns_set:
                missing_columns.append(self.seed_column)
            else:
                present_columns.append(self.seed_column)

        required_columns = self.groupby_columns + self.columns_to_agg
        if self.seed_column is not None:
            required_columns.append(self.seed_column)
        unused_columns = [d for d in data_columns if d not in required_columns]

        if missing_columns:
            index_names = data.index.names
            missing_in_index = []
            for index_name in index_names:
                if index_name in missing_columns:
                    missing_in_index.append(index_name)
            if missing_in_index:
                msg_extra = (
                    "Columns exist in the index that are required to be columns! "
                    "\n\tEnsure you reset your index to make these columns available: `data = data.reset_index()`\n"
                )
            else:
                msg_extra = ""
            raise ValueError(
                f"{msg_extra}"
                f"Missing required columns:"
                f"\n\tMissing columns ({len(missing_columns)}): {missing_columns}"
                f"\n\tExisting columns ({len(present_columns)}): {present_columns}"
                f"\n\tUnused columns ({len(unused_columns)}): {unused_columns}"
                f"\n\tIndex names ({len(index_names)}): {index_names}"
            )
        if unused_columns:
            print(f"Unused columns: {unused_columns}")

        for c in self.groupby_columns:
            assert data[c].isnull().sum() == 0, f"groupby column {c!r} contains NaN!"
        for c in self.columns_to_agg:
            assert is_numeric_dtype(data[c]), f"aggregation columns must be numeric!"
        for c in self.columns_to_agg:
            if data[c].isnull().sum() != 0:
                invalid_samples = data[data[c].isnull()]

                raise AssertionError(
                    f"Column {c} should not contain null values. "
                    f"Found {data[c].isnull().sum()}/{len(data)} null values! "
                    f"Invalid samples:\n{invalid_samples.head(100).to_markdown()}"
                )

        # TODO: Check no duplicates
        len_data = len(data)
        unique_val_columns = [self.task_col, self.method_col]
        if self.seed_column is not None:
            unique_val_columns.append(self.seed_column)
        len_data_dedupe = len(data.drop_duplicates(unique_val_columns))
        assert len_data == len_data_dedupe

        self.verify_data_is_dense(data=data)
        self.verify_error(data=data)

    def verify_data_is_dense(self, data: pd.DataFrame):
        methods = list(data[self.method_col].unique())
        num_methods = len(methods)
        # FIXME: seed_column
        datasets = list(data[self.task_col].unique())
        num_datasets = len(datasets)

        counts = data[[self.method_col, self.task_col]].value_counts()
        task_cols = [self.task_col]
        if self.seed_column is not None:
            task_cols.append(self.seed_column)
        unique_tasks = data[task_cols].drop_duplicates().reset_index(drop=True)

        unique_seeds_per_dataset = unique_tasks[self.task_col].value_counts()
        num_tasks = unique_seeds_per_dataset.sum()
        valid_tasks_per_method = data[self.method_col].value_counts()
        valid_methods_per_dataset = data[self.task_col].value_counts()
        valid_methods_per_task = data[task_cols].value_counts()
        invalid_tasks_per_method = (-valid_tasks_per_method + num_tasks).sort_values(ascending=False)
        invalid_methods_per_dataset = (
                -valid_methods_per_dataset + valid_methods_per_dataset.index.map(unique_seeds_per_dataset) * num_methods
        ).sort_values(ascending=False)
        invalid_methods_per_task = (
                -valid_methods_per_task + num_methods
        ).sort_values(ascending=False)

        if (invalid_tasks_per_method != 0).any():
            invalid_tasks_per_method_filtered = invalid_tasks_per_method[invalid_tasks_per_method != 0]
            invalid_methods_per_dataset_filtered = invalid_methods_per_dataset[invalid_methods_per_dataset != 0]
            invalid_methods_per_task_filtered = invalid_methods_per_task[invalid_methods_per_task != 0]
            num_invalid_results = invalid_tasks_per_method.sum()
            # num_invalid_tasks = invalid_methods_per_task_filtered.sum()

            df_experiments_dense = unique_tasks.merge(
                pd.Series(data=methods, name=self.method_col),
                how="cross",
            )
            experiment_cols = task_cols + [self.method_col]
            overlap = pd.merge(df_experiments_dense, data[experiment_cols], on=experiment_cols, how='left', indicator='exist')
            df_missing_experiments = overlap[overlap["exist"] == "left_only"][experiment_cols].sort_values(by=experiment_cols).reset_index(drop=True)

            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                if len(df_missing_experiments) <= 500:
                    print(f"\nFailed Experiments ({len(df_missing_experiments)}):")
                    print(df_missing_experiments)
                print("\nMethods sorted by failure count:")
                print(invalid_tasks_per_method_filtered)
                print("\nDatasets sorted by failure count:")
                print(invalid_methods_per_dataset_filtered)
            # missing results
            raise AssertionError(
                f"Missing results for some methods. Ensure that all methods have results for all tasks.\n"
                f"If failures exist, fill missing values before passing into this method.\n"
                f"{len(invalid_tasks_per_method_filtered)}/{num_methods} methods with missing tasks. {num_invalid_results} missing results.\n"
                f"{len(invalid_methods_per_dataset_filtered)}/{num_datasets} datasets with missing methods.\n"
                f"{len(invalid_methods_per_task_filtered)}/{num_tasks} tasks with missing methods.\n"
                f"Methods sorted by failure count:\n"
                f"{invalid_tasks_per_method_filtered}"
            )

    def verify_error(self, data: pd.DataFrame):
        min_error = data[self.error_col].min()
        if min_error < 0:
            data_invalid = data[data[self.error_col] < 0]
            num_invalid = len(data_invalid)
            raise ValueError(
                f"Found {num_invalid} rows where {self.error_col} is less than 0! Error can never be less than 0. "
                f"Ensure your error is computed correctly."
                f"\nMinimum value found: {min_error}"
                f"\nSometimes floating point precision can result in a tiny negative value. "
                f"You can fix this by doing: data['{self.error_col}'] = data['{self.error_col}'].clip(lower=0)"
            )

    # TODO: Consider moving this to a different class or finding a better separation.
    #  The eval code becomes a lot more complicated if we need to account for improperly formatted / invalid data.
    def clean_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        data = copy.deepcopy(data)
        min_error = data[self.error_col].min()
        if min_error < 0:
            if min_error >= self.negative_error_threshold:
                data[self.error_col] = data[self.error_col].clip(0)
            else:
                self.verify_error(data=data)
        return data

    # FIXME: Cleanup
    # FIXME: Other fill methods
    # FIXME: What about fill value of other columns besides self.error_col?
    def fillna_data(
        self,
        data: pd.DataFrame,
        df_fillna: pd.DataFrame | None = None,
        fillna_method: str = "worst",
    ) -> pd.DataFrame:
        """
        Fills missing (dataset, fold, framework) rows in data with the (dataset, fold) row in df_fillna.

        Parameters
        ----------
        data
        df_fillna

        Returns
        -------

        """
        if self.seed_column:
            task_columns = [self.task_col, self.seed_column]
        else:
            task_columns = [self.task_col]

        unique_methods = list(data[self.method_col].unique())

        if fillna_method == "worst":
            assert df_fillna is None, f"df_fillna must be None if fillna_method='worst'"
            idx_worst = data.groupby(task_columns)[self.error_col].idxmax()
            df_fillna = data.loc[idx_worst]
            df_fillna = df_fillna.drop(columns=[self.method_col])
            pass

        data = data.set_index([*task_columns, self.method_col], drop=True)

        assert self.method_col not in df_fillna.columns, f"Method column '{self.method_col}' must not be in df_fillna"

        df_filled = df_fillna[task_columns].merge(
            pd.Series(data=unique_methods, name=self.method_col),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(data.index)

        # fill valid values
        fill_cols = list(data.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(data.dtypes)
        df_filled.loc[data.index] = data

        df_fillna = df_fillna.set_index(task_columns, drop=True)
        a = df_fillna.loc[nan_vals.droplevel(level=self.method_col)]
        a.index = nan_vals
        df_filled.loc[nan_vals] = a
        data = df_filled

        data = data.reset_index(drop=False)

        return data

    # FIXME: Don't hard-code time_train_s time_infer_s
    # FIXME: Failures, mean when folds are missing is wrong, prevent duplicates
    def compute_results_per_task(self, data: pd.DataFrame) -> pd.DataFrame:
        groupby_cols = self.groupby_columns
        columns_to_agg = self.columns_to_agg
        results_per_task = data[groupby_cols + columns_to_agg].groupby(groupby_cols).mean().reset_index()

        best_error_per_task, worst_error_per_task = self.compute_best_and_worst_error_per_task(results_per_task=results_per_task)

        results_per_task[BEST_ERROR] = results_per_task[self.task_col].map(best_error_per_task)
        results_per_task[WORST_ERROR] = results_per_task[self.task_col].map(worst_error_per_task)

        results_per_task[IMPROVABILITY] = 1 - (results_per_task[BEST_ERROR] / results_per_task[self.error_col])
        results_per_task[IMPROVABILITY] = results_per_task[IMPROVABILITY].fillna(0)

        results_per_task[LOSS_RESCALED] = (results_per_task[self.error_col] - results_per_task[BEST_ERROR]) / (
                results_per_task[WORST_ERROR] - results_per_task[BEST_ERROR]
        )
        results_per_task[LOSS_RESCALED] = results_per_task[LOSS_RESCALED].fillna(0)
        results_per_task = results_per_task.drop([BEST_ERROR, WORST_ERROR], axis=1)

        for time_attr in [TIME_TRAIN_S, TIME_INFER_S]:
            if time_attr in columns_to_agg:
                best_time_attr = "BEST_" + time_attr
                best_speed = (
                    results_per_task[[self.task_col, time_attr]].sort_values(time_attr, ascending=True).drop_duplicates(self.task_col)
                )
                best_speed.columns = [self.task_col, best_time_attr]
                results_per_task = results_per_task.merge(best_speed, on=self.task_col)
                results_per_task[time_attr + "_rescaled"] = results_per_task[time_attr] / results_per_task[best_time_attr]
                results_per_task = results_per_task.drop([best_time_attr], axis=1)

        results_per_task = self.rank_result(results_per_task)
        return results_per_task

    def compute_best_and_worst_error_per_task(self, results_per_task: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
        """

        Parameters
        ----------
        results_per_task: pd.DataFrame

        Returns
        -------
        best_error_per_task: dict[str, float]
            Mapping of task to error, where error is the lowest (best) error value of all methods in `results_per_task`.
        worst_error_per_task: dict[str, float]
            Mapping of task to error, where error is the highest (worst) error value of all methods in `results_per_task`.
        """
        best_error_per_task = results_per_task[[self.task_col, self.error_col]].sort_values(
            self.error_col, ascending=True,
        ).drop_duplicates(self.task_col).set_index(self.task_col)[self.error_col].to_dict()
        worst_error_per_task = results_per_task[[self.task_col, self.error_col]].sort_values(
            self.error_col, ascending=False,
        ).drop_duplicates(self.task_col).set_index(self.task_col)[self.error_col].to_dict()
        return best_error_per_task, worst_error_per_task

    def aggregate(self, results_by_dataset: pd.DataFrame) -> pd.DataFrame:
        results_by_dataset = copy.deepcopy(results_by_dataset)
        results_agg = results_by_dataset.groupby([self.method_col, self.task_col]).mean(numeric_only=True)
        # Compute mean
        mean_df = results_agg.groupby([self.method_col]).mean(numeric_only=True)

        # Compute median and prefix column names
        median_df = results_by_dataset.groupby([self.method_col]).median(numeric_only=True)
        median_df.columns = [f'median_{col}' for col in median_df.columns]

        # Combine mean and median
        results_agg = pd.concat([mean_df, median_df], axis=1)
        # results_agg = results_by_dataset.groupby([self.method_col]).mean(numeric_only=True)
        return results_agg

    def compute_ranks(self, results_per_task: pd.DataFrame) -> pd.DataFrame:
        results_ranked = pd.DataFrame(index=list(results_per_task[self.method_col].unique()))
        rank_1 = results_per_task[results_per_task[RANK] == 1]
        rank_1_count = rank_1[self.method_col].value_counts()
        results_ranked["rank=1_count"] = rank_1_count
        results_ranked["rank=1_count"] = results_ranked["rank=1_count"].fillna(0).astype(int)

        rank_2 = results_per_task[(results_per_task[RANK] > 1) & (results_per_task[RANK] <= 2)]
        rank_2_count = rank_2[self.method_col].value_counts()

        results_ranked["rank=2_count"] = rank_2_count
        results_ranked["rank=2_count"] = results_ranked["rank=2_count"].fillna(0).astype(int)

        rank_3 = results_per_task[(results_per_task[RANK] > 2) & (results_per_task[RANK] <= 3)]
        rank_3_count = rank_3[self.method_col].value_counts()

        results_ranked["rank=3_count"] = rank_3_count
        results_ranked["rank=3_count"] = results_ranked["rank=3_count"].fillna(0).astype(int)

        rank_l3 = results_per_task[(results_per_task[RANK] > 3)]
        rank_l3_count = rank_l3[self.method_col].value_counts()

        results_ranked["rank>3_count"] = rank_l3_count
        results_ranked["rank>3_count"] = results_ranked["rank>3_count"].fillna(0).astype(int)
        return results_ranked

    def compute_mrr(self, results_per_task: pd.DataFrame) -> pd.Series:
        """Compute mean reciprocal rank"""
        results_per_task = results_per_task.copy()
        results_per_task["mrr"] = 1 / results_per_task["rank"]
        results_mrr = results_per_task.groupby(self.method_col)["mrr"].mean()
        return results_mrr

    def compute_failure_count(self, results_per_task: pd.DataFrame) -> pd.Series:
        datasets = sorted(list(results_per_task[self.task_col].unique()))
        frameworks = list(results_per_task[self.method_col].unique())
        num_datasets = len(datasets)
        results_framework = results_per_task.drop_duplicates(self.groupby_columns)
        framework_counts = results_framework[self.method_col].value_counts()
        framework_failure_counts = -framework_counts + num_datasets
        framework_failure_counts.name = "error_count"
        framework_failure_counts = framework_failure_counts.reindex(frameworks)
        return framework_failure_counts

    def compute_skill_score(
        self,
        results_per_task: pd.DataFrame,
        method_baseline: str,
    ) -> pd.Series:
        relative_error_gmean = self.compute_relative_error(
            results_per_task=results_per_task, method_baseline=method_baseline, agg="gmean",
        )
        skill_score = 1 - relative_error_gmean
        skill_score.name = "skill_score"
        return skill_score

    def compute_elo(
        self,
        results_per_task: pd.DataFrame,
        calibration_framework: str | None = None,
        calibration_elo: int | None = None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
        include_quantiles: bool = True,
        round_decimals: int | None = 1,
        use_bootstrap_median: bool = False,
        use_bootstrap_median_for_quantiles: bool = False,
        clip_negative_ci: bool = True,
    ) -> pd.DataFrame:
        """
        Compute Elo ratings for methods evaluated across multiple tasks.

        This aggregates per-task results into head-to-head “battles” and estimates
        per-method Elo scores either by maximum likelihood (single fit) or by a
        bootstrap procedure. Optionally returns uncertainty bars derived from the
        bootstrap distribution.

        Parameters
        ----------
        results_per_task
            Long-form DataFrame with one row per (method, task) containing an error metric.
            Must contain the columns referenced by ``self.method_col`` (method identifier),
            ``self.task_col`` (task identifier), and ``self.error_col`` (lower is better).
        calibration_framework
            Optional name of a reference method to anchor the Elo scale (e.g.,
            set that method’s Elo to ``calibration_elo``).
        calibration_elo
            Elo value assigned to ``calibration_framework`` when provided.
            Ignored if ``calibration_framework`` is ``None``.
        INIT_RATING
            Initial rating used to start optimization / simulation.
        BOOTSTRAP_ROUNDS
            Number of bootstrap resamples of tasks to estimate uncertainty.
            If set to 1, no resampling is performed and quantiles (if requested) collapse to the point estimate.
        SCALE
            Logistic scale factor in the Elo win-probability model (typical value is 400).
            Larger values make probabilities less sensitive to rating differences.
        include_quantiles
            If ``True``, include 2.5%/97.5% quantile bars (or point bars when ``BOOTSTRAP_ROUNDS == 1``).
        round_decimals
            If not ``None``, round the returned values to this many decimal places.
        use_bootstrap_median
            If ``True``, use the bootstrap median rating as the primary Elo estimate instead of the MLE point estimate.
        use_bootstrap_median_for_quantiles
            If ``True``, center the ± bars around the bootstrap median,
            otherwise they are centered around the chosen Elo point estimate.
        clip_negative_ci
            If ``True``, negative widths for ``elo+``/``elo-`` are clipped to 0.
            Negative width can occur if ``use_bootstrap_median=False`` and ``use_bootstrap_median_for_quantiles=False``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by method (index name = ``self.method_col``) sorted
            by descending Elo. Always contains:

            - ``elo`` : float
                The Elo rating for each method (rounded if ``round_decimals`` is set).

            If ``include_quantiles`` is ``True``, also contains:

            - ``elo+`` : float
                Upper error bar width (e.g., 97.5% quantile minus center).
            - ``elo-`` : float
                Lower error bar width (e.g., center minus 2.5% quantile).

            When ``BOOTSTRAP_ROUNDS == 1``, ``elo+`` and ``elo-`` will be 0.
        """
        elo_helper = EloHelper(method_col=self.method_col, task_col=self.task_col, error_col=self.error_col)
        battles = elo_helper.convert_results_to_battles(results_df=results_per_task)

        bootstrap_median = None
        bootstrap_elo_lu = None
        bars_quantiles = None
        if use_bootstrap_median or (include_quantiles and BOOTSTRAP_ROUNDS > 1):
            bootstrap_elo_lu = elo_helper.compute_elo_ratings(
                battles=battles,
                calibration_framework=calibration_framework,
                calibration_elo=calibration_elo,
                INIT_RATING=INIT_RATING,
                BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,
                SCALE=SCALE,
            )
            bootstrap_median = bootstrap_elo_lu.quantile(.5)

        if use_bootstrap_median:
            elo = bootstrap_median
        else:
            elo = elo_helper.compute_mle_elo(
                battles=battles,
                INIT_RATING=INIT_RATING,
                SCALE=SCALE,
                calibration_framework=calibration_framework,
                calibration_elo=calibration_elo,
            )

        if include_quantiles:
            if BOOTSTRAP_ROUNDS > 1:
                assert bootstrap_elo_lu is not None
                bars_quantiles = pd.DataFrame(dict(
                    lower=bootstrap_elo_lu.quantile(.025),
                    upper=bootstrap_elo_lu.quantile(.975),
                ))
            else:
                print(
                    f"Warning: Returning 95% CI quantiles for elo when BOOTSTRAP_ROUNDS<=1. "
                    f"The CI is invalid and widths will be set to 0."
                )
                bars_quantiles = pd.DataFrame(dict(
                    lower=elo,
                    upper=elo,
                ))

        bars = pd.DataFrame(dict(
            elo=elo,
        ))

        if include_quantiles:
            assert bars_quantiles is not None
            if use_bootstrap_median_for_quantiles:
                relative_to = bootstrap_median
            else:
                relative_to = elo
            bars['elo+'] = bars_quantiles['upper'] - relative_to
            bars['elo-'] = relative_to - bars_quantiles["lower"]

            if clip_negative_ci:
                bars['elo+'] = bars['elo+'].clip(lower=0)
                bars['elo-'] = bars['elo-'].clip(lower=0)

        bars = bars.sort_values(by="elo", ascending=False)
        if round_decimals is not None:
            bars['elo'] = np.round(bars['elo'], round_decimals)
            if include_quantiles:
                bars['elo+'] = np.round(bars['elo+'], round_decimals)
                bars['elo-'] = np.round(bars['elo-'], round_decimals)

        bars.index.name = self.method_col

        return bars

    def compute_relative_error(
        self,
        results_per_task: pd.DataFrame,
        method_baseline: str | None,
        agg: str = "mean",
        use_optimal: bool = False,
    ) -> pd.Series:
        assert agg in ["mean", "gmean"]
        results_per_task = results_per_task.copy()
        results_per_task["relative_error"] = self._relative_error_per_task(
            results_per_task=results_per_task,
            method_baseline=method_baseline,
            use_optimal=use_optimal,
        )
        if agg == "mean":
            relative_error = results_per_task.groupby(self.method_col)["relative_error"].mean()
        elif agg == "gmean":
            relative_error = results_per_task.groupby(self.method_col)["relative_error"].apply(gmean)
        else:
            raise ValueError(f"Invalid value for `agg`: {agg}")
        return relative_error

    def _relative_error_per_task(
        self,
        results_per_task: pd.DataFrame,
        method_baseline: str | None,
        use_optimal: bool = False,
    ) -> pd.Series:
        tasks = set(results_per_task[self.task_col].unique().tolist())

        if use_optimal:
            baseline_result = results_per_task.groupby(self.task_col)[self.error_col].min()
        else:
            assert method_baseline is not None, f"method_baseline must not be None!"
            baseline_result = results_per_task[results_per_task[self.method_col] == method_baseline]
            assert len(baseline_result) > 0, f"Baseline '{method_baseline}' does not exist!"
            tasks_baseline = set(baseline_result[self.task_col].unique().tolist())
            assert tasks == tasks_baseline, (
                f"Baseline '{method_baseline}' missing results for "
                f"{len(tasks) - len(tasks_baseline)}/{len(tasks)} tasks!"
            )
            # TODO: Assert baseline in all dataset results
            baseline_result = baseline_result.set_index(self.task_col)[self.error_col]
        baseline_error = results_per_task[self.task_col].map(baseline_result)
        relative_error = results_per_task[self.error_col] / baseline_error
        return relative_error

    def compute_winrate(self, results_per_task: pd.DataFrame) -> pd.Series:
        """
        This code is more complex than simply rescaling ranks as it accounts for missing (task, framework) results

        If the results are dense, this method is equivalent to:
        ```
        results_winrate = 1 - ((results["rank"] - 1) / (len(results)-1))
        ```
        """
        elo_helper = EloHelper(method_col=self.method_col, task_col=self.task_col, error_col=self.error_col)
        battles = elo_helper.convert_results_to_battles(results_df=results_per_task)
        results_winrate = elo_helper.compute_winrate_from_battles(battles=battles)
        return results_winrate

    # FIXME: Rounding, parameterize
    #  Maybe rounding should be done as a preprocessing?
    # FIXME: Why a for-loop?
    #  Can we simply groupby rank?
    def rank_result(self, result_df: pd.DataFrame) -> pd.DataFrame:
        datasets = list(result_df[self.task_col].unique())
        result_df = result_df.copy()
        result_df[self.error_col] = [round(x[0], 5) for x in zip(result_df[self.error_col])]
        num_frameworks = len(result_df[self.method_col].unique())
        if num_frameworks == 1:
            sorted_df_full = result_df
            sorted_df_full[RANK] = 1
        else:
            dfs = []
            for dataset in datasets:
                dataset_df = result_df[result_df[self.task_col] == dataset]
                sorted_df = dataset_df.copy()
                sorted_df[RANK] = sorted_df[self.error_col].rank()
                dfs.append(sorted_df)
            sorted_df_full = pd.concat(dfs, ignore_index=True)

        return sorted_df_full

    def dataset_outlier(self, results_per_task: pd.DataFrame):
        # Compute how much of an outlier the results of a given dataset are (squared rank differential?)
        raise NotImplementedError

    # TODO: Should plotting live in a separate class?
    def plot_critical_diagrams(self, results_per_task, save_path: str | None = None, show: bool = False, reverse: bool = False):
        import matplotlib.pyplot as plt
        with plt.rc_context({'text.usetex': False}):
            from autorank import autorank
            from autorank._util import cd_diagram
            plt.rcParams.update({'font.size': 12})

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)

            data = results_per_task.pivot_table(index=self.task_col, columns=self.method_col, values="rank")
            result = autorank(data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric")

            try:
                ax = cd_diagram(result, reverse=reverse, ax=ax, width=6)
            except KeyError:
                print(f"Not enough methods to generate cd_diagram, skipping...")
                return

            # plt.tight_layout()  # cuts off text
            if save_path is not None:
                parent_dir = str(Path(save_path).parent)
                os.makedirs(parent_dir, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            if show:
                plt.show()

    def _weighted_groupby_mean(self, tasks: list[str], data: pd.DataFrame, agg_column: str) -> pd.Series:
        num_tasks = len(tasks)
        data = data[[self.method_col, self.task_col, agg_column]].copy()

        counts = {}
        for task in tasks:
            counts[task] = counts.get(task, 0) + 1
        counts = {k: v / num_tasks for k, v in counts.items()}
        weights = data[self.task_col].map(counts).fillna(0)
        data["_weighted_column"] = data[agg_column] * weights
        column_mean = data.groupby(self.method_col)["_weighted_column"].sum()
        return column_mean


def get_bootstrap_result_lst(data: list, func_, rng=None, num_round: int = None, func_kwargs=None, seed: int = 0):
    rows = []
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if func_kwargs is None:
        func_kwargs = {}
    if num_round is None:
        rows.append(func_(data, **func_kwargs))
    else:
        num_data = len(data)
        for i in range(num_round):
            data_new = rng.choice(data, size=num_data, replace=True)
            rows.append(func_(data_new, **func_kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]
