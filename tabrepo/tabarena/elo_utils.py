from __future__ import annotations

from collections import defaultdict
from typing import List
import logging
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EloHelper:
    def __init__(
        self,
        method_col: str = "method",
        task_col: str = "task",
        error_col: str = "metric_error",
        split_col: str | None = None,
    ):
        self.method_col = method_col
        self.task_col = task_col
        self.error_col = error_col
        self.split_col = split_col

        self.method_1 = f"{self.method_col}_1"
        self.method_2 = f"{self.method_col}_2"

    def compute_mle_elo(
        self,
        battles: pd.DataFrame,
        SCALE: int | float = 400,
        BASE: int = 10,
        INIT_RATING: int | float = 1000,
        calibration_framework: str = None,
        calibration_elo: float = None,
        force_iterative_elo: bool = False,
    ) -> pd.Series:
        """
        MLE ELO with per-task equal weighting via sample_weight.

        Adapted from ChatBot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=4_x-vXL4yxvC
        """
        models = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
        models = pd.Series(np.arange(len(models)), index=models)
        df_original = battles

        # duplicate battles
        battles = pd.concat([battles, battles], ignore_index=True)
        p = len(models.index)
        n = battles.shape[0]

        X = np.zeros([n, p])
        X[np.arange(n), models[battles[self.method_1]]] = +math.log(BASE)
        X[np.arange(n), models[battles[self.method_2]]] = -math.log(BASE)

        Y = np.zeros(n)
        Y[battles["winner"] == "1"] = 1.0

        # tie handling: make half wins for each side
        tie_idx = battles["winner"] == "tie"
        tie_idx[len(tie_idx) // 2:] = False
        Y[tie_idx] = 1.0

        # Build sample weights so each task contributes total weight 1 across its splits
        if "weight" in battles.columns:
            sample_weight = battles["weight"].to_numpy(dtype=float).copy()
        else:
            sample_weight = np.ones(n, dtype=float)

        if len(np.unique(Y)) == 1 or force_iterative_elo:
            logger.warning(
                "compute_mle_elo fell back to iterative ELO (dominance in labels or forced)."
            )
            elo_scores = self.compute_iterative_elo_scores(
                df_original,
                INIT_RATING=INIT_RATING,
                SCALE=SCALE,
                models=models,
            )
        else:
            lr = LogisticRegression(fit_intercept=False)
            lr.fit(X, Y, sample_weight=sample_weight)
            elo_scores = SCALE * lr.coef_[0] + INIT_RATING

        if calibration_framework is not None:
            if calibration_elo is None:
                calibration_elo = INIT_RATING
            elo_scores += (calibration_elo - elo_scores[models[calibration_framework]])
        return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

    def compute_iterative_elo_scores(
        self,
        battles: pd.DataFrame,
        INIT_RATING: int = 1000,
        SCALE: int = 400,
        K_factor: float = 1,
        models: pd.Series = None,
        *,
        epochs: int = 10,
        shuffle: bool = True,
        seed: int | np.random.Generator | None = 0,
    ):
        """
        Iterative ELO with optional deterministic shuffling and multi-epoch passes.

        Parameters
        ----------
        battles : pd.DataFrame
            Battles dataframe (from convert_results_to_battles). If present,
            uses df['weight'] to scale each update so tasks contribute equally.
        INIT_RATING : int
            Initial rating for all models.
        SCALE : int
            ELO scale (e.g., 400).
        K_factor : float
            Step size per update (will be multiplied by per-row 'weight' if present).
        models : pd.Series
            Mapping: index=model name, value=integer id. Built if None.
        epochs : int, default=10
            Number of full passes over the (possibly shuffled) battles.
        shuffle : bool, default=True
            Shuffle battle order each epoch.
        seed : int | np.random.Generator | None, default=0
            Controls the shuffle order. If an int or Generator is provided and
            shuffle=True, the traversal order is deterministic across runs.

        Returns
        -------
        np.ndarray
            ELO scores aligned to `models.index` order.
        """
        if models is None:
            model_names = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
            models = pd.Series(np.arange(len(model_names)), index=model_names)

        # Build numeric arrays for fast iteration
        # Map methods -> indices
        m1_idx = battles[self.method_1].map(models).to_numpy()
        m2_idx = battles[self.method_2].map(models).to_numpy()

        # Winner encoding: 1 -> m1 wins, 2 -> m2 wins, tie -> 0
        winner_raw = battles["winner"].to_numpy()
        # encode: 1, 2, tie
        w_code = np.zeros(len(winner_raw), dtype=np.int8)
        w_code[winner_raw == "1"] = 1
        w_code[winner_raw == "2"] = 2
        # weight per row (defaults to 1)
        if "weight" in battles.columns:
            w_row = battles["weight"].to_numpy(dtype=float, copy=False)
        else:
            w_row = np.ones(len(battles), dtype=float)

        def expected_result(r1, r2):
            return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / SCALE))

        elo_scores = np.full(len(models.index), float(INIT_RATING))

        # RNG (only used if shuffle=True)
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        n = len(battles)
        base_order = np.arange(n)

        for _ in range(max(1, int(epochs))):
            if shuffle:
                order = rng.permutation(n)
            else:
                order = base_order

            for i in order:
                a = m1_idx[i]
                b = m2_idx[i]
                w = float(w_row[i])
                outcome = w_code[i]

                ra = elo_scores[a]
                rb = elo_scores[b]
                ea = expected_result(ra, rb)
                eb = 1.0 - ea  # symmetry

                if outcome == 0:  # tie
                    elo_scores[a] += (K_factor * w) * (0.5 - ea)
                    elo_scores[b] += (K_factor * w) * (0.5 - eb)
                elif outcome == 1:  # m1 wins
                    elo_scores[a] += (K_factor * w) * (1.0 - ea)
                    elo_scores[b] += (K_factor * w) * (0.0 - eb)
                else:  # outcome == 2, m2 wins
                    elo_scores[b] += (K_factor * w) * (1.0 - eb)
                    elo_scores[a] += (K_factor * w) * (0.0 - ea)

        return elo_scores

    def get_bootstrap_result(
        self,
        battles: pd.DataFrame,
        func_compute_elo,
        rng=None,
        num_round: int | None = None,
        func_kwargs=None,
    ):
        """
        Task-level (cluster) bootstrap with multiplicity up-weighting.

        For each bootstrap draw:
          1) Sample tasks with replacement (size = #unique tasks).
          2) Concatenate each *unique* sampled task's rows once (no duplication).
          3) Multiply each row's 'weight' by the task's multiplicity in the draw.

        Notes
        -----
        - This is mathematically equivalent to duplicating rows k times per sampled
          task for the MLE (logistic-regression) path, and practically indistinguishable
          for the iterative ELO path (especially with small K, shuffle, epochs).
        - If the input `battles` has no 'weight' column, it is treated as 1.0
          before multiplicity scaling.

        Parameters
        ----------
        battles : pd.DataFrame
            Battles DF produced by convert_results_to_battles; must include self.task_col.
        func_compute_elo : callable
            Callable(battles_df, **kwargs) -> pd.Series of ELO ratings.
        rng : int | np.random.Generator | None
            Random seed or Generator for deterministic bootstraps.
        num_round : int | None
            Number of bootstrap draws. If None, compute a single point estimate.
        func_kwargs : dict | None
            Extra kwargs forwarded to func_compute_elo.

        Returns
        -------
        pd.DataFrame
            Rows = bootstrap draws; columns = model names. Columns are sorted by
            bootstrap median (descending).
        """
        if func_kwargs is None:
            func_kwargs = {}

        # RNG handling
        if isinstance(rng, np.random.Generator):
            _rng = rng
        elif rng is None:
            _rng = np.random.default_rng()
        else:
            _rng = np.random.default_rng(int(rng))

        # Single (non-bootstrap) run
        if num_round is None:
            res = func_compute_elo(battles, **func_kwargs)
            df = pd.DataFrame([res])
            return df[df.median().sort_values(ascending=False).index]

        task_col = self.task_col
        if task_col not in battles.columns:
            raise ValueError(f"Expected column '{task_col}' in battles for task-level bootstrap.")

        # Pre-group once for efficiency
        grouped = {t: g for t, g in battles.groupby(task_col, sort=False)}
        tasks = np.array(list(grouped.keys()))
        n_tasks = len(tasks)
        if n_tasks == 0:
            raise ValueError("No tasks present in battles; cannot bootstrap.")

        rows = []
        for _ in tqdm(range(num_round), desc="Calculating Elo 95% CI via Bootstrap"):
            # Sample tasks with replacement; compute multiplicities
            sampled = _rng.choice(tasks, size=n_tasks, replace=True)
            uniq, counts = np.unique(sampled, return_counts=True)
            multiplicity = dict(zip(uniq, counts))

            # Concatenate each unique sampled task once
            parts = [grouped[t] for t in uniq]
            battles_boot = pd.concat(parts, axis=0, ignore_index=True)

            # Ensure a weight column exists (treated as 1.0 if missing)
            if "weight" not in battles_boot.columns:
                battles_boot["weight"] = 1.0

            # Up-weight by task multiplicity (vectorized)
            battles_boot["weight"] = battles_boot["weight"].to_numpy(dtype=float) * \
                                     battles_boot[task_col].map(multiplicity).to_numpy(dtype=float)

            # Compute ELO on the bootstrap sample
            rows.append(func_compute_elo(battles_boot, **func_kwargs))

        df = pd.DataFrame(rows)
        # Stable, tidy column order: sort by bootstrap median descending
        return df[df.median().sort_values(ascending=False).index]

    def calc_battle_outcome(self, error_1: float, error_2: float) -> str:
        if error_1 < error_2:
            winner = "1"
        elif error_1 > error_2:
            winner = "2"
        else:
            winner = "tie"
        return winner

    def convert_results_to_battles(
        self,
        results_df: pd.DataFrame,
        frameworks: List[str] = None,
        datasets: List[str] = None,
    ) -> pd.DataFrame:
        # Keep only needed columns (+ split if available)
        cols = [self.method_col, self.task_col, self.error_col]
        if self.split_col is not None:
            cols.append(self.split_col)
        results_df = results_df[cols].copy()

        if datasets is not None:
            results_df = results_df[results_df[self.task_col].isin(datasets)]
        if frameworks is not None:
            results_df = results_df[results_df[self.method_col].isin(frameworks)]

        # Determine how to pair: by (task, split) if split_col is given; else by task only
        if self.split_col is not None:
            on_cols = [self.task_col, self.split_col]
        else:
            on_cols = [self.task_col]

        results_df_after_dedupe = results_df.drop_duplicates(subset=[self.method_col, *on_cols])
        if len(results_df_after_dedupe) != len(results_df):
            raise AssertionError(f"Found {len(results_df) - len(results_df_after_dedupe)} duplicate rows!")

        # Pair each method with every other method on the same task (and split if provided)
        results_pairs_df = pd.merge(
            results_df,
            results_df,
            on=on_cols,
            suffixes=('_1', '_2'),
            how='inner',
        )

        # Keep only pairs with different methods
        mask_diff_method = results_pairs_df[self.method_1] != results_pairs_df[self.method_2]
        results_pairs_df = results_pairs_df.loc[mask_diff_method].copy()

        # Winner of the pair
        results_pairs_df["winner"] = [
            self.calc_battle_outcome(e1, e2)
            for e1, e2 in zip(results_pairs_df[f"{self.error_col}_1"], results_pairs_df[f"{self.error_col}_2"])
        ]

        # Avoid counting each battle twice (dedupe A vs B with B vs A) by method pair (orderless) within the same on_cols
        # Build a canonical key for the unordered pair
        pair_key = list(zip(
            np.minimum(results_pairs_df[self.method_1], results_pairs_df[self.method_2]),
            np.maximum(results_pairs_df[self.method_1], results_pairs_df[self.method_2]),
        ))

        # Create a boolean mask that keeps the first occurrence of each (task[, split], unordered pair)
        # Easiest way: use a DataFrame of keys and drop_duplicates, then reindex
        key_df = pd.DataFrame({**{c: results_pairs_df[c].values for c in on_cols},
                               "__a": [k[0] for k in pair_key],
                               "__b": [k[1] for k in pair_key]})
        keep_idx = key_df.drop_duplicates(subset=on_cols + ["__a", "__b"]).index
        results_pairs_df = results_pairs_df.iloc[keep_idx].copy()

        # Compute per-task weights so each task contributes total weight 1.
        # If split_col exists: n_splits = nunique splits per task, else n_splits := 1
        if self.split_col is not None:
            splits_per_task = results_df.groupby(self.task_col)[self.split_col].nunique()
        else:
            splits_per_task = results_df.groupby(self.task_col).size()
            splits_per_task[:] = 1  # treat as 1 split per task

        # Map weight = 1 / n_splits(task)
        results_pairs_df["weight"] = 1.0 / results_pairs_df[self.task_col].map(splits_per_task).astype(float)

        # Final output columns (keep weight; keep split if present for debugging/analysis)
        out_cols = [self.method_1, self.method_2, "winner", self.task_col, "weight"]
        if self.split_col is not None and self.split_col in results_pairs_df.columns:
            out_cols.append(self.split_col)
        return results_pairs_df[out_cols]

    def compute_elo_ratings(
        self,
        battles: pd.DataFrame,
        seed: int = 0,
        calibration_framework=None,
        calibration_elo=None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed=seed)
        bootstrap_elo_lu = self.get_bootstrap_result(
            battles=battles,
            func_compute_elo=self.compute_mle_elo,
            num_round=BOOTSTRAP_ROUNDS,
            rng=rng,
            func_kwargs={
                "INIT_RATING": INIT_RATING,
                "SCALE": SCALE,
                "calibration_framework": calibration_framework,
                "calibration_elo": calibration_elo,
            }
        )
        return bootstrap_elo_lu

    def compute_elo_rating_dataset_contributon(
        self,
        results_ranked_fillna_df: pd.DataFrame,
        seed: int = 0,
        calibration_framework=None,
        calibration_elo=None,
        INIT_RATING: float = 1000,
        SCALE: int = 400,
    ) -> pd.DataFrame:
        datasets = list(results_ranked_fillna_df[self.task_col].unique())
        battles = self.convert_results_to_battles(results_ranked_fillna_df, datasets=datasets)

        rng = np.random.default_rng(seed=seed)
        bootstrap_elo_lu = self.get_bootstrap_result(
            battles=battles,
            func_compute_elo=self.compute_mle_elo,
            num_round=None,
            rng=rng,
            func_kwargs={
                "INIT_RATING": INIT_RATING,
                "SCALE": SCALE,
                "calibration_framework": calibration_framework,
                "calibration_elo": calibration_elo,
            }
        )

        bars = pd.DataFrame(dict(
            rating=bootstrap_elo_lu.quantile(.5),
        )).sort_values("rating", ascending=False)

        elo_impact_by_dataset_list = []
        for dataset_to_skip in datasets:
            battles_w_dataset_removed = battles[battles[self.task_col] != dataset_to_skip]
            bootstrap_elo_lu_w_dataset_removed = self.get_bootstrap_result(
                battles=battles_w_dataset_removed,
                func_compute_elo=self.compute_mle_elo,
                num_round=None,
                rng=rng,
                func_kwargs={
                    "INIT_RATING": INIT_RATING,
                    "SCALE": SCALE,
                    "calibration_framework": calibration_framework,
                    "calibration_elo": calibration_elo,
                }
            )
            bars_by_dataset = pd.DataFrame(dict(
                rating=bootstrap_elo_lu_w_dataset_removed.quantile(.5),
            ))

            delta = bars["rating"] - bars_by_dataset["rating"]
            delta.name = dataset_to_skip
            elo_impact_by_dataset_list.append(delta)
        elo_impact_by_dataset = pd.concat(elo_impact_by_dataset_list, axis=1)
        return elo_impact_by_dataset

    def get_rank_confidence(self, df: pd.DataFrame):
        df = df.copy()
        df = df.sort_values(by=["Arena Elo"], ascending=False)

        elo_ratings = df["Arena Elo"].to_list()
        uppers = df["upper"].to_list()
        lowers = df["lower"].to_list()

        ranks = []

        cur_rank = 0
        prev_lower = None
        num_models = len(elo_ratings)
        for i in range(num_models):
            cur_elo = elo_ratings[i]
            cur_upper = uppers[i]
            cur_lower = lowers[i]
            if prev_lower is None or cur_upper < prev_lower:
                cur_rank = i + 1
                prev_lower = cur_lower
            ranks.append(cur_rank)

        df["Rank"] = ranks

        return df

    def predict_win_rate(self, elo_ratings: dict, SCALE=400, BASE=10):
        names = sorted(list(elo_ratings.keys()))
        wins = defaultdict(lambda: defaultdict(lambda: 0))
        for a in names:
            for b in names:
                ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
                wins[a][b] = ea
                wins[b][a] = 1 - ea

        data = {
            a: [wins[a][b] if a != b else np.NAN for b in names]
            for a in names
        }

        df = pd.DataFrame(data, index=names)
        df.index.name = "model_a"
        df.columns.name = "model_b"
        return df.T

    def compute_winrate_from_battles(self, battles) -> pd.Series:
        framework_battle_counts = defaultdict(int)
        framework_win_counts = defaultdict(float)
        for f, win_val in [(self.method_1, "1"), (self.method_2, "2")]:
            counts = battles[f].value_counts().to_dict()
            win_counts = battles[[f, "winner"]].value_counts().reset_index()
            win_counts["count"] = win_counts["count"].astype(float)
            win_counts.loc[win_counts["winner"] == "tie", "count"] *= 0.5
            win_counts = win_counts.loc[win_counts["winner"].isin([win_val, "tie"]), :]
            win_counts = win_counts.drop(columns=["winner"]).groupby(f)["count"].sum().to_dict()
            for framework in counts:
                framework_battle_counts[framework] += counts[framework]
            for framework in win_counts:
                framework_win_counts[framework] += win_counts[framework]
        framework_battle_counts = dict(framework_battle_counts)

        results_wins = pd.Series(framework_win_counts, name="wins")
        results_battles = pd.Series(framework_battle_counts, name="battles")
        results_winrate = results_wins / results_battles
        results_winrate.name = "winrate"
        results_winrate.index.name = self.method_col
        return results_winrate

    def _fix_missing(self, df, missing_A, missing_B):
        df = df.copy()
        for b in missing_B:
            if b not in df:
                df[b] = 0
        df = df.T
        for a in missing_A:
            if a not in df:
                df[a] = 0
        df = df.T
        return df

    def compute_pairwise_win_fraction(self, battles, max_num_models=30) -> pd.DataFrame:
        unique_A = list(battles[self.method_1].unique())
        unique_B = list(battles[self.method_2].unique())
        missing_A = [b for b in unique_B if b not in unique_A]
        missing_B = [a for a in unique_A if a not in unique_B]
        unique_all = unique_A + missing_A
        # Times each model wins as Model A
        a_win_ptbl = pd.pivot_table(
            battles[battles['winner'] == "1"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        # Table counting times each model wins as Model B
        b_win_ptbl = pd.pivot_table(
            battles[battles['winner'] == "2"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        # Table counting times each model wins as Model B
        tie_ptbl = pd.pivot_table(
            battles[battles['winner'] == "tie"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)
        tie_ptbl *= 0.5

        # Table counting number of A-B pairs
        num_battles_ptbl = pd.pivot_table(battles,
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        missing_A = unique_all
        missing_B = unique_all
        a_win_ptbl = self._fix_missing(df=a_win_ptbl, missing_A=missing_A, missing_B=missing_B)
        b_win_ptbl = self._fix_missing(df=b_win_ptbl, missing_A=missing_A, missing_B=missing_B)
        tie_missing_A = [a for a in unique_all if a not in tie_ptbl.index]
        tie_missing_B = [b for b in unique_all if b not in tie_ptbl.columns]
        tie_ptbl = self._fix_missing(df=tie_ptbl, missing_A=tie_missing_A, missing_B=tie_missing_B)
        num_battles_ptbl = self._fix_missing(df=num_battles_ptbl, missing_A=missing_A, missing_B=missing_B)

        # Computing the proportion of wins for each model as A and as B
        # against all other models
        row_beats_col_freq = (
            (a_win_ptbl + b_win_ptbl.T + tie_ptbl + tie_ptbl.T) /
            (num_battles_ptbl + num_battles_ptbl.T)
        )

        # Arrange ordering according to proprition of wins
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        prop_wins = prop_wins[:max_num_models]
        model_names = list(prop_wins.keys())
        row_beats_col = row_beats_col_freq.loc[model_names, model_names]
        return row_beats_col

    def get_arena_leaderboard(self, bootstrap_elo_lu: pd.DataFrame, results_df: pd.DataFrame):
        bars = pd.DataFrame(dict(
            lower=bootstrap_elo_lu.quantile(.025),
            rating=bootstrap_elo_lu.quantile(.5),
            upper=bootstrap_elo_lu.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
        bars['error_y'] = bars['upper'] - bars["rating"]
        bars['error_y_minus'] = bars['rating'] - bars["lower"]
        bars['rating_rounded'] = np.round(bars['rating'], 2)
        battles = self.convert_results_to_battles(results_df=results_df)
        from collections import defaultdict
        framework_battle_counts = defaultdict(int)
        framework_win_counts = defaultdict(float)
        for f, win_val in [(self.method_1, "1"), (self.method_2, "2")]:
            counts = battles[f].value_counts().to_dict()
            win_counts = battles[[f, "winner"]].value_counts().reset_index()
            win_counts["count"] = win_counts["count"].astype(float)
            win_counts.loc[win_counts["winner"] == "tie", "count"] *= 0.5
            win_counts = win_counts.loc[win_counts["winner"].isin([win_val, "tie"]), :]
            win_counts = win_counts.drop(columns=["winner"]).groupby(f)["count"].sum().to_dict()
            for framework in counts:
                framework_battle_counts[framework] += counts[framework]
            for framework in win_counts:
                framework_win_counts[framework] += win_counts[framework]
        framework_battle_counts = dict(framework_battle_counts)

        def _get_95_ci(upper, lower):
            return f"+{upper:.0f}/-{lower:.0f}"

        leaderboard = bars.copy()
        leaderboard["95% CI"] = [_get_95_ci(upper, lower) for upper, lower in zip(leaderboard["error_y"], leaderboard["error_y_minus"])]
        leaderboard["Arena Elo"] = np.round(leaderboard['rating'], 0).astype(int)
        leaderboard["Battles"] = leaderboard["model"].map(framework_battle_counts)
        leaderboard["Wins"] = np.round(leaderboard["model"].map(framework_win_counts), decimals=0).astype(int)
        leaderboard["Winrate"] = np.round(leaderboard["Wins"] / leaderboard["Battles"], decimals=2)
        leaderboard["Rank (Simple)"] = leaderboard["Arena Elo"].rank(method="min", ascending=False).astype(int)
        leaderboard["Model"] = leaderboard["model"]
        leaderboard = self.get_rank_confidence(df=leaderboard)

        results_mean_agg = results_df[[self.method_col, "rank", "bestdiff", "loss_rescaled"]].groupby("framework").mean()
        leaderboard["mean_rank"] = leaderboard["model"].map(results_mean_agg["rank"])
        leaderboard["mean_bestdiff"] = leaderboard["model"].map(results_mean_agg["bestdiff"])
        leaderboard["mean_loss_rescaled"] = leaderboard["model"].map(results_mean_agg["loss_rescaled"])

        leaderboard["Rank Avg"] = np.round(leaderboard["mean_rank"], decimals=1)
        leaderboard["Champ Delta %"] = np.round(leaderboard["mean_bestdiff"] * 100, decimals=1)
        # leaderboard["Champ Delta % 2"] = np.round((1/(1 - leaderboard["mean_bestdiff"]) - 1) * 100, decimals=1)
        leaderboard["Rescaled Acc"] = np.round(1 - leaderboard["mean_loss_rescaled"], decimals=2)
        leaderboard["Elo"] = leaderboard["Arena Elo"]

        leaderboard_print = leaderboard[[
            "Rank",
            "Model",
            "Elo",
            "95% CI",
            # "Battles",
            # "Wins",
            "Winrate",
            "Rescaled Acc",
            # "Rank Avg",
            "Champ Delta %",
            # "Champ Delta % 2",
        ]]

        return leaderboard, leaderboard_print
