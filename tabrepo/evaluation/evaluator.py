from __future__ import annotations

from pathlib import Path

import pandas as pd

from autogluon.common.savers import save_pd

from ..repository import EvaluationRepository
from ..repository.repo_utils import convert_time_infer_s_from_sample_to_batch


# TODO: This class is WIP.
# TODO: Add unit tests
class Evaluator:
    """
    Computes metrics and statistics to compare methods.
    """
    def __init__(
        self,
        repo: EvaluationRepository,
    ):
        self.repo = repo

    # TODO: repo time_infer_s is per row, results_df is the total time for all rows, need to align later
    # TODO: Error if unknown configs/baselines requested
    # TODO: Add fillna
    # TODO: Docstring
    # Q:Whether to keep these functions a part of TabRepo or keep them separate as a part of new fit()-package
    def compare_metrics(
        self,
        results_df: pd.DataFrame = None,
        datasets: list[str] = None,
        folds: list[int] = None,
        configs: list[str] = None,
        baselines: list[str] = None,
    ) -> pd.DataFrame:
        if datasets is None:
            datasets = self.repo.datasets()
        columns = ["metric_error", "time_train_s", "time_infer_s", "metric", "problem_type"]

        if results_df is not None:
            df_exp = results_df.reset_index().set_index(["dataset", "fold", "framework"])[columns]
        else:
            df_exp = None

        # Dropping task column in df_tr
        df_tr = self.repo._zeroshot_context.df_configs.set_index(["dataset", "fold", "framework"])[columns]

        mask = df_tr.index.get_level_values("dataset").isin(datasets)
        if folds is not None:
            mask = mask & df_tr.index.get_level_values("fold").isin(folds)
        if configs is not None:
            mask = mask & df_tr.index.get_level_values("framework").isin(configs)
        df_tr = df_tr[mask]

        if self.repo.task_metadata is not None:
            df_tr = convert_time_infer_s_from_sample_to_batch(df_tr, repo=self.repo)

        if self.repo._zeroshot_context.df_baselines is not None:
            df_baselines = self.repo._zeroshot_context.df_baselines.set_index(["dataset", "fold", "framework"])[columns]

            mask = df_baselines.index.get_level_values("dataset").isin(datasets)
            if folds is not None:
                mask = mask & df_baselines.index.get_level_values("fold").isin(folds)
            if baselines is not None:
                mask = mask & df_baselines.index.get_level_values("framework").isin(baselines)
            df_baselines = df_baselines[mask]

            if self.repo.task_metadata is not None:
                df_baselines = convert_time_infer_s_from_sample_to_batch(df_baselines, repo=self.repo)
        else:
            if baselines:
                raise AssertionError(f"Baselines specified but no baseline methods exist! (baselines={baselines})")
            df_baselines = None

        df = pd.concat([df_exp, df_tr, df_baselines], axis=0)
        df = df.sort_index()

        return df

    # TODO: Rename to something better?
    def plot_overall_rank_comparison(
        self,
        results_df: pd.DataFrame,
        save_dir: str,
        evaluator_kwargs: dict = None,
        calibration_framework: str = None,
    ) -> "EvaluatorOutput":
        """
        Requires `autogluon_benchmark` to be installed.

        Parameters
        ----------
        results_df: pd.DataFrame
            The input data to calculate metrics with.
            An easy way to obtain a valid `results_df` is to call `evaluator.compare_metrics(...)`
            It should have a multi-index of (dataset, fold, framework), with the following columns:
                metric_error: float
                metric: str
                time_train_s: float
                time_infer_s: float
                problem_type: str
        save_dir: str
            The local directory to save comparison results and figures to.
        evaluator_kwargs: dict, default = None
            The evaluator kwargs.
        calibration_framework: str, default = None
            The framework to fix at 1000 elo.

        Returns
        -------
        EvaluatorOutput object from autogluon_benchmark
        """
        try:
            from autogluon_benchmark.evaluation.evaluator import Evaluator as Eval
            from autogluon_benchmark.plotting.plotter import Plotter
        except ImportError:
            raise ImportError(f"To use `Evaluator.plot_overall_rank_comparison, you must first install autogluon_benchmark.")
        if evaluator_kwargs is None:
            evaluator_kwargs = {}
        results_df = results_df.reset_index().copy()
        results_df["tid"] = results_df["dataset"].apply(self.repo.dataset_to_tid)
        evaluator = Eval(task_metadata=self.repo.task_metadata, **evaluator_kwargs)
        evaluator_output = evaluator.transform(results_df)
        output_path = Path(save_dir)
        figure_savedir = str(output_path / "figures")
        save_pd.save(path=str(output_path / "results.csv"), df=results_df)
        save_pd.save(path=str(output_path / "results_ranked_agg.csv"), df=evaluator_output.results_ranked_agg)
        save_pd.save(path=str(output_path / "results_ranked.csv"), df=evaluator_output.results_ranked)

        plotter = Plotter(
            results_ranked_fillna_df=evaluator_output.results_ranked,
            results_ranked_df=evaluator_output.results_ranked,
            save_dir=figure_savedir,
            show=False,
        )

        plotter.plot_all(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=100,  # Reduce this to lower values for a faster execution. Use 1000 for the final plot.
            plot_critical_difference=False,
        )

        return evaluator_output
