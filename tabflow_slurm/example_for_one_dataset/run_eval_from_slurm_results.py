# TODO: refactor and update the plotting code to use the new evaluation framework


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from tabarena import EvaluationRepository
from tabarena.benchmark.task import UserTask
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
from tabarena.nips2025_utils.generate_repo import generate_repo
from tabarena.paper.paper_runner_tabarena import PaperRunTabArena
from tabarena.paper.paper_utils import get_framework_type_method_names


def run_example_for_evaluate_results_on_custom_dataset(
    *, eval_dir: Path, repo_dir: Path, tabarena_dir: Path, user_task_str: str
) -> None:
    """Example for evaluating the cached results with similar plots to the TabArena paper."""
    clf_task = UserTask.from_task_id_str(user_task_str)
    task_metadata = load_task_metadata(paper=True)
    task_metadata = pd.DataFrame(columns=task_metadata.columns)
    task_metadata["tid"] = [clf_task.task_id]
    task_metadata["name"] = [clf_task.tabarena_task_name]
    task_metadata["task_type"] = ["Supervised Classification"]
    task_metadata["dataset"] = [
        clf_task.tabarena_task_name,
    ]
    task_metadata["NumberOfInstances"] = [2466]
    repo: EvaluationRepository = generate_repo(experiment_path=tabarena_dir, task_metadata=task_metadata)
    repo.to_dir(str(repo_dir))
    repo: EvaluationRepository = EvaluationRepository.from_dir(str(repo_dir))
    repo.set_config_fallback(repo.configs()[0])
    plotter = PaperRunTabArena(repo=repo, output_dir=str(eval_dir), backend="native")
    df_results = plotter.run_no_sim()
    is_default = df_results["framework"].str.contains("_c1_") & (df_results["method_type"] == "config")
    df_results.loc[is_default, "framework"] = df_results.loc[is_default]["config_type"].apply(
        lambda c: f"{c} (default)"
    )
    list(df_results["config_type"].unique())
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(df_results=df_results)
    df_results.to_csv(eval_dir / "results.csv")


def run_tuning_impact_plot(*, eval_dir: Path):
    df = pd.read_csv(eval_dir / "results.csv", index_col=0)
    df["roc_auc"] = 1 - df["metric_error"]
    df["framework"] = df["framework"].str.replace("TA-", "")
    df["config_type"] = df["config_type"].str.replace("TA-", "")

    subset_df = df[df["framework"].str.contains(r"\(")]
    numbers_df = subset_df.groupby(["dataset", "framework"])["roc_auc"].agg(["mean", "std"])
    print(numbers_df)
    numbers_df.to_csv(eval_dir / "tuning_impact_roc_auc_numbers.csv")

    framework_types = list(df["config_type"].unique())
    df = df.rename(columns={"framework": "method"})

    use_lim = True
    use_y = False
    lim = [0.65, 0.83]
    xlim = None
    ylim = None
    imputed_names = []
    framework_col = "framework_type"
    lower_is_better = False
    use_score = True
    same_width = False

    # Set ROC AUC
    df["roc_auc"] = 1 - df["metric_error"]
    df["roc_auc_val"] = 1 - df["metric_error_val"]
    metric = "roc_auc"
    metric_name = "ROC AUC"

    # Aggregate over folds and datasets
    groupby_columns_extra = ["dataset", "fold"]

    f_map, f_map_type, f_map_inverse, f_map_type_name = get_framework_type_method_names(
        framework_types=framework_types,
        max_runtimes=[
            (3600 * 4, "_4h"),
            (None, None),
        ],
    )

    df["framework_type"] = df["method"].map(f_map_type).fillna(df["method"])
    df["tune_method"] = df["method"].map(f_map_inverse).fillna("default")
    df["framework_type"] = df["framework_type"].map(f_map_type_name).fillna(df["framework_type"])
    framework_types = [f_map_type_name.get(ft, ft) for ft in framework_types]
    df_plot = df[df["framework_type"].isin(framework_types)]
    df_plot_w_mean_per_dataset = (
        df_plot.groupby(["framework_type", "tune_method", *groupby_columns_extra])[metric].mean().reset_index()
    )
    df_plot_w_mean_2 = (
        df_plot_w_mean_per_dataset.groupby(["framework_type", "tune_method"])[metric].mean().reset_index()
    )
    df_plot_w_mean_2 = df_plot_w_mean_2.sort_values(by=metric, ascending=lower_is_better)
    df_plot_mean_dedupe = df_plot_w_mean_2.drop_duplicates(subset=["framework_type"], keep="first")
    framework_type_order = list(df_plot_mean_dedupe["framework_type"].to_list())
    framework_type_order.reverse()

    with sns.axes_style("whitegrid"):
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

        linewidth = 0.0 if use_y else 0.3
        err_linewidth = 1.6
        err_linewidths = {
            "tuned_ensembled": err_linewidth,
            "tuned": err_linewidth * 0.8,
            "default": err_linewidth * 0.6,
        }
        err_alpha = 0.6

        to_plot = [
            {
                "x": pos,
                "y": y,
                "label": "Tuned + Ensembled",
                "data": df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned_ensembled"],
                "ax": ax,
                "order": framework_type_order,
                "color": colors[2],
                "width": 0.6,
                "linewidth": linewidth,
                "err_kws": {
                    "color": errcolors[2],
                    "linewidth": err_linewidths["tuned_ensembled"],
                    "alpha": err_alpha,
                },
            },
            {
                "x": pos,
                "y": y,
                # hue="tune_method",  # palette=["m", "g", "r],
                "label": "Tuned",
                "data": df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "tuned"],
                "ax": ax,
                "order": framework_type_order,
                "color": colors[1],
                "width": 0.5,
                "linewidth": linewidth,
                "err_kws": {
                    "color": errcolors[1],
                    "linewidth": err_linewidths["tuned"],
                    "alpha": err_alpha,
                },
            },
            {
                "x": pos,
                "y": y,
                # hue="tune_method",  # palette=["m", "g", "r],
                "label": "Default",
                "data": df_plot_w_mean_per_dataset[df_plot_w_mean_per_dataset["tune_method"] == "default"],
                "ax": ax,
                "order": framework_type_order,
                "color": colors[0],
                "width": 0.4,
                "linewidth": linewidth,
                "err_kws": {
                    "color": errcolors[0],
                    "linewidth": err_linewidths["default"],
                    "alpha": err_alpha,
                },
                "alpha": 1.0,
            },
        ]

        if use_score:
            widths = [plot_line["width"] for plot_line in to_plot]
            colors = [plot_line["color"] for plot_line in to_plot]
            err_kws_lst = [plot_line["err_kws"] for plot_line in to_plot]

            for plot_line, width, _color, _err_kws in zip(to_plot, widths, colors, err_kws_lst):
                if same_width:
                    plot_line["width"] = 0.6 * 1.3
                else:
                    plot_line["width"] = width * 1.3

        for plot_line in to_plot:
            boxplot = sns.barplot(**plot_line)

        if use_y:
            boxplot.set(xlabel="Elo" if metric == "elo" else metric_name, ylabel=None)
        else:
            boxplot.set(xlabel=None, ylabel="Elo" if metric == "elo" else metric_name)  # remove "Method" in the x-axis

        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        ticks = boxplot.get_yticks() if use_y else boxplot.get_xticks()
        ticklabels = [tick.get_text() for tick in (boxplot.get_yticklabels() if use_y else boxplot.get_xticklabels())]

        # ----- highlight bars that contain imputed results -----

        # Map x-tick positions to category labels
        label_lookup = dict(zip(ticks, ticklabels))
        has_imputed = False
        for _i, bar in enumerate(boxplot.patches):
            # Get x-position and convert to category label
            # todo: this only works for vertical barplots
            pos = bar.get_y() + bar.get_height() / 2 if use_y else bar.get_x() + bar.get_width() / 2
            category_index = round(pos)  # x-ticks are usually 0, 1, 2, ...
            category = label_lookup.get(category_index)
            if category in imputed_names:
                has_imputed = True
                bar.set_hatch("xx")

        if not use_y:
            # ----- alternate rows of x tick labels -----
            # Get current x tick labels
            labels = [label.get_text() for label in boxplot.get_xticklabels()]

            # Add newline to every second label
            new_labels = [label if i % 2 == 0 else r"$\uparrow$" + "\n" + label for i, label in enumerate(labels)]

            # Apply modified labels
            boxplot.set_xticklabels(new_labels)

        # remove unnecessary extra space on the sides
        if use_y:
            plt.ylim(len(boxplot.get_yticklabels()) - 0.35, -0.65)
        else:
            plt.xlim(-0.5, len(boxplot.get_xticklabels()) - 0.5)

        ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.02])
        handles, labels = ax.get_legend_handles_labels()
        if has_imputed:
            # Create a custom legend patch for "imputed"
            imputed_patch = Patch(
                facecolor="gray",
                edgecolor="white",
                hatch="xx",
                label="Partially imputed",
            )

            # Add to existing legend
            handles.append(imputed_patch)
            labels.append("Partially imputed")
        order = list(range(len(labels)))
        order = list(reversed(order))
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="lower center",
            ncol=(len(labels) + 1) // 2 if has_imputed and use_y else len(labels),
            bbox_to_anchor=[0.35 if use_y else 0.5, 1.05],
        )
        plt.tight_layout()

        plt.savefig(eval_dir / "tuning_impact_plot.pdf")
        plt.show()


if __name__ == "__main__":
    BASE_EVAL_DIR = Path(__file__).parent / "tabarena_out" / "slurm_results"
    BASE_REPO_DIR = Path(__file__).parent / "tabarena_out" / "repos"
    BASE_TABARENA_DIR = Path("/work/dlclarge2/purucker-tabarena/output")

    for benchmark_name, user_task_str in [
        (
            "biopsie_preprocessed_full_cohort",
            "UserTask|5928299900|BiopsyCancerPrediction_biopsie_preprocessed_full_cohort.csv|/work/dlclarge2/purucker-tabarena/code/tabarena_benchmarking_examples/tabarena_applications/biopsy_predictions/tabarena_out/local_tasks",
        ),
    ]:
        eval_dir = BASE_EVAL_DIR / benchmark_name
        repo_dir = BASE_REPO_DIR / benchmark_name
        tabarena_dir = BASE_TABARENA_DIR / benchmark_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_example_for_evaluate_results_on_custom_dataset(
            eval_dir=eval_dir,
            repo_dir=repo_dir,
            tabarena_dir=tabarena_dir,
            user_task_str=user_task_str,
        )
        run_tuning_impact_plot(eval_dir=eval_dir)
