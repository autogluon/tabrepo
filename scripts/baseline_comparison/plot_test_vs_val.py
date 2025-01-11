import ast
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import ArtistAnimation

from tabrepo.repository.abstract_repository import AbstractRepository


def plot_test_vs_val(df: pd.DataFrame, repo: AbstractRepository, baselines: list[str] = None):
    budget_suffix = f"\(4h\)"
    df_selected = df[
        df.method.str.contains(f"Portfolio-N.* \(ensemble\) {budget_suffix}")
    ].copy()

    df_expanded_metadata = pd.DataFrame(df_selected["metadata"].map(ast.literal_eval).to_list(), index=df_selected.index)
    df_selected = pd.concat([df_selected, df_expanded_metadata], axis=1)

    df_selected = df_selected.sort_values(by="n_portfolio")
    df_selected_first = df_selected[df_selected["n_portfolio"] == 1]
    df_selected_first["norm_factor"] = df_selected_first["test_error"]
    df_selected_first = df_selected_first[["dataset", "fold", "norm_factor"]]
    df_selected = df_selected.merge(df_selected_first, on=["dataset", "fold"])

    df_selected = df_selected[df_selected["norm_factor"] != 0]
    df_selected["metric_error_rescaled"] = df_selected["test_error"] / df_selected["norm_factor"]
    df_selected["metric_error_val_rescaled"] = df_selected["metric_error_val"] / df_selected["norm_factor"]

    plot_w(df=df_selected)

    task_metadata = repo.task_metadata.copy()
    task_metadata = task_metadata[task_metadata["dataset"].isin(df_selected["dataset"].unique())]
    task_metadata = task_metadata[["dataset", "NumberOfInstances"]].set_index("dataset")
    task_metadata = task_metadata.sort_values(by=["NumberOfInstances"])

    baselines = ["AutoGluon best (4h)", "Autosklearn2 (4h)"]
    if baselines is not None:
        df_baselines = df[df["method"].isin(baselines)].copy()
        df_baselines = df_baselines.merge(df_selected_first, on=["dataset", "fold"])
        df_baselines = df_baselines[df_baselines["norm_factor"] != 0]
        df_baselines["metric_error_rescaled"] = df_baselines["test_error"] / df_baselines["norm_factor"]
        # df_baselines["metric_error_val_rescaled"] = df_baselines["metric_error_val"] / df_baselines["norm_factor"]
    else:
        df_baselines = None

    sliding_window_size = 50

    n_datasets = len(task_metadata)

    fig, ax = plt.subplots(figsize=(8, 6))
    artists = []
    for i in range(n_datasets + sliding_window_size - 1):
    # for i in range(5):
        i_min = max(0, i - sliding_window_size + 1)
        i_max = min(n_datasets-1, i)

        cur_datasets = task_metadata.iloc[i_min:i_max+1]
        cur_datasets_names = cur_datasets.index
        print(i_min, i_max, n_datasets, i_max - i_min + 1, len(cur_datasets))
        samples_min = cur_datasets["NumberOfInstances"].min()
        samples_max = cur_datasets["NumberOfInstances"].max()
        text = f"Dataset #{i_min+1} - #{i_max+1} | # Rows: {samples_min} - {samples_max} | Window Size: {i_max - i_min + 1}"

        if df_baselines is not None:
            df_baselines_cur = df_baselines[df_baselines["dataset"].isin(cur_datasets_names)]
        else:
            df_baselines_cur = None

        df_selected_cur = df_selected[df_selected["dataset"].isin(cur_datasets_names)]
        if i == 0:
            update_ax = True
        else:
            update_ax = False

        artists_subplot = plot_w(df=df_selected_cur, name=f"my_fig_i{i}.png", ax=ax, update_ax=update_ax, text=text, df_baselines=df_baselines_cur)
        artists.append(artists_subplot)
    ani = ArtistAnimation(fig=fig, artists=artists, interval=200, blit=True)
    # ani.save('animation.html', writer='html')
    ani.save("animation.gif")

    from autogluon.common.utils.s3_utils import upload_file
    upload_file(file_name="animation.gif", bucket="autogluon-zeroshot")


def plot_w(df: pd.DataFrame, name: str = "test_vs_val.png", ax=None, update_ax: bool = False, text: str = None, df_baselines: pd.DataFrame = None):
    quantile_levels_og = [0.75]
    quantile_levels = [0.5] + quantile_levels_og + [1 - q for q in quantile_levels_og]
    minimal_data_quantile = df.groupby("n_portfolio")[["metric_error_rescaled", "metric_error_val_rescaled"]].quantile(q=quantile_levels)
    minimal_data = df.groupby("n_portfolio")[["metric_error_rescaled", "metric_error_val_rescaled"]].median().reset_index()

    results = minimal_data

    val_losses = results["metric_error_val_rescaled"]
    test_losses = results["metric_error_rescaled"]
    models = results["n_portfolio"]

    colors_baselines = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    save_fig = ax is None

    ymin = -200
    ymax = 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    artists = []
    for q in quantile_levels:
        results_q = minimal_data_quantile.loc[:, q, :]
        val_losses_q = results_q["metric_error_val_rescaled"]
        test_losses_q = results_q["metric_error_rescaled"]
        models = results_q.index

        alpha = math.pow((1 - abs(q - 0.5)), 5)
        if q != 0.5:
            line_val = ax.fill_betweenx([-m for m in models], val_losses_q, val_losses, alpha=alpha, color='#1f77b4')
            line_test = ax.fill_betweenx([-m for m in models], test_losses_q, test_losses, alpha=alpha, color='#ff7f0e')
            artists += [line_val, line_test]
        else:
            line_val, = ax.plot(val_losses_q, [-m for m in models], marker='o', linestyle="--", label=f"Val loss ({q})", alpha=alpha, color='#1f77b4')
            line_test, = ax.plot(test_losses_q, [-m for m in models], marker='o', linestyle="--", label=f"Test loss ({q})", alpha=alpha, color='#ff7f0e')
            artists += [line_val, line_test]

    if df_baselines is not None:
        minimal_data_quantile = df_baselines.groupby("method")[["metric_error_rescaled"]].quantile(q=quantile_levels)
        minimal_data = df_baselines.groupby("method")[["metric_error_rescaled"]].median()

        test_losses = minimal_data["metric_error_rescaled"]

        baselines = list(df_baselines["method"].unique())

        for q in quantile_levels:
            results_q = minimal_data_quantile.loc[:, q, :]
            test_losses_q = results_q["metric_error_rescaled"]
            alpha = math.pow((1 - abs(q - 0.5)), 5)
            for baseline, colors in zip(baselines, colors_baselines[:len(baselines)]):
                test_losses_q_baseline = test_losses_q.loc[baseline]
                if q != 0.5:
                    pass
                    # line_test = ax.fill_betweenx([ymin, ymax], test_losses_q_baseline, test_losses.loc[baseline], alpha=alpha*0.4, color=colors)
                    # artists += [line_test]
                else:
                    line_test = ax.vlines(x=test_losses_q_baseline, ymin=ymin, ymax=ymax, linestyles="solid", alpha=0.4, label=baseline, colors=colors, linewidth=1.5)
                    artists += [line_test]

    if text is not None:
        text = ax.text(0.5, 1.01, text, fontsize="large")
        artists.append(text)

    if update_ax:
        ax.legend()
        ax.set_xlabel("Relative Loss")
        ax.set_ylabel("Portfolio Size")
        ax.set_yticks([-m for m in models])
        ax.set_yticklabels(models)
        ax.grid(True)
        ax.set_xlim(0.5, 1.1)
        ax.set_ylim(ymin, ymax)

    if save_fig:
        plt.savefig(name)
    # plt.show()
    return artists
