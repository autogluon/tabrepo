import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tabrepo.repository.abstract_repository import AbstractRepository


def plot_test_vs_val(df: pd.DataFrame, repo: AbstractRepository):
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



    sliding_window_size = 10

    n_datasets = len(task_metadata)


    for i in range(n_datasets - sliding_window_size):
        cur_datasets = task_metadata.iloc[i:sliding_window_size+i]
        cur_datasets_names = cur_datasets.index
        samples_min = cur_datasets.iloc[0]
        samples_max = cur_datasets.iloc[-1]

        df_selected_cur = df_selected[df_selected["dataset"].isin(cur_datasets_names)]

        plot_w(df=df_selected_cur, name=f"my_fig_i{i}.png")

    # intermediate = {}
    # for dataset in repo.datasets():
    #     for fold in repo.folds:
    #         df_selected_task = df_selected[(df_selected["dataset"] == dataset) & (df_selected["fold"] == fold)]
    #         print(df_selected_task)
    #         df_selected_task = df_selected_task.sort_values(by="n_portfolio")
    #
    #         minimal_data = df_selected_task[["n_portfolio", "metric_error_val", "test_error"]]
    #         minimal_data = list(minimal_data.values)
    #
    #         results = minimal_data
    #
    #         val_losses = [val_loss for _, val_loss, _ in results]
    #         test_losses = [test_loss for _, _, test_loss in results]
    #         models = [model for model, _, _ in results]
    #
    #         norm_factor = test_losses[0]
    #
    #         if norm_factor == 0:
    #             print(f"norm_factor==0 for {dataset} {fold}, skipping!")
    #             continue
    #         test_losses = [t/norm_factor for t in test_losses]
    #         val_losses = [v/norm_factor for v in val_losses]
    #
    #         intermediate[(dataset, fold)] = (models, test_losses, val_losses)
    #
    #         fig, ax = plt.subplots(figsize=(12, 9))
    #         ax.plot(val_losses, [-m for m in models], marker='o', linestyle="--", label="Validation loss")
    #         ax.plot(test_losses, [-m for m in models], marker='o', linestyle="--", label="Test loss")
    #         ax.legend()
    #         ax.set_xlabel("Relative loss")
    #         ax.set_yticks([-m for m in models])
    #         ax.set_yticklabels(models)
    #
    #         plt.savefig("my_fig.png")
    #         plt.show()


def plot_w(df, name="my_fig3.png"):
    quantile_levels_og = [0.75]
    quantile_levels = [0.5] + quantile_levels_og + [1 - q for q in quantile_levels_og]
    minimal_data_quantile = df.groupby("n_portfolio")[["metric_error_rescaled", "metric_error_val_rescaled"]].quantile(q=quantile_levels)
    minimal_data = df.groupby("n_portfolio")[["metric_error_rescaled", "metric_error_val_rescaled"]].median().reset_index()

    results = minimal_data

    val_losses = results["metric_error_val_rescaled"]
    test_losses = results["metric_error_rescaled"]
    models = results["n_portfolio"]

    fig, ax = plt.subplots(figsize=(12, 9))

    for q in quantile_levels:
        results_q = minimal_data_quantile.loc[:, q, :]
        val_losses_q = results_q["metric_error_val_rescaled"]
        test_losses_q = results_q["metric_error_rescaled"]
        models = results_q.index

        import math
        alpha = math.pow((1 - abs(q - 0.5)), 5)
        if q != 0.5:
            ax.fill_betweenx([-m for m in models], val_losses_q, val_losses, alpha=alpha, color='#1f77b4')
            ax.fill_betweenx([-m for m in models], test_losses_q, test_losses, alpha=alpha, color='#ff7f0e')
        else:
            ax.plot(val_losses_q, [-m for m in models], marker='o', linestyle="--", label=f"Validation loss ({q})", alpha=alpha, color='#1f77b4')
            ax.plot(test_losses_q, [-m for m in models], marker='o', linestyle="--", label=f"Test loss ({q})", alpha=alpha, color='#ff7f0e')

    ax.legend()
    ax.set_xlabel("Relative loss")
    ax.set_yticks([-m for m in models])
    ax.set_yticklabels(models)
    ax.grid(True)

    plt.savefig(name)
    # plt.show()
