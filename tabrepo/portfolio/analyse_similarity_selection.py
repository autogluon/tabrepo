import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tabrepo import load_repository
from tabrepo.portfolio.similarity import distance_tasks_from_repo
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs


def vary_and_plot_temperature(repo, distances, test_dataset):
    errors = []
    temperatures = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16]
    for temperature in temperatures:
        weights = np.exp(temperature * -np.array(list(distances.values())))
        # weights = 1 + (1 + temperature * np.array(list(distances.values())))
        # if temperature != 0:
        #     weights = (weights - weights.min()) / (weights.max() - weights.min())
        dd = repo._zeroshot_context.df_configs_ranked
        # TODO use normalized scores
        df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
        df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
        # keep only tasks different than the current one to evaluate
        train_tasks = []
        for task in df_rank.columns:
            t, fold = task.split("_")
            if t != repo.dataset_to_tid(test_dataset):
                train_tasks.append(task)
        val_scores = - df_rank[train_tasks].values.T
        portfolio_indices = zeroshot_configs(val_scores=-val_scores, output_size=20, weights=weights)
        portfolio_configs = np.array(repo.configs())[portfolio_indices]

        # print("**Computing portfolio with weights**")
        # print(f"Portfolio indices: {portfolio_indices}")
        # print(f"Portfolio configs: {portfolio_configs}")

        test_errors, _ = repo.evaluate_ensemble(datasets=[test_dataset], configs=portfolio_configs)
        print(temperature, test_errors.values)
        errors.append(test_errors.mean())
    plt.plot(temperatures, errors, marker="o")
    # plt.xscale("log")
    plt.xlabel("temperature")
    plt.ylabel(f"Test error on dataset {test_dataset}")
    plt.tight_layout()
    plt.savefig(f"temperature-{test_dataset}.pdf")
    plt.show()


def vary_and_plot_nclosest(repo, distances, test_dataset):
    errors = []
    nclosests = [5, 10, 20, 40, 100, 200, 400, 600]

    df_distance = pd.DataFrame([
        (dataset, fold, value)
        for (dataset, fold), value in distances.items() if dataset != test_dataset
    ], columns=["dataset", "fold", "metric_error"])
    closest_datasets = df_distance.groupby("dataset").mean()["metric_error"].sort_values()
    selected_tids = [
        repo.task_name_from_tid(repo.dataset_to_tid(dataset), fold)
        for dataset in closest_datasets.index
        for fold in range(3)
    ]
    for nclosest in nclosests:
        # weights = 1 + (1 + temperature * np.array(list(distances.values())))
        # if temperature != 0:
        #     weights = (weights - weights.min()) / (weights.max() - weights.min())
        dd = repo._zeroshot_context.df_configs_ranked
        # TODO use normalized scores
        df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
        df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
        train_tasks = selected_tids[:nclosest]
        val_scores = - df_rank[train_tasks].values.T
        portfolio_indices = zeroshot_configs(val_scores=-val_scores, output_size=20)
        portfolio_configs = np.array(repo.configs())[portfolio_indices]

        # print("**Computing portfolio with weights**")
        # print(f"Portfolio indices: {portfolio_weighted_indices}")
        # print(f"Portfolio configs: {portfolio_weighted_configs}")

        test_errors, _ = repo.evaluate_ensemble(datasets=[test_dataset], configs=portfolio_configs)
        print(nclosest, test_errors.values, test_errors.values.mean())
        errors.append(test_errors.mean())
    plt.plot(nclosests, errors, marker="o")
    # plt.xscale("log")
    plt.xlabel("Number of closest tasks")
    plt.ylabel(f"Test error on dataset {test_dataset}")
    plt.tight_layout()
    plt.savefig(f"nclosest-{test_dataset}.pdf")
    plt.show()


def main():
    """
    This script first selects a task where a framework performs well.
    Then it studies the performance of portfolio selection when filtering task closest in performance either by applying
    a hard filtering or a soft filtering controlled by temperature.
    :return:
    """
    repo = load_repository("D244_F3_C1530_200")

    df_rank = repo.metrics().pivot_table(index="framework", columns=["dataset", "fold"], values="metric_error").rank(
        pct=True)

    # framework = "KNeighbors"
    # framework = "LinearModel"
    framework = "CatBoost"
    linear_models = [config for config in repo.configs() if framework in config]
    tasks_best_for_framework = df_rank.loc[linear_models].mean(axis=0).sort_values().index.tolist()
    print(f"Best tasks for {framework} models: {tasks_best_for_framework[:10]}")

    # Picks a task where model from frameworks works well, dont pick the first as may come from anomaly
    for i in [3, 5, 7]:
        test_dataset, fold = tasks_best_for_framework[3*i]

        # Picks evaluation of the dataset selected for the new task
        # TODO pick instead portfolio configurations
        # model_metrics_new_task = {
        #     k: v for k, v in df_rank.loc[:, (test_dataset, fold)].to_dict().items()
        # }

        train_tasks = [
            (dataset, fold)
            for dataset in repo.datasets() if dataset != test_dataset
            for fold in range(3)
        ]
        val_scores = - df_rank[train_tasks].values.T
        portfolio_indices = zeroshot_configs(val_scores=-val_scores, output_size=20)
        portfolio_configs = np.array(repo.configs())[portfolio_indices]
        model_metrics_new_task = {
            k: v for k, v in df_rank.loc[portfolio_configs, (test_dataset, fold)].to_dict().items()
        }

        distances = distance_tasks_from_repo(repo, model_metrics=model_metrics_new_task)

        vary_and_plot_nclosest(repo, distances, test_dataset)
        vary_and_plot_temperature(repo, distances, test_dataset)


if __name__ == '__main__':
    main()
