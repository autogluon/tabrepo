from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.baseline_comparison.plot_utils import figure_path
from tabrepo import load_repository
from tabrepo.portfolio.similarity import distance_tasks_from_repo
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs


def vary_temperature(repo, distances, test_dataset):
    errors = []
    temperatures = [0.0, 0.5, 1, 2, 4, 8, 16, 32, 64]
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

    pd.DataFrame({"temperature": temperatures, "test-error": errors, "dataset": [test_dataset] * len(errors)}).to_csv(
        figure_path("similarity") / f"{test_dataset}-temperature.csv",
        index=False
    )


def vary_nclosest(repo, distances, test_dataset):
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
    pd.DataFrame({"nclosest": nclosests, "test-error": errors, "dataset": [test_dataset] * len(errors)}).to_csv(
        figure_path("similarity") / f"{test_dataset}-nclosest.csv",
        index=False
    )

def compute_portfolio_scores(repo, df_rank, test_dataset, test_fold) -> Dict[str, float]:
    train_tasks = [
        (dataset, fold)
        for dataset in repo.datasets() if dataset != test_dataset
        for fold in range(3)
    ]
    val_scores = - df_rank[train_tasks].values.T
    portfolio_indices = zeroshot_configs(val_scores=-val_scores, output_size=20)
    portfolio_configs = np.array(repo.configs())[portfolio_indices]
    model_metrics_new_task = {
        k: v for k, v in df_rank.loc[portfolio_configs, (test_dataset, test_fold)].to_dict().items()
    }
    return model_metrics_new_task

def plot():
    from pathlib import Path
    import pandas as pd
    path = Path(__file__).parent / "similarity/figures/"
    dfs = []
    for csv_file in Path(path).glob("*csv"):
        dfs.append(pd.read_csv(csv_file))
    df = pd.concat(dfs, ignore_index=True)
    df.head()
    df_temperatures = df[~df.temperature.isna()].dropna(axis=1)
    df_nclosest = df[~df.nclosest.isna()].dropna(axis=1)
    datasets = df_temperatures.dataset.unique()
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 2, sharex='col', sharey='row', figsize=(12, 3 * n_datasets))

    for i, dataset in enumerate(datasets):
        ax = df_temperatures[df_temperatures.dataset == dataset].plot(x="temperature", y="test-error", ax=axes[i][0],
                                                                      title=dataset)
        ax = df_nclosest[df_nclosest.dataset == dataset].plot(x="nclosest", y="test-error", ax=axes[i][1],
                                                              title=dataset)

    plt.tight_layout()
    plt.savefig(figure_path("similarity") / "similarity-analysis.pdf")
    plt.show()

    xx = df_nclosest.pivot_table(index="nclosest", columns="dataset", values="test-error")
    xx /= xx.values[-1, :]
    improvement = (100 * (1 - xx))
    ax = improvement.mean(axis=1).plot(lw=3.0, color="r", ls="--")
    improvement.plot(title="Improvement over portfolio (%)", ax=ax)
    plt.savefig(figure_path("similarity") / "avg-improvement-nclosest.pdf")
    plt.show()

    xx = df_temperatures.pivot_table(index="temperature", columns="dataset", values="test-error")
    xx /= xx.values[0, :]
    improvement = (100 * (1 - xx))
    ax = improvement.mean(axis=1).plot(lw=3.0, color="r", ls="--")
    improvement.plot(title="Improvement over portfolio (%)", ax=ax)
    plt.savefig(figure_path("similarity") / "avg-improvement-temperature.pdf")
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

    # First, we pick a task in our test task where catboost model performs well, this allows us to select a cluster
    # that resembles well a given task
    # framework = "KNeighbors"
    # framework = "LinearModel"
    dd = repo.metrics().pivot_table(index="framework", columns=["dataset"], values="metric_error").rank(pct=True)
    framework = "CatBoost"
    linear_models = [config for config in repo.configs() if framework in config]
    best_dataset_for_framework = dd.loc[linear_models].mean(axis=0).sort_values().index.tolist()
    print(f"Best tasks for {framework} models: {best_dataset_for_framework[:10]}")

    recompute = False
    if recompute:
        # Picks a task where model from frameworks works well, dont pick the first as may come from anomaly
        for i in range(10):
            test_dataset = best_dataset_for_framework[i]

            # Picks model evaluation from all configs of tabrepo of the dataset selected for the new task
            # model_metrics_new_task = {
            #     k: v for k, v in df_rank.loc[:, (test_dataset, fold)].to_dict().items()
            # }

            # Picks configurations evaluated on the portfolio of all tasks but the one being evaluated
            model_metrics_new_task = compute_portfolio_scores(
                # could compute average over all folds to smoothen the noise
                repo=repo, df_rank=df_rank, test_dataset=test_dataset, test_fold=0
            )

            distances = distance_tasks_from_repo(repo, model_metrics=model_metrics_new_task)

            vary_nclosest(repo, distances, test_dataset)
            vary_temperature(repo, distances, test_dataset)
    plot()


if __name__ == '__main__':
    main()
