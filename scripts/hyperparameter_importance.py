"""
Computes hyperparameter importance, requires optuna and syne-tune installed

"""
import json
from pathlib import Path
from typing import NamedTuple, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from optuna.importance._fanova._fanova import _Fanova
from syne_tune.config_space import randint, uniform, loguniform, choice, lograndint
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges

from scripts.baseline_comparison.plot_utils import figure_path
from tabrepo import load_repository
from tabrepo.utils.parallel_for import parallel_for


class FrameworkInfo(NamedTuple):
    name: str  # name of the framework used in configs.parquet
    config_space: Dict  # used to map values to ndarray to compute f-anova scores


framework_infos = {
    "CatBoost": FrameworkInfo(
        name="CatBoost",
        config_space={
            "depth": randint(4, 8),
            "grow_policy": choice(["Depthwise", "SymmetricTree"]),
            "l2_leaf_reg": uniform(1, 5),
            "learning_rate": loguniform(5e-3, 0.1),
            "one_hot_max_size": choice([2, 3, 5, 10]),
            "max_ctr_complexity": randint(1, 5),
        }
    ),
    "MLP": FrameworkInfo(
        name="NeuralNetTorch",
        config_space={
            "learning_rate": loguniform(1e-4, 3e-2),
            'weight_decay': loguniform(1e-12, 0.1),
            'dropout_prob': uniform(0.0, 0.4),
            'use_batchnorm': choice([False, True]),
            'num_layers': randint(1, 5),
            'hidden_size': randint(8, 256),
            'activation': choice(['relu', 'elu']),
        }
    ),
    "LightGBM": FrameworkInfo(
        name="LightGBM",
        config_space={
            'learning_rate': loguniform(lower=5e-3, upper=0.1, ),
            'feature_fraction': uniform(lower=0.4, upper=1.0),
            'min_data_in_leaf': randint(lower=2, upper=60),
            'num_leaves': randint(lower=16, upper=255),
            'extra_trees': choice([False, True]),
        }
    ),
    "XGBoost": FrameworkInfo(
        name="XGBoost",
        config_space={
            'learning_rate': loguniform(lower=5e-3, upper=0.1),
            'max_depth': randint(lower=4, upper=10),
            'min_child_weight': uniform(0.5, 1.5),
            'colsample_bytree': uniform(0.5, 1.0),
            'enable_categorical': choice([True, False]),
        }
    ),
    "LinearModel": FrameworkInfo(
        name="LinearModel",
        config_space={
            "C": uniform(lower=0.1, upper=1e3),
            "proc.skew_threshold": choice([0.99, 0.9, 0.999, None]),
            "proc.impute_strategy": choice(["median", "mean"]),
            "penalty": choice(["L2", "L1"]),
        }
    ),
    "KNeighbors": FrameworkInfo(
        name="KNeighbors",
        config_space={
            'n_neighbors': lograndint(3, 50),
            'weights': choice(['uniform', 'distance']),
            'p': choice([2, 1]),
        }
    ),
    "RandomForest": FrameworkInfo(
        name="RandomForest",
        config_space={
            'max_leaf_nodes': randint(5000, 50000),
            'min_samples_leaf': lograndint(1, 80),
            'max_features': choice(['sqrt', 'log2', 0.5, 0.75, 1.0])
        }
    ),
    "ExtraTrees": FrameworkInfo(
        name="ExtraTrees",
        config_space={
            'max_leaf_nodes': randint(5000, 50000),
            'min_samples_leaf': lograndint(1, 80),
            'max_features': choice(['sqrt', 'log2', 0.5, 0.75, 1.0])
        }
    )
}


def filter_framework(repo, framework_info: FrameworkInfo) -> Tuple[pd.DataFrame, dict]:
    # returns only data containing the provided framework
    framework_config_dict = {
        k: v
        for k, v in repo._zeroshot_context.configs_hyperparameters.items()
        if framework_info.name in k
    }
    df_configs = repo._zeroshot_context.df_configs
    df_configs = df_configs[df_configs.framework.str.contains(f"{framework_info.name}_r")]
    return df_configs, framework_config_dict


def _evaluate_task(
        config_space: dict,
        dict_configs: dict,
        df_scores: pd.DataFrame,
        config_names: List[str],
        task: str
) -> Optional[Dict[str, float]]:
    """
    Evaluates feature importance of a given task from tabrepo
    :param config_space:
    :param dict_configs:
    :param df_scores:
    :param config_names:
    :param task:
    :return:
    """
    hp_ranges = make_hyperparameter_ranges(config_space=config_space)
    X = hp_ranges.to_ndarray_matrix(dict_configs)

    # print(f"encoded feature shape for config space: {X.shape}")

    # compute mapping to columns, categorical corresponds to multiple columns of the X matrix
    column_to_encoded_columns = []
    current = 0
    for (hp_name, hp) in config_space.items():
        size = len(hp.categories) if hasattr(hp, "categories") else 1
        column_to_encoded_columns.append(np.arange(current, current + size))
        current += size

    y_series = df_scores.loc[task]
    y = y_series.loc[config_names].values

    # Defaults from optuna
    fanova = _Fanova(
        n_trees=64,
        max_depth=64,
        min_samples_split=2,
        min_samples_leaf=1,
        seed=0,
    )
    is_valid = ~np.isnan(y)  # remove potential nans values
    try:
        fanova.fit(
            X=X[is_valid],
            y=y[is_valid],
            search_spaces=np.array(hp_ranges.get_ndarray_bounds()),
            column_to_encoded_columns=column_to_encoded_columns,
        )
        row = {hp: fanova.get_importance(i)[0] for i, hp in enumerate(config_space.keys())}
        row["task"] = task
        return row
    except RuntimeError as e:
        print(f"An error occured while estimating feature importance: {str(e)}")
        # some tasks throws an error "Encountered zero total variance in all trees."
        return None


def generate_importances(frameworks: List[str], version: str):
    repo = load_repository(version=version)

    framework_importances = {}
    # compute f-anova scores for all frameworks and tasks
    for framework in frameworks:
        framework_info = framework_infos[framework]
        config_space = framework_info.config_space

        # First converts hyperparameters to list of dict to convert later to feature matrix with syne tune
        df_configs, config_dict = filter_framework(repo, framework_info)
        assert len(config_dict) > 0
        config_names = []
        dict_configs = []
        for name, config in config_dict.items():
            if "_c" not in name:
                config['hyperparameters'].pop("ag_args", None)
                dict_configs.append(config['hyperparameters'])
                config_names.append(name + "_BAG_L1")

        # Gets the evaluations for all the tasks
        assert len(df_configs) > 0
        df_scores = df_configs.pivot_table(index="dataset", columns="framework", values="metric_error")

        # Remove tasks that are missing, for instance because of failures
        tasks = [dataset for dataset in repo.datasets() if dataset in df_scores.index]
        if num_tasks is not None:
            tasks = tasks[:num_tasks]

        # Iterates over all tasks to compute f-anova scores
        context = {
            "config_space": config_space,
            "df_scores": df_scores,
            "config_names": config_names,
            "dict_configs": dict_configs,
        }
        rows = parallel_for(
            _evaluate_task, [{"task": task} for task in tasks], context=context
        )
        rows = [row for row in rows if row is not None]
        df_importance = pd.DataFrame(rows).set_index("task")
        print(framework, df_importance.mean().sort_values(ascending=False))
        framework_importances[framework] = df_importance.mean().sort_values(ascending=False).to_dict()

    with open(results_file, "w") as f:
        f.write(json.dumps(framework_importances))


def plot_importances(log):
    with open(results_file, "r") as f:
        dict_importances = json.load(f)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(frameworks), figsize=(14, 2), sharey=True)
    axes = np.ravel(axes)
    for i, (framework, importances) in enumerate(dict_importances.items()):
        pd.Series(importances).plot(kind="bar", title=framework, ax=axes[i], fontsize=9, logy=log)
        axes[i].title.set_size(11)
        axes[i].grid()
    fig.suptitle("Hyperparameter importance of all frameworks", y=1.15, fontsize=13)
    plt.savefig(figure_path() / "hyperparameter-importance.pdf", bbox_inches='tight')


def main():
    generate_importances(frameworks, version)
    plot_importances(log=True)


if __name__ == '__main__':
    version = "D244_F3_C1530_200"  # tabrepo context to use
    num_tasks = 200
    frameworks = [
        "KNeighbors",
        "LinearModel",
        "RandomForest",
        "ExtraTrees",
        "MLP",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
    results_file = Path(__file__).parent / "importances.json"
    main()
