from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from autogluon_zeroshot.repository.evaluation_repository import load
from autogluon_zeroshot.utils.cache import cache_function


def order_clustermap(df):
    # TODO we could just call scipy
    cg = sns.clustermap(df)
    row_indices = cg.dendrogram_row.reordered_ind
    col_indices = cg.dendrogram_col.reordered_ind
    plt.close()
    return df.index[row_indices], df.columns[col_indices]


def index(name):
    config_number = name.split("-")[-1]
    if "c" in config_number:
        return None
    else:
        return int(config_number)


num_models_to_plot = 20
title_size = 20
figsize = (20, 7)

repo = cache_function(lambda: load(version="BAG_D244_F10_C608_FULL"), cache_name="repo")
zsc = repo._zeroshot_context

# remove tasks with some lightGBM models missing, todo fix
df = zsc.df_results_by_dataset_vs_automl.copy()
missing_tids = [359932, 359944, 359933, 359946]
df = df[~df.tid.isin(missing_tids)]

metric = "metric_error"
df_pivot = df.pivot_table(
    index="framework", columns="tid", values=metric
)
df_rank = df_pivot.rank() / len(df_pivot)
df_rank.index = [x.replace("NeuralNetFastAI", "MLP").replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]
# shorten framework names
#df_rank.index = [x.replace("ExtraTrees", "ET").replace("CatBoost", "CB").replace("LightGBM", "LGBM").replace("NeuralNetFastAI", "MLP").replace("RandomForest", "RF").replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]

df_rank = df_rank[[index(name) is not None and index(name) < num_models_to_plot for name in df_rank.index]]

ordered_rows, ordered_cols = order_clustermap(df_rank)
df_rank = df_rank.loc[ordered_rows]
df_rank = df_rank[ordered_cols]
df_rank.columns.name = "dataset"

# task-model rank
fig, axes = plt.subplots(1, 3, figsize=figsize)
ax = axes[0]
sns.heatmap(
    df_rank, cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax,
)
ax.set_xticks([])
ax.set_xlabel("datasets", fontdict={'size': title_size})
ax.set_title("Ranks of models per dataset", fontdict={'size': title_size})

# model-model correlation
ax = axes[1]
sns.heatmap(
    df_rank.T.corr(), cmap="vlag", vmin=-1, vmax=1, ax=ax,
)
ax.set_title("Model rank correlation", fontdict={'size': title_size})

# runtime figure
ax = axes[2]
df['framework_type'] = df.apply(lambda x: x["framework"].split("_")[0], axis=1)
for framework in df['framework_type'].unique():
    df_framework = df.loc[df.framework_type == framework, :]
    ax = df_framework.groupby("tid").max()['time_train_s'].sort_values().reset_index(drop=True).plot(marker=".", label=framework)
    ax.set_yscale('log')
ax.grid()
ax.legend();
ax.set_ylabel("Training runtime (s)", fontdict={'size': title_size})
ax.set_title("Training runtime distribution", fontdict={'size': title_size})

plt.tight_layout()
fig_save_path = Path(__file__).parent / "figures"
fig_save_path.mkdir(exist_ok=True)
plt.savefig(fig_save_path / "data-analysis.pdf")
plt.show()
