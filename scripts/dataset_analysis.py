import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from autogluon_zeroshot.repository.evaluation_repository import load, EvaluationRepository
from autogluon_zeroshot.utils.cache import cache_function
from baseline_comparison.plot_utils import save_latex_table
from scripts import load_context


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


def generate_dataset_info_latex(repo: EvaluationRepository):
    metadata = repo._df_metadata.copy()
    assert len(metadata) == len(repo.tids())

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        metadata['problem_type'] = ''
        metadata['problem_type'][metadata['NumberOfClasses'] == 2] = 'binary'
        metadata['problem_type'][metadata['NumberOfClasses'] > 2] = 'multiclass'
        metadata['problem_type'][metadata['NumberOfClasses'] == 0] = 'regression'

    metadata_min = metadata[["tid", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", 'problem_type']]
    metadata_min_sorted = metadata_min.sort_values(by=["name"])
    metadata_latex = metadata_min_sorted.copy()

    metadata_latex.columns = ['Task ID', 'name', 'n', 'f', 'C', 'Problem Type']
    metadata_latex['n'] = metadata_latex['n'].astype(int)
    metadata_latex['f'] = metadata_latex['f'].astype(int)
    metadata_latex['C'] = metadata_latex['C'].astype(int)
    metadata_latex['name'] = metadata_latex['name'].apply(lambda x: x[:25])

    problem_types = ['binary', 'multiclass', 'regression']
    latex_kwargs = dict(
        index=False,
    )

    # Separate by problem types and into batches to ensure each table fits on one page
    for p in problem_types:
        metadata_latex_p = metadata_latex[metadata_latex['Problem Type'] == p]
        num_datasets = len(metadata_latex_p)
        batch_size = 40
        num_batches = math.ceil(num_datasets / batch_size)
        for batch in range(num_batches):
            save_latex_table(df=metadata_latex_p.iloc[batch * batch_size:(batch + 1) * batch_size], title=f"dataset_info_{p}_{batch + 1}", show_table=True,
                             latex_kwargs=latex_kwargs)


num_models_to_plot = 20
title_size = 20
figsize = (20, 7)

repo_version = "BAG_D244_F3_C1416"
repo: EvaluationRepository = load_context(version=repo_version)

# Fails with: ValueError: Unknown format code 'f' for object of type 'str'
# generate_dataset_info_latex(repo=repo)

zsc = repo._zeroshot_context

df = zsc.df_results_by_dataset_vs_automl.copy()
# # remove tasks with some lightGBM models missing, todo fix
# missing_tids = [359932, 359944, 359933, 359946]
# df = df[~df.tid.isin(missing_tids)]

config_regexp = "(" + "|".join([str(x) for x in range(6)]) + ")"
df = df[df.framework.str.contains(f"r{config_regexp}_BAG_L1")]
df.framework = df.framework.str.replace("NeuralNetTorch", "MLP")

metric = "metric_error"
df_pivot = df.pivot_table(
    index="framework", columns="tid", values=metric
)
df_rank = df_pivot.rank() / len(df_pivot)
df_rank.index = [x.replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]
# shorten framework names
#df_rank.index = [x.replace("ExtraTrees", "ET").replace("CatBoost", "CB").replace("LightGBM", "LGBM").replace("NeuralNetFastAI", "MLP").replace("RandomForest", "RF").replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]

df_rank = df_rank[[index(name) is not None and index(name) < num_models_to_plot for name in df_rank.index]]

ordered_rows, ordered_cols = order_clustermap(df_rank)
df_rank = df_rank.loc[ordered_rows]
df_rank = df_rank[ordered_cols]
df_rank.columns.name = "dataset"

# task-model rank
fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=300)
ax = axes[0]
sns.heatmap(
    df_rank, cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax,
)
ax.set_xticks([])
ax.set_xlabel("Datasets", fontdict={'size': title_size})
ax.set_title("Ranks of models per dataset", fontdict={'size': title_size})

# model-model correlation
ax = axes[1]
sns.heatmap(
    df_rank.T.corr(), cmap="vlag", vmin=-1, vmax=1, ax=ax,
)
ax.set_title("Model rank correlation", fontdict={'size': title_size})

# runtime figure
df = zsc.df_results_by_dataset_vs_automl
ax = axes[2]
df['framework_type'] = df.apply(lambda x: x["framework"].split("_")[0], axis=1)
for framework in df['framework_type'].unique():
    df_framework = df.loc[df.framework_type == framework, :]
    ax = df_framework.groupby("tid").max()['time_train_s'].sort_values().reset_index(drop=True).plot(marker=".", label=framework)
    ax.set_yscale('log')
ax.grid()
ax.legend();
ax.set_xlabel("Datasets", fontdict={'size': title_size})
ax.set_ylabel("Training runtime (s)", fontdict={'size': title_size})
ax.set_title("Training runtime distribution", fontdict={'size': title_size})

plt.tight_layout()
fig_save_path = Path(__file__).parent / "figures"
fig_save_path.mkdir(exist_ok=True)
plt.savefig(fig_save_path / "data-analysis.pdf")
plt.show()
