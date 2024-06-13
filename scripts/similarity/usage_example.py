import numpy as np

from tabrepo import load_repository
from tabrepo.portfolio.similarity import distance_tasks_from_repo
from tabrepo.portfolio.zeroshot_selection import zeroshot_configs

if __name__ == '__main__':
    nclosest = 20
    repo = load_repository("D244_F3_C1530_200")

    # sets scores of configurations evaluated
    model_metrics_new_task = {
        'CatBoost_r177_BAG_L1': 0.018300653594771243,
        'NeuralNetTorch_r79_BAG_L1': 0.677124183006536,
        'LightGBM_r131_BAG_L1': 0.3326797385620915,
        'FTTransformer_c7_BAG_L1': 0.954248366013072,
        'CatBoost_r71_BAG_L1': 0.08169934640522876,
        'TabPFN_c2_BAG_L1': 0.9450980392156862,
        'FTTransformer_c3_BAG_L1': 0.9529411764705882,
        'NeuralNetFastAI_r134_BAG_L1': 0.4620915032679739,
        'FTTransformer_c5_BAG_L1': 0.9535947712418301,
        'FTTransformer_c4_BAG_L1': 0.9549019607843138,
        'FTTransformer_c1_BAG_L1': 0.9513071895424836,
        'FTTransformer_c6_BAG_L1': 0.9522875816993464
    }

    # get the closest tasks compared to the given evaluations
    distances = distance_tasks_from_repo(repo, model_metrics=model_metrics_new_task)
    tasks_with_distances = [
        ((dataset, fold), distance)
        for (dataset, fold), distance in distances.items()
    ]
    tasks_with_distances = list(sorted(tasks_with_distances, key=lambda x: x[1]))
    closest_tasks = [
        repo.task_name_from_tid(repo.dataset_to_tid(dataset), fold)
        for (dataset, fold), distance in tasks_with_distances
    ][:nclosest]

    # compute zeroshot configurations on the k closest datasets, why do we need so many lines? :-o
    dd = repo._zeroshot_context.df_configs_ranked
    df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    portfolio_indices = zeroshot_configs(val_scores=-df_rank[closest_tasks].values.T, output_size=20)
    portfolio_configs = np.array(repo.configs())[portfolio_indices]

    print("**Computing portfolio with weights**")
    print(f"Portfolio indices: {portfolio_indices}")
    print(f"Portfolio configs: {portfolio_configs}")