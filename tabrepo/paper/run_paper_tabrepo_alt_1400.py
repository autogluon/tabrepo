from __future__ import annotations

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from examples.run_scripts_v5.script_utils import load_ag11_bq_baseline, load_ag12_eq_baseline
from tabrepo.paper.paper_runner import PaperRunMini


"""
Ablation for paper to justify the new search spaces

Results ran on TabRepo 1.0 datasets for 1 fold, 191 datasets (remove image datasets)

200 alt search space configs each for:
- RandomForest
- ExtraTrees
- LightGBM
- XGBoost
- CatBoost
- RealMLP
- EBM

"""


if __name__ == '__main__':
    # The original TabRepo artifacts for the 1530 configs
    context_name = "D244_F3_C1530_200"
    context_name_cache = context_name + "_alt_1400"
    use_repo_cache = False
    repo_path = f"./repo/{context_name_cache}.pkl"

    if use_repo_cache:
        my_repo = EvaluationRepositoryCollection.load(path=repo_path)
    else:
        my_repo: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
        df_baselines_ag11 = load_ag11_bq_baseline(datasets=my_repo.datasets(), folds=my_repo.folds, repo=my_repo)
        df_baselines_ag12 = load_ag12_eq_baseline(datasets=my_repo.datasets(), folds=my_repo.folds, repo=my_repo)
        repo_ag11 = EvaluationRepository.from_raw(df_configs=None, df_baselines=df_baselines_ag11, task_metadata=my_repo.task_metadata, results_lst_simulation_artifacts=None)
        repo_ag12 = EvaluationRepository.from_raw(df_configs=None, df_baselines=df_baselines_ag12, task_metadata=my_repo.task_metadata, results_lst_simulation_artifacts=None)
        repo_realmlp = EvaluationRepository.from_dir("./../../examples/repo_realmlp_r100")
        repo_realmlp = repo_realmlp.subset(datasets=my_repo.datasets())
        repo_realmlp = repo_realmlp.subset(configs=["RealMLP_c1_BAG_L1"])

        repo_realmlp_alt = EvaluationRepository.from_dir("./../../examples/repos/tabarena_big_s3_realmlp_alt")
        repo_realmlp_alt = repo_realmlp_alt.subset(datasets=my_repo.datasets())

        repo_alt_500 = EvaluationRepository.from_dir("./../../examples/repos/tabarena_big_alt_500")
        repo_alt_1400 = EvaluationRepository.from_dir("./../../examples/repos/tabarena_big_alt_1400")
        folds = [0]

        repo_alt_500_configs = repo_alt_500.configs()
        banned_alt_500_configs = [f"XGBoost_r{i}_alt_BAG_L1" for i in range(1, 101)]
        repo_alt_500_configs = [c for c in repo_alt_500_configs if c not in banned_alt_500_configs]
        repo_alt_500 = repo_alt_500.subset(configs=repo_alt_500_configs)

        # datasets_dense_realmlp_alt = repo_realmlp_alt.datasets(union=False)
        repo_ag11 = repo_ag11.subset(datasets=my_repo.datasets())
        my_repo = EvaluationRepositoryCollection(repos=[
            my_repo,
            repo_realmlp,
            # repo_realmlp_alt,
            repo_ag11,
            repo_ag12,
            repo_alt_500,
            repo_alt_1400,
        ])

        my_repo = my_repo.subset(folds=folds)
        my_repo = my_repo.subset(datasets=repo_alt_1400.datasets())

        my_repo_datasets = my_repo.datasets()
        # datasets_to_keep = [d for d in my_repo_datasets if d in datasets_dense_realmlp_alt]
        # my_repo = my_repo.subset(datasets=datasets_to_keep)

        my_repo.save(path=repo_path)

    configs = my_repo.configs()

    # my_repo = my_repo.subset(datasets=my_repo.datasets()[:5])

    # my_repo = my_repo.subset(problem_types=["regression"])

    my_repo = my_repo.subset(configs=configs)
    my_repo.set_config_fallback(config_fallback="ExtraTrees_c1_BAG_L1")
    print(configs)

    my_repo.print_info()

    paper_run = PaperRunMini(repo=my_repo)

    # paper_run.generate_data_analysis(expname_outdir="tmp3")

    df_results = paper_run.run()

    from autogluon.common.savers import save_pd, save_pkl
    from autogluon.common.loaders import load_pd, load_pkl

    # save_path = f"./tmp/{context_name}/df_results.pkl"
    # save_pkl.save(path=save_path, object=df_results)
    # df_results = load_pkl.load(path=save_path)

    save_path = f"./tmp/{context_name_cache}/df_results.parquet"
    save_pd.save(df=df_results, path=save_path)
    df_results = load_pd.load(path=save_path)

    paper_run.eval(df_results=df_results)
