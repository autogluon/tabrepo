from __future__ import annotations

import random

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from examples.run_scripts_v5.script_utils import load_ag11_bq_baseline, load_ag12_eq_baseline
from tabrepo.paper.paper_runner import PaperRunMini


def sample_configs(configs: list[str], n_configs: int) -> list[str]:
    configs_defaults = [c for c in configs if "_c1" in c]
    configs_nondefault = [c for c in configs if c not in configs_defaults]
    n_configs_nondefault = n_configs - len(configs_defaults)
    if n_configs_nondefault > 0:
        random.seed(0)
        configs_sampled_nondefault = random.sample(configs_nondefault, n_configs_nondefault)
        configs = list(set(configs_defaults + configs_sampled_nondefault))
    else:
        configs = configs_defaults
    return configs


if __name__ == '__main__':
    context_name = "tabarena_big"
    n_configs = None
    use_repo_cache = False
    repo_path = f"./repo/{context_name}.pkl"

    if use_repo_cache:
        my_repo = EvaluationRepositoryCollection.load(path=repo_path)
    else:
        my_repo: EvaluationRepository = EvaluationRepository.from_dir("../../../examples/repos/tabarena_big_s3")
        tabrepo_repo: EvaluationRepository = EvaluationRepository.from_context("D244_F3_C1530_200", cache=True)
        df_baselines_ag11 = load_ag11_bq_baseline(datasets=my_repo.datasets(), folds=my_repo.folds, repo=tabrepo_repo)
        df_baselines_ag12 = load_ag12_eq_baseline(datasets=my_repo.datasets(), folds=my_repo.folds, repo=tabrepo_repo)
        tabrepo_repo = tabrepo_repo.subset(configs=[], datasets=my_repo.datasets(), folds=my_repo.folds)
        repo_ag11 = EvaluationRepository.from_raw(df_configs=None, df_baselines=df_baselines_ag11, task_metadata=my_repo.task_metadata, results_lst_simulation_artifacts=None)
        repo_ag12 = EvaluationRepository.from_raw(df_configs=None, df_baselines=df_baselines_ag12, task_metadata=my_repo.task_metadata, results_lst_simulation_artifacts=None)
        repo_realmlp = EvaluationRepository.from_dir("./../../../examples/repo_realmlp_r100")
        repo_realmlp = repo_realmlp.subset(datasets=my_repo.datasets(), folds=my_repo.folds)
        repo_ag11 = repo_ag11.subset(datasets=my_repo.datasets())
        repo_ag12 = repo_ag12.subset(datasets=my_repo.datasets())

        realmlp_configs = repo_realmlp.configs()
        my_repo = my_repo.subset(configs=[c for c in my_repo.configs() if c not in realmlp_configs])

        my_repo = EvaluationRepositoryCollection(repos=[my_repo, repo_ag11, repo_ag12, tabrepo_repo, repo_realmlp])

        my_repo.save(path=repo_path)

    my_configs = my_repo.configs()
    if n_configs is not None:
        configs = sample_configs(configs=my_configs, n_configs=n_configs)
    else:
        configs = my_configs

    my_repo = my_repo.subset(configs=configs)
    # my_repo = my_repo.subset(datasets=my_repo.datasets(union=False))
    my_repo.set_config_fallback(config_fallback="ExtraTrees_c1_BAG_L1")
    print(configs)

    my_repo.print_info()

    paper_run = PaperRunMini(repo=my_repo)

    # paper_run.generate_data_analysis(expname_outdir="tmp3")

    df_results = paper_run.run()

    from autogluon.common.savers import save_pd
    from autogluon.common.loaders import load_pd

    save_path = f"./tmp/{context_name}/df_results.parquet"
    save_pd.save(df=df_results, path=save_path)
    df_results = load_pd.load(path=save_path)

    paper_run.eval(df_results=df_results)
