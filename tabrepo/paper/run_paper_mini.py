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
    # The original TabRepo artifacts for the 1530 configs
    context_name = "D244_F3_C1530_200"
    n_configs = None
    use_repo_cache = False
    repo_path = f"./repo/{context_name}_mini.pkl"

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
        repo_ag11 = repo_ag11.subset(datasets=my_repo.datasets())
        my_repo = EvaluationRepositoryCollection(repos=[my_repo, repo_realmlp, repo_ag11, repo_ag12])

        my_repo.save(path=repo_path)

    my_configs = my_repo.configs()
    if n_configs is not None:
        configs = sample_configs(configs=my_configs, n_configs=n_configs)
    else:
        configs = my_configs

    my_repo = my_repo.subset(configs=configs)
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
