from __future__ import annotations

import random

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from examples.run_scripts_v5.script_utils import load_ag11_bq_baseline
from tabrepo.paper.paper_runner import PaperRun


if __name__ == '__main__':

    # The original TabRepo artifacts for the 1530 configs
    context_name = "D244_F3_C1530_200"
    my_repo: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    df_baselines_ag11 = load_ag11_bq_baseline(datasets=my_repo.datasets(), folds=my_repo.folds, repo=my_repo)
    repo_ag11 = EvaluationRepository.from_raw(df_configs=None, df_baselines=df_baselines_ag11, task_metadata=my_repo.task_metadata, results_lst_simulation_artifacts=None)
    repo_realmlp = EvaluationRepository.from_dir("./../../examples/repo_realmlp_r100")
    repo_realmlp = repo_realmlp.subset(datasets=my_repo.datasets())
    repo_ag11 = repo_ag11.subset(datasets=my_repo.datasets())
    my_repo = EvaluationRepositoryCollection(repos=[my_repo, repo_realmlp, repo_ag11])

    my_configs = my_repo.configs()
    random.seed(0)
    # configs = random.sample(my_configs, 30)
    configs = my_configs
    configs_defaults = [c for c in my_configs if "_c1" in c]
    configs = list(set(configs + configs_defaults))
    my_repo = my_repo.subset(configs=configs)
    my_repo.set_config_fallback(config_fallback="ExtraTrees_c1_BAG_L1")
    print(configs)

    my_repo.print_info()

    paper_run = PaperRun(repo=my_repo)

    # df_results = paper_run.run()

    from autogluon.common.savers import save_pd
    from autogluon.common.loaders import load_pd

    save_path = f"./tmp/{context_name}/df_results.parquet"
    # save_pd.save(df=df_results, path=save_path)
    df_results = load_pd.load(path=save_path)

    paper_run.eval(df_results=df_results)
    paper_run.eval(df_results=df_results, use_gmean=True)
