from __future__ import annotations

from pathlib import Path

import pandas as pd

from autogluon.common.savers import save_pd

from tabrepo.utils.pickle_utils import fetch_all_pickles
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from tabrepo.nips2025_utils.generate_repo import generate_repo_from_paths
from tabrepo.loaders import Paths
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena


# FIXME: Remove "extra_prefix", find a more elegant solution
class TabArenaContext:
    def __init__(self):
        self.name = "tabarena51"
        self.data_raw_prefix: str = str(Paths.data_root_cache_raw / self.name)
        self.data_repo_prefix: str = str(Paths.data_root_cache_processed / self.name)
        self.method_results_prefix: str = str(Paths.results_root_cache_tabarena / self.name)
        self.task_metadata = load_task_metadata(paper=True)  # FIXME: Instead download?
        self.engine = "ray"

    def generate_repo(self, method: str, extra_prefix: str = None) -> str:
        data_prefix = self.data_raw_prefix
        repo_prefix = self.data_repo_prefix
        name_suffix = None

        if extra_prefix is not None:
            data_prefix = f"{data_prefix}/{extra_prefix}"
            repo_prefix = f"{repo_prefix}/{extra_prefix}"
            name_suffix = f"_{extra_prefix}"
        data_raw_dir = f"{data_prefix}/{method}"
        repo_dir = f"{repo_prefix}/{method}"
        file_paths_method = fetch_all_pickles(dir_path=data_raw_dir)
        repo: EvaluationRepository = generate_repo_from_paths(
            result_paths=file_paths_method,
            task_metadata=self.task_metadata,
            engine=self.engine,
            name_suffix=name_suffix,
        )

        repo.to_dir(repo_dir)
        return repo_dir

    def _results_dir(self, method: str, extra_prefix: str = None) -> str:
        if extra_prefix is not None:
            extra_prefix = f"/{extra_prefix}"
        else:
            extra_prefix = ""
        results_dir = f"{self.method_results_prefix}{extra_prefix}/{method}"
        return results_dir

    def _results_path(self, method: str, extra_prefix: str = None) -> str:
        return f"{self._results_dir(method=method, extra_prefix=extra_prefix)}/results.parquet"

    def _results_configs_path(self, method: str, extra_prefix: str = None) -> str:
        return f"{self._results_dir(method=method, extra_prefix=extra_prefix)}/config_results.parquet"

    def simulate_repo(self, method: str, extra_prefix: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_repo_prefix = self.data_repo_prefix
        if extra_prefix is not None:
            data_repo_prefix = f"{data_repo_prefix}/{extra_prefix}"
        repo_dir = f"{data_repo_prefix}/{method}"
        repo_dir_rf = f"{self.data_repo_prefix}/RandomForest"
        save_file = self._results_path(method=method, extra_prefix=extra_prefix)
        save_file_config = self._results_configs_path(method=method, extra_prefix=extra_prefix)
        repo = EvaluationRepository.from_dir(path=repo_dir)
        config_fallback = "RandomForest_c1_BAG_L1"
        model_types = repo.config_types()
        # FIXME: Try to avoid this being expensive
        if config_fallback not in repo.configs():
            repo_rf = EvaluationRepository.from_dir(path=repo_dir_rf)
            repo_rf_mini = repo_rf.subset(configs=[config_fallback])
            repo = EvaluationRepositoryCollection(repos=[repo, repo_rf_mini])
        repo.set_config_fallback(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo)
        # FIXME: do this in simulator automatically
        results = simulator.run_minimal_paper(model_types=model_types)
        if extra_prefix is not None:
            results["framework"] = results["framework"].apply(lambda x: x.split(" ", 1)[0] + "_" + extra_prefix + " " + x.split(" ", 1)[1])
            results["config_type"] = results["config_type"] + f"_{extra_prefix}"
        results = results.rename(columns={"framework": "method"})  # FIXME: Don't do this, make it method by default
        save_pd.save(path=save_file, df=results)

        config_results = simulator.run_configs(model_types=model_types)
        if extra_prefix is not None:
            config_results["config_type"] = config_results["config_type"] + f"_{extra_prefix}"
        config_results = config_results.rename(columns={"framework": "method"})  # FIXME: Don't do this, make it method by default
        save_pd.save(path=save_file_config, df=config_results)

        return results, config_results

    def simulate_portfolio(self, methods: list[str | tuple[str, str]], config_fallback: str):
        repos = []
        for method in methods:
            extra_prefix = None
            if isinstance(method, tuple):
                method, extra_prefix = method

            data_repo_prefix = self.data_repo_prefix
            if extra_prefix is not None:
                data_repo_prefix = f"{data_repo_prefix}/{extra_prefix}"
            repo_dir = f"{data_repo_prefix}/{method}"

            cur_repo = EvaluationRepository.from_dir(path=repo_dir)
            repos.append(cur_repo)
        repo = EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo)

        df_results_n_portfolio = []
        n_portfolios = [200]
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                simulator.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
            # df_results_n_portfolio.append(simulator.run_zs(n_portfolios=n_portfolio, n_ensemble=1, n_ensemble_in_name=False))
        results = pd.concat(df_results_n_portfolio, ignore_index=True)

        results = results.rename(columns={"framework": "method"})
        return results

    def load_results(self, method: str, extra_prefix: str = None) -> pd.DataFrame:
        path = self._results_path(method=method, extra_prefix=extra_prefix)
        return pd.read_parquet(path=path)

    def load_config_results(self, method: str, extra_prefix: str = None) -> pd.DataFrame:
        path = self._results_configs_path(method=method, extra_prefix=extra_prefix)
        return pd.read_parquet(path=path)

    def find_missing(self, method: str, extra_prefix: str = None):
        data_repo_prefix = self.data_repo_prefix
        if extra_prefix is not None:
            data_repo_prefix = f"{data_repo_prefix}/{extra_prefix}"
        repo_dir = f"{data_repo_prefix}/{method}"
        repo = EvaluationRepository.from_dir(path=repo_dir)

        tasks = repo.tasks()

        n_tasks = len(tasks)
        print(f"Method: {method} | n_tasks={n_tasks}")

        metrics = repo.metrics()
        metrics = metrics.reset_index(drop=False)

        configs = repo.configs()

        n_configs = len(configs)

        runs_missing_lst = []

        fail_dict = {}
        for i, config in enumerate(configs):
            metrics_config = metrics[metrics["framework"] == config]
            n_tasks_config = len(metrics_config)

            tasks_config = list(metrics_config[["dataset", "fold"]].values)
            tasks_config = set([tuple(t) for t in tasks_config])

            n_tasks_missing = n_tasks - n_tasks_config
            if n_tasks_missing != 0:
                tasks_missing = [t for t in tasks if t not in tasks_config]
            else:
                tasks_missing = []

            for dataset, fold in tasks_missing:
                runs_missing_lst.append(
                    (dataset, fold, config)
                )

            print(f"{n_tasks_missing}\t{config}\t{i + 1}/{n_configs}")
            fail_dict[config] = n_tasks_missing

        import pandas as pd
        # fail_series = pd.Series(fail_dict).sort_values()

        df_missing = pd.DataFrame(data=runs_missing_lst, columns=["dataset", "fold", "framework"])
        print(df_missing)

        # save_pd.save(path="missing_runs.csv", df=df_missing)

        return df_missing
