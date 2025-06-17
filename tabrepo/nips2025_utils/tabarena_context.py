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
from tabrepo.nips2025_utils.artifacts import tabarena_method_metadata_map
from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.tabarena51_artifact_loader import TabArena51ArtifactLoader
from tabrepo.nips2025_utils.eval_all import evaluate_all


_methods_paper = [
    "AutoGluon_v130",
    "Portfolio-N200-4h",

    "CatBoost",
    # "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    # "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    # "RealMLP",
    # "TabM",
    "XGBoost",

    # "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
]


class TabArenaContext:
    def __init__(self):
        self.name = "tabarena-2025-06-12"
        self.method_metadata_map: dict[str, MethodMetadata] = tabarena_method_metadata_map
        self.root_cache = Paths.artifacts_root_cache_tabarena
        self.task_metadata = load_task_metadata(paper=True)  # FIXME: Instead download?
        self.backend = "ray"
        assert self.backend in ["ray", "native"]
        self.engine = "ray" if self.backend == "ray" else "sequential"

    def _method_metadata(self, method: str) -> MethodMetadata:
        return self.method_metadata_map[method]

    def generate_repo(self, method: str) -> str:
        metadata = self._method_metadata(method=method)

        path_raw = metadata.path_raw
        path_processed = metadata.path_processed

        name_suffix = metadata.name_suffix

        file_paths_method = fetch_all_pickles(dir_path=path_raw)
        repo: EvaluationRepository = generate_repo_from_paths(
            result_paths=file_paths_method,
            task_metadata=self.task_metadata,
            engine=self.engine,
            name_suffix=name_suffix,
        )

        repo.to_dir(path_processed)
        return path_processed

    def simulate_repo(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        metadata = self._method_metadata(method=method)
        path_processed = metadata.path_processed

        metadata_rf = self._method_metadata(method="RandomForest")
        path_processed_rf = metadata_rf.path_processed
        config_fallback = metadata_rf.config_default

        save_file = str(metadata.path_results_hpo)
        save_file_model = str(metadata.path_results_model)
        repo = EvaluationRepository.from_dir(path=path_processed)

        model_types = repo.config_types()
        # FIXME: Try to avoid this being expensive
        if config_fallback not in repo.configs():
            repo_rf = EvaluationRepository.from_dir(path=path_processed_rf)
            repo_rf_mini = repo_rf.subset(configs=[config_fallback])
            repo = EvaluationRepositoryCollection(repos=[repo, repo_rf_mini])
        repo.set_config_fallback(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        # FIXME: do this in simulator automatically

        if metadata.can_hpo:
            hpo_results = simulator.run_minimal_paper(model_types=model_types)
            hpo_results["ta_name"] = metadata.method
            hpo_results["ta_suite"] = metadata.artifact_name
            hpo_results = hpo_results.rename(columns={"framework": "method"})  # FIXME: Don't do this, make it method by default
            save_pd.save(path=save_file, df=hpo_results)
        else:
            hpo_results = None

        config_results = simulator.run_configs(model_types=model_types)
        baseline_results = simulator.run_baselines()
        results_lst = [config_results, baseline_results]
        results_lst = [r for r in results_lst if r is not None]
        model_results = pd.concat(results_lst, ignore_index=True)

        model_results["ta_name"] = metadata.method
        model_results["ta_suite"] = metadata.artifact_name
        model_results = model_results.rename(columns={"framework": "method"})  # FIXME: Don't do this, make it method by default
        save_pd.save(path=save_file_model, df=model_results)

        return hpo_results, model_results

    def simulate_portfolio(self, methods: list[str | tuple[str, str]], config_fallback: str):
        repos = []
        for method in methods:
            metadata = self._method_metadata(method=method)
            cur_repo = EvaluationRepository.from_dir(path=metadata.path_processed)
            repos.append(cur_repo)
        repo = EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        df_results_n_portfolio = []
        n_portfolios = [200]
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                simulator.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False))
        results = pd.concat(df_results_n_portfolio, ignore_index=True)

        results = results.rename(columns={"framework": "method"})
        return results

    def load_hpo_results(self, method: str) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return pd.read_parquet(path=metadata.path_results_hpo)

    def load_config_results(self, method: str) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return pd.read_parquet(path=metadata.path_results_model)

    def load_portfolio_results(self, method: str) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return pd.read_parquet(path=metadata.path_results_portfolio)

    def load_results_paper(self, methods: list[str] | None = None, download_results: str | bool = False) -> pd.DataFrame:
        if isinstance(download_results, bool) and download_results:
            loader = TabArena51ArtifactLoader()
            loader.download_results()
        try:
            df_results = self._load_results_paper(methods=methods)
        except FileNotFoundError as err:
            if isinstance(download_results, str) and download_results == "auto":
                print(f"Missing local results files! Attempting to download them and retry... (download_results={download_results})")
                loader = TabArena51ArtifactLoader()
                loader.download_results()
                df_results = self._load_results_paper(methods=methods)
            else:
                print(f"Missing local results files! Try setting `download_results=True` to get the required files.")
                raise err
        return df_results

    def _load_results_paper(self, methods: list[str] | None = None) -> pd.DataFrame:
        if methods is None:
            methods = _methods_paper
        assert methods is not None and len(methods) > 0
        df_metadata_lst = []
        for method in methods:
            metadata = self._method_metadata(method=method)
            if metadata.method_type == "config":
                df_metadata = self.load_hpo_results(method=method)
            elif metadata.method_type == "baseline":
                df_metadata = self.load_config_results(method=method)
            elif metadata.method_type == "portfolio":
                df_metadata = self.load_portfolio_results(method=method)
            else:
                raise ValueError(f"Unknown method_type: {metadata.method_type} for method {method}")
            df_metadata_lst.append(df_metadata)
        df_metadata = pd.concat(df_metadata_lst, ignore_index=True)
        return df_metadata

    @classmethod
    def evaluate_all(
        cls,
        df_results: pd.DataFrame,
        save_path: str | Path,
        elo_bootstrap_rounds: int = 100,
    ):
        evaluate_all(df_results=df_results, eval_save_path=save_path, elo_bootstrap_rounds=elo_bootstrap_rounds)

    def find_missing(self, method: str):
        metadata = self._method_metadata(method=method)
        repo = EvaluationRepository.from_dir(path=metadata.path_processed)

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
        df_missing = df_missing.rename(columns={"framework": "method"})
        print(df_missing)

        # save_pd.save(path="missing_runs.csv", df=df_missing)

        return df_missing
