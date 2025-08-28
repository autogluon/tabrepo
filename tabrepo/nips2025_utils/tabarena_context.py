from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from autogluon.common.savers import save_pd

from tabrepo.benchmark.result import BaselineResult
from tabrepo.utils.pickle_utils import fetch_all_pickles
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from tabrepo.repository.abstract_repository import AbstractRepository
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
    "RealMLP",
    # "TabM",
    "XGBoost",

    # "Mitra_GPU",
    "ModernNCA_GPU",
    # "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
]


class TabArenaContext:
    def __init__(
        self,
        backend: Literal["ray", "native"] = "ray",
    ):
        self.name = "tabarena-2025-06-12"
        self.method_metadata_map: dict[str, MethodMetadata] = tabarena_method_metadata_map
        self.root_cache = Paths.artifacts_root_cache_tabarena
        self.s3_cache_root_url = "https://tabarena.s3.us-west-2.amazonaws.com/cache"
        self.task_metadata = load_task_metadata(paper=True)  # FIXME: Instead download?
        assert backend in ["ray", "native"]
        self.backend = backend
        self.engine = "ray" if self.backend == "ray" else "sequential"

    @property
    def methods(self) -> list[str]:
        return list(self.method_metadata_map.keys())

    def method_metadata(self, method: str) -> MethodMetadata:
        return self._method_metadata(method=method)

    def _method_metadata(self, method: str) -> MethodMetadata:
        return self.method_metadata_map[method]

    def load_raw(self, method: str, as_holdout: bool = False) -> list[BaselineResult]:
        metadata: MethodMetadata = self.method_metadata(method=method)
        results_lst = metadata.load_raw(engine=self.engine, as_holdout=as_holdout)
        return results_lst

    def load_repo(self, methods: list[str | MethodMetadata], config_fallback: str | None = None) -> EvaluationRepositoryCollection:
        repos = []
        for method in methods:
            if isinstance(method, MethodMetadata):
                metadata = method
            else:
                metadata = self.method_metadata(method=method)
            cur_repo = EvaluationRepository.from_dir(
                path=metadata.path_processed,
            )
            repos.append(cur_repo)
        repo = EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)
        return repo

    def generate_repo(self, method: str) -> Path:
        metadata = self._method_metadata(method=method)

        path_raw = metadata.path_raw
        path_processed = metadata.path_processed

        name_suffix = metadata.name_suffix

        file_paths_method = fetch_all_pickles(dir_path=path_raw, suffix="results.pkl")
        repo: EvaluationRepository = generate_repo_from_paths(
            result_paths=file_paths_method,
            task_metadata=self.task_metadata,
            engine=self.engine,
            name_suffix=name_suffix,
        )

        repo.to_dir(path_processed)
        return path_processed

    def generate_repo_holdout(self, method: str) -> Path:
        metadata = self._method_metadata(method=method)

        path_raw = metadata.path_raw
        path_processed = metadata.path_processed_holdout

        name_suffix = metadata.name_suffix

        file_paths_method = fetch_all_pickles(dir_path=path_raw, suffix="results.pkl")
        repo: EvaluationRepository = generate_repo_from_paths(
            result_paths=file_paths_method,
            task_metadata=self.task_metadata,
            engine=self.engine,
            name_suffix=name_suffix,
            as_holdout=True,
        )

        repo.to_dir(path_processed)
        return path_processed

    def simulate_repo(
        self,
        method: str | MethodMetadata,
        repo: EvaluationRepository | None = None,
        holdout: bool = False,
        use_rf_config_fallback: bool = True,
        cache: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(method, MethodMetadata):
            metadata = method
            method = metadata.method
        else:
            metadata = self._method_metadata(method=method)

        save_file = str(metadata.path_results_hpo(holdout=holdout))
        save_file_model = str(metadata.path_results_model(holdout=holdout))
        if repo is None:
            if holdout:
                path_processed = metadata.path_processed_holdout
            else:
                path_processed = metadata.path_processed
            repo = EvaluationRepository.from_dir(path=path_processed)

        model_types = repo.config_types()

        if use_rf_config_fallback:
            metadata_rf = self._method_metadata(method="RandomForest")
            if holdout:
                config_fallback = "RandomForest_r1_BAG_L1_HOLDOUT"  # FIXME: Avoid hardcoding
                path_processed_rf = metadata_rf.path_processed_holdout
            else:
                config_fallback = metadata_rf.config_default
                path_processed_rf = metadata_rf.path_processed

            # FIXME: Try to avoid this being expensive
            if config_fallback not in repo.configs():
                repo_rf = EvaluationRepository.from_dir(path=path_processed_rf)
                repo_rf_mini = repo_rf.subset(configs=[config_fallback])
                repo = EvaluationRepositoryCollection(repos=[repo, repo_rf_mini])
            repo.set_config_fallback(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        # FIXME: do this in simulator automatically

        if metadata.method_type == "config":
            hpo_results = simulator.run_minimal_paper(model_types=model_types, tune=metadata.can_hpo)
            hpo_results["ta_name"] = metadata.method
            hpo_results["ta_suite"] = metadata.artifact_name
            hpo_results = hpo_results.rename(columns={"framework": "method"})  # FIXME: Don't do this, make it method by default
            if cache:
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
        if cache:
            save_pd.save(path=save_file_model, df=model_results)

        return hpo_results, model_results

    def simulate_portfolio(self, methods: list[str], config_fallback: str):
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

    def load_hpo_results(self, method: str, holdout: bool = False) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return metadata.load_hpo_results(holdout=holdout)

    def load_config_results(self, method: str, holdout: bool = False) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return metadata.load_model_results(holdout=holdout)

    def load_portfolio_results(self, method: str, holdout: bool = False) -> pd.DataFrame:
        metadata = self._method_metadata(method=method)
        return metadata.load_portfolio_results(holdout=holdout)

    def load_results_paper(
        self,
        methods: list[str] | None = None,
        holdout: bool = False,
        download_results: str | bool = "auto",
        methods_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        if methods is None:
            methods = _methods_paper
            if holdout:
                # only include methods that can have holdout
                methods = [m for m in methods if self._method_metadata(m).is_bag]
        if methods_drop is not None:
            for method in methods_drop:
                if method not in methods:
                    raise AssertionError(
                        f"Specified '{method}' in `methods_drop`, "
                        f"but '{method}' is not present in methods: {methods}"
                    )
            methods = [method for method in methods if method not in methods_drop]
        if isinstance(download_results, bool) and download_results:
            loader = TabArena51ArtifactLoader()
            loader.download_results(holdout=holdout)
        try:
            df_results = self._load_results_paper(methods=methods, holdout=holdout)
        except FileNotFoundError as err:
            if isinstance(download_results, str) and download_results == "auto":
                print(f"Missing local results files! Attempting to download them and retry... (download_results={download_results})")
                loader = TabArena51ArtifactLoader()
                loader.download_results(holdout=holdout)
                df_results = self._load_results_paper(methods=methods, holdout=holdout)
            else:
                print(f"Missing local results files! Try setting `download_results=True` to get the required files.")
                raise err
        return df_results

    def _load_results_paper(self, methods: list[str], holdout: bool = False) -> pd.DataFrame:
        assert methods is not None and len(methods) > 0
        df_metadata_lst = []
        for method in methods:
            metadata = self._method_metadata(method=method)
            if metadata.method_type == "config":
                df_metadata = self.load_hpo_results(method=method, holdout=holdout)
            elif metadata.method_type == "baseline":
                df_metadata = self.load_config_results(method=method, holdout=holdout)
            elif metadata.method_type == "portfolio":
                df_metadata = self.load_portfolio_results(method=method, holdout=holdout)
            else:
                raise ValueError(f"Unknown method_type: {metadata.method_type} for method {method}")
            df_metadata_lst.append(df_metadata)
        df_metadata = pd.concat(df_metadata_lst, ignore_index=True)
        return df_metadata

    def load_configs_hyperparameters(self, methods: list[str] | None = None, holdout: bool = False, download: bool | str = False) -> dict[str, dict]:
        if methods is None:
            methods = _methods_paper
            methods = [m for m in methods if self._method_metadata(m).method_type == "config"]
            if holdout:
                # only include methods that can have holdout
                methods = [m for m in methods if self._method_metadata(m).is_bag]
        configs_hyperparameters_lst = []
        for method in methods:
            metadata = self._method_metadata(method=method)
            if isinstance(download, bool) and download:
                metadata.download_configs_hyperparameters(s3_cache_root=self.s3_cache_root_url, holdout=holdout)
            try:
                configs_hyperparameters = metadata.load_configs_hyperparameters(holdout=holdout)
            except FileNotFoundError as err:
                if isinstance(download, str) and download == "auto":
                    print(
                        f"Cache miss detected for configs_hyperparameters.json "
                        f"(method={metadata.method}), attempting download..."
                    )
                    metadata.download_configs_hyperparameters(s3_cache_root=self.s3_cache_root_url, holdout=holdout)
                    configs_hyperparameters = metadata.load_configs_hyperparameters(holdout=holdout)
                    print(f"\tDownload successful")
                else:
                    raise err
            configs_hyperparameters_lst.append(configs_hyperparameters)

        def merge_dicts_no_duplicates(dicts: list[dict]) -> dict:
            merged = {}
            for d in dicts:
                for key in d:
                    if key in merged:
                        raise KeyError(
                            f"Duplicate key found in configs_hyperparameters: {key}\n"
                            f"This should never happen and may mean that a given config name "
                            f"belongs to multiple different hyperparameters!"
                        )
                merged.update(d)
            return merged

        configs_hyperparameters = merge_dicts_no_duplicates(configs_hyperparameters_lst)
        return configs_hyperparameters

    def evaluate_all(
        self,
        df_results: pd.DataFrame,
        save_path: str | Path,
        df_results_holdout: pd.DataFrame = None,
        df_results_cpu: pd.DataFrame = None,
        configs_hyperparameters: dict[str, dict] = None,
        elo_bootstrap_rounds: int = 100,
        use_latex: bool = False,
    ):

        ta_names = list(df_results["ta_name"].unique())
        if df_results_cpu is not None:
            ta_names_cpu = list(df_results_cpu["ta_name"].unique())
            ta_names += ta_names_cpu
            ta_names = list(set(ta_names))
        ta_names = [c for c in ta_names if c != np.nan]

        df_results_configs_lst = []
        for method_key in ta_names:
            metadata = self.method_metadata_map[method_key]
            if metadata.method_type == "config":
                df_results_configs_lst.append(self.load_config_results(method_key))
        df_results_configs = pd.concat(df_results_configs_lst, ignore_index=True)

        evaluate_all(
            df_results=df_results,
            df_results_holdout=df_results_holdout,
            df_results_cpu=df_results_cpu,
            df_results_configs=df_results_configs,
            configs_hyperparameters=configs_hyperparameters,
            eval_save_path=save_path,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            use_latex=use_latex,
        )

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

    @classmethod
    def fillna_metrics(cls, df_to_fill: pd.DataFrame, df_fillna: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing (dataset, fold, framework) rows in df_to_fill with the (dataset, fold) row in df_fillna.

        Parameters
        ----------
        df_to_fill
        df_fillna

        Returns
        -------

        """

        method_col = "method"
        split_col = "fold"
        dataset_col = "dataset"

        df_to_fill = df_to_fill.set_index([dataset_col, split_col, method_col], drop=True)
        df_fillna = df_fillna.set_index([dataset_col, split_col], drop=True).drop(columns=[method_col])

        unique_frameworks = list(df_to_fill.index.unique(level=method_col))

        df_filled = df_fillna.index.to_frame().merge(
            pd.Series(data=unique_frameworks, name=method_col),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(df_to_fill.index)

        # fill valid values
        fill_cols = list(df_to_fill.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(df_to_fill.dtypes)
        df_filled.loc[df_to_fill.index] = df_to_fill

        a = df_fillna.loc[nan_vals.droplevel(level=method_col)]
        a.index = nan_vals
        df_filled.loc[nan_vals] = a

        if "imputed" not in df_filled.columns:
            df_filled["imputed"] = False
        df_filled.loc[nan_vals, "imputed"] = True

        df_to_fill = df_filled

        df_to_fill = df_to_fill.reset_index(drop=False)

        return df_to_fill
