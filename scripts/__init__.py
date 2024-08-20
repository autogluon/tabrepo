from pathlib import Path

from tabrepo.utils.cache import cache_function
from tabrepo import load_repository, EvaluationRepository

output_path = Path(__file__).parent


def load_context(version: str = "D244_F3_C1416_200", as_paper: bool = True, ignore_cache: bool = False) -> EvaluationRepository:
    def _load_fun():
        return load_repository(version=version)
    repo: EvaluationRepository = cache_function(_load_fun, cache_name=f"repo_{version}", ignore_cache=ignore_cache)

    if as_paper:
        # only pick models that have suffix rXX or c1
        # filter NeuralNetFastAI model family
        models = repo.configs()
        models_only_c1_or_r = []
        for m in models:
            # RandomForest_r9_BAG_L1 -> r9
            suffix = m.replace("_BAG_L1", "").split("_")[-1]
            if suffix[0] == "r" or suffix == "c1":
                models_only_c1_or_r.append(m)
        models_only_c1_or_r = [m for m in models_only_c1_or_r if not "NeuralNetFastAI" in m]

        repo = repo.subset(
            configs=models_only_c1_or_r,
            force_to_dense=False,
            inplace=True,
        )
        return repo
    else:
        return repo
