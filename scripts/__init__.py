from pathlib import Path

from autogluon_zeroshot.utils.cache import cache_function
from autogluon_zeroshot.repository.evaluation_repository import load, EvaluationRepository

output_path = Path(__file__).parent

def load_context(version: str = "BAG_D244_F1_C1416") -> EvaluationRepository:
    def _load_fun():
        repo = load(version=version)
        repo = repo.subset(models=[m for m in repo.list_models() if not "NeuralNetFastAI" in m])
        return repo
    return cache_function(_load_fun, cache_name=f"repo_{version}")
