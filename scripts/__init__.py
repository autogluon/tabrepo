from pathlib import Path

from tabrepo.utils.cache import cache_function
from tabrepo.repository.evaluation_repository import load, EvaluationRepository

output_path = Path(__file__).parent

def load_context(version: str = "D244_F3_C1416_200", filter_very_large_dataset: bool = True, ignore_cache: bool = False) -> EvaluationRepository:
    def _load_fun():
        repo = load(version=version)
        repo = repo.subset(configs=[m for m in repo.configs() if not "NeuralNetFastAI" in m])
        return repo.force_to_dense(verbose=True)
    repo: EvaluationRepository = cache_function(_load_fun, cache_name=f"repo_{version}", ignore_cache=ignore_cache)


    if filter_very_large_dataset:
        # For some reason, only 184 datasets are found from this list
        # from tabrepo.contexts.context_2023_08_21 import datasets
        # filtered_datasets = datasets[-200:]
        # tids = [repo.dataset_to_tid(d) for d in filtered_datasets if d in repo.dataset_names()]
        # return repo.subset(tids=[tid for tid in repo.tids() if tid in tids])

        # Therefore we set the list manually
        very_large_datasets = [
            "nyc-taxi-green-dec-2016",
            "volkert",
            # "CIFAR_10",  # This dataset is not present
            "Fashion-MNIST",
            "Kuzushiji-MNIST",
            "mnist_784",
            "fars",
            "tamilnadu-electricity",
            "albert",
            "airlines",
            "ldpa",
            "porto-seguro",
        ]
        very_large_datasets = [d for d in very_large_datasets if d in repo.datasets()]

        # only pick models that have suffix rXX or c1
        models = repo.configs()
        models_only_c1_or_r = []
        for m in models:
            # RandomForest_r9_BAG_L1 -> r9
            suffix = m.replace("_BAG_L1", "").split("_")[-1]
            if suffix[0] == "r" or suffix == "c1":
                models_only_c1_or_r.append(m)

        return repo.subset(
            datasets=very_large_datasets,
            configs=models_only_c1_or_r,
        )
    else:
        return repo
