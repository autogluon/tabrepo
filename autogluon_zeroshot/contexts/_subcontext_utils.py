import copy
import random
from typing import Callable, List

from autogluon_zeroshot.repository import EvaluationRepositoryZeroshot
from autogluon_zeroshot.utils.cache import cache_function


def gen_sample_repo_exact(
        repo: EvaluationRepositoryZeroshot,
        folds: List[int] = None,
        datasets: List[int] = None,
        models: List[str] = None,
) -> EvaluationRepositoryZeroshot:
    return copy.deepcopy(repo).subset(folds=folds, datasets=datasets, models=models)


def gen_sample_repo(
        fun: Callable[[], EvaluationRepositoryZeroshot],
        n_models: int = None,
        n_folds: int = None,
        n_datasets: int = None,
        random_seed: int = 0,
) -> EvaluationRepositoryZeroshot:
    repo = fun()
    models = repo.list_models()
    datasets = [repo.taskid_to_dataset(taskid) for taskid in repo.get_datasets()]
    folds_sample = None
    if n_folds is not None:
        folds_sample = list(range(n_folds))

    random.seed(random_seed)
    tid_sample = None
    if n_datasets is not None:
        datasets_sample = random.sample(datasets, n_datasets)
        tid_sample = [repo.dataset_to_taskid(d) for d in datasets_sample]

    random.seed(random_seed + 1)
    models_sample = None
    if n_models is not None:
        models_sample = random.sample(models, n_models)

    return gen_sample_repo_exact(
        repo=repo,
        folds=folds_sample,
        datasets=tid_sample,
        models=models_sample,
    )


def gen_sample_repo_with_cache(
        fun: Callable[[], EvaluationRepositoryZeroshot],
        cache_name_prefix: str,
        *,
        n_folds: int = None,
        n_datasets: int = None,
        n_models: int = None,
        random_seed: int = 0,
        ignore_cache: bool = False,
) -> EvaluationRepositoryZeroshot:
    f"""
    Generate and cache a subset of a EvaluationRepository.
    Future calls will reload the cache, which is uniquely identified by the automatically generated cache file name.

    @param fun: The function call that will load the original repository prio to calling subset.
    @param cache_name_prefix: The name to use as the prefix of the cache name.
    @param n_folds: The number of folds to subset to. If None, will use all folds.
    @param n_datasets: The number of datasets to subset to. If None, will use all datasets.
    @param n_models: The number of models to subset to. If None, will use all models.
    @param random_seed: The random seed used when randomly selecting {n_datasets} and {n_models}.
    @param ignore_cache: If True, will compute the repo subset even if the cache already existed, and will overwrite it.
    @return: An EvaluationRepository object that is subset according to the parameters, and cached to disk.
    """
    cache_name = cache_name_prefix
    if n_datasets:
        cache_name += f'_D{n_datasets}'
    if n_folds:
        cache_name += f'_F{n_folds}'
    if n_models:
        cache_name += f'_C{n_models}'
    cache_name += f'_S{random_seed}'
    repo: EvaluationRepositoryZeroshot = cache_function(lambda: gen_sample_repo(
        fun=fun,
        n_folds=n_folds,
        n_models=n_models,
        n_datasets=n_datasets,
        random_seed=random_seed,
    ), cache_name=cache_name, ignore_cache=ignore_cache)
    return repo


if __name__ == '__main__':
    repo: EvaluationRepositoryZeroshot = gen_sample_repo_with_cache(
        n_folds=2,
        n_models=20,
        n_datasets=10,
    )

    import pickle
    import sys

    size_bytes = sys.getsizeof(pickle.dumps(repo, protocol=4))
    print(f'NEW repo Size: {round(size_bytes / 1e6, 3)} MB')

    cv = repo.simulate_zeroshot(num_zeroshot=10)

    print(cv)
