from pathlib import Path
from typing import List, Union

from .context import BenchmarkContext, BenchmarkPaths
from ..loaders import Paths
from ..simulation.repository import EvaluationRepository


# TODO: Allow BenchmarkSubcontext as a parent
class BenchmarkSubcontext:
    def __init__(self,
                 parent: BenchmarkContext,
                 *,
                 path: Union[str, Path] = None,
                 name: str = None,
                 folds: List[int] = None,
                 configs: List[str] = None,
                 datasets: List[Union[int, str]] = None,
                 problem_type: Union[str, List[str]] = None):

        self.parent = parent
        self.folds = folds
        self.configs = configs
        self.datasets = datasets
        self.problem_type = problem_type
        if name is None:
            if path is not None:
                name = path
            else:
                name_suffix = ''
                if self.datasets is not None:
                    name_suffix += f'_D{len(self.datasets)}'
                if self.folds is not None:
                    name_suffix += f'_F{len(self.folds)}'
                if self.configs is not None:
                    name_suffix += f'_C{len(self.configs)}'
                if len(name_suffix) > 0:
                    name_suffix = '_SUBSET' + name_suffix
                name = parent.name + name_suffix
        self.name = name
        if path is None:
            path_prefix = Paths.data_root / 'repos'
            path_suffix = '.pkl'
            path_main = self.name
            path = path_prefix / (path_main + path_suffix)
        self.path = path

    def download(self, exists: str = 'raise') -> EvaluationRepository:
        assert exists in ['ignore', 'raise', 'overwrite'], f'Invalid exists value: {exists}'
        _exists = self.exists()
        if exists == 'ignore':
            if _exists:
                return self._load()
        elif exists == 'raise':
            if _exists:
                raise AssertionError(f'{self.path} already exists, but exists="{exists}"')
        return self._download()

    def _download(self) -> EvaluationRepository:
        repo = self.load_from_parent()
        repo.save(self.path)
        return repo

    def load_from_parent(self) -> EvaluationRepository:
        # TODO: Consider adding configs_full to Repo
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = self.parent.load(load_predictions=True)

        repo = EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=zeroshot_pred_proba,
            ground_truth=zeroshot_gt,
        )
        repo = repo.subset(folds=self.folds,
                           models=self.configs,
                           datasets=self.datasets)
        return repo

    def exists(self):
        return BenchmarkPaths.exists(self.path)

    def load(self, download_files=True, exists='ignore') -> EvaluationRepository:
        if not self.exists():
            if not download_files:
                raise FileNotFoundError(f'Missing file: "{self.path}", try calling `load` with `download_files=True`')
            print(f'Downloading subcontext {self.name}...')
            return self.download(exists=exists)
        return self._load()

    def _load(self) -> EvaluationRepository:
        return EvaluationRepository.load(self.path)
