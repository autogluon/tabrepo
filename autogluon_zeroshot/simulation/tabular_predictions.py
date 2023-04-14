import copy
from collections import defaultdict
import json
import pickle
import shutil
import sys
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers.save_pkl import save as save_pkl

from .task_predictions import ConfigPredictionsDict, TaskModelPredictions, TaskModelPredictionsOpt

# dictionary mapping a particular fold of a dataset (a task) to split to config name to predictions
TaskPredictionsDict = Dict[str, ConfigPredictionsDict]

# dictionary mapping the folds of a dataset to split to config name to predictions
DatasetPredictionsDict = Dict[int, TaskPredictionsDict]

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = Dict[str, DatasetPredictionsDict]


class TabularModelPredictions:
    """
    Class that allows to query offline predictions.
    """

    def predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        """
        :param dataset:
        :param fold:
        :param splits: split to consider values must be in 'val' or 'test'
        :param models: list of models to be evaluated, by default uses all models available
        :return: for each split, a tensor with shape (num_models, num_points) for regression and
        (num_models, num_points, num_classes) for classification corresponding the predictions of the model.
        """
        if splits is None:
            splits = ['val', 'test']
        for split in splits:
            assert split in ['val', 'test']
        assert models is None or len(models) > 0
        return self._predict(dataset, fold, splits, models)

    def print_summary(self):
        folds = self.folds
        datasets = self.datasets
        tasks = self.tasks
        models = self.models

        num_folds = len(folds)
        num_datasets = len(datasets)
        num_tasks = len(tasks)
        num_models = len(models)

        folds_dense = self.get_folds_dense()
        models_dense = self.get_models_dense()

        num_folds_dense = len(folds_dense)
        num_models_dense = len(models_dense)

        is_dense = self.is_dense()
        is_dense_folds = self.is_dense_folds()
        is_dense_models = self.is_dense_models()

        print(f'Summary of {self.__class__.__name__}:\n'
              f'\tdatasets={num_datasets}\t| folds={num_folds} (dense={num_folds_dense})\t| tasks={num_tasks}\t'
              f'| models={num_models} (dense={num_models_dense})\n'
              f'\tis_dense={is_dense} | is_dense_folds={is_dense_folds} | is_dense_models={is_dense_models}')

    @property
    def models(self) -> List[str]:
        """
        :return: list of models present in at least one dataset.
        """
        raise NotImplementedError()

    def get_models(self, present_in_all=False) -> List[str]:
        """
        Gets all valid models
        :param present_in_all:
            If True, only returns models present in every dataset (dense)
            If False, returns every model that appears in at least 1 dataset (sparse)
        """
        if not present_in_all:
            return self.models
        else:
            return self.get_models_dense()

    def get_models_dense(self) -> List[str]:
        """
        Returns models that appears in all lists, eg that are available for all tasks and splits
        """
        models = []
        for dataset in self.datasets:
            models_in_dataset = set(self.models_available_in_dataset(dataset=dataset, present_in_all=True))
            models.append(models_in_dataset)
        if models:
            return sorted(list(set.intersection(*map(set, models))))
        else:
            return []

    def is_dense_models(self) -> bool:
        """
        Return True if all tasks have all models
        """
        models_dense = self.get_models(present_in_all=True)
        models_sparse = self.get_models(present_in_all=False)
        return set(models_dense) == set(models_sparse)

    def is_dense_folds(self) -> bool:
        """
        Return True if all datasets have all folds
        """
        return set(self.folds) == set(self.get_folds_dense())

    def is_dense(self) -> bool:
        """
        Return True if all datasets have all folds, and all tasks have all models
        """
        return self.is_dense_folds() and self.is_dense_models()

    def is_empty(self) -> bool:
        """
        Return True if no models or datasets exist
        """
        return len(self.datasets) == 0 or len(self.get_models(present_in_all=False)) == 0

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        raise NotImplementedError()

    def get_task(self, dataset: str, fold: int) -> TaskPredictionsDict:
        return self.get_dataset(dataset=dataset)[fold]

    def _check_dataset_exists(self, dataset: str) -> bool:
        """
        Simple implementation to check if a dataset exists.
        Consider implementing optimized version in inheriting classes if this is time-consuming.
        """
        return dataset in self.datasets

    def _check_task_exists(self, dataset: str, fold: int) -> bool:
        """
        Simple implementation to check if a task exists.
        Consider implementing optimized version in inheriting classes if this is time-consuming.
        """
        try:
            self.get_task(dataset=dataset, fold=fold)
            return True
        except:
            return False

    def models_available_in_task(self,
                                 *,
                                 task: Optional[TaskPredictionsDict] = None,
                                 dataset: Optional[str] = None,
                                 fold: Optional[int] = None,
                                 split: Optional[str] = None) -> List[str]:
        """
        Get list of valid models for a given task

        Either task must be specified or dataset & fold must be specified.

        If 'split' is not None, will only check for the given split.
        If 'split' is None, will return models that are present in every split (dense).
        """
        if task is not None and isinstance(task, tuple):
            dataset = task[0]
            fold = task[1]
            task = None
        if task is None:
            assert dataset is not None
            assert fold is not None
            if self._check_task_exists(dataset=dataset, fold=fold):
                task = self.get_task(dataset=dataset, fold=fold)
            else:
                return []
        else:
            assert dataset is None
            assert fold is None
        if split is not None:
            models = list(task[split].keys())
        else:
            splits = task.keys()
            models = [set(task[split]) for split in splits]
            models = list(set.intersection(*map(set, models)))
        return models

    def models_available_in_task_dict(self) -> Dict[str, Dict[int, List[str]]]:
        """Get dict of valid models per task"""
        datasets = self.datasets

        model_fold_dataset_dict = dict()
        for d in datasets:
            dataset_predictions = self.get_dataset(dataset=d)
            model_fold_dataset_dict[d] = dict()
            for f in dataset_predictions:
                model_fold_dataset_dict[d][f] = self.models_available_in_task(task=dataset_predictions[f])
        return model_fold_dataset_dict

    def models_available_in_dataset(self, dataset: str, present_in_all: bool = True) -> List[str]:
        """Returns the models available on both validation and test splits on all tasks in a dataset"""
        models = []
        dataset_predictions = self.get_dataset(dataset=dataset)
        for fold in dataset_predictions:
            task_predictions = dataset_predictions[fold]
            models.append(set(self.models_available_in_task(task=task_predictions)))
        # returns models that appears in all lists, eg that are available for all folds and splits
        if present_in_all:
            return sorted(list(set.intersection(*map(set, models))))
        else:
            all_models = set()
            for model_set in models:
                all_models = all_models.union(model_set)
            return sorted(list(all_models))

    def folds_available_in_dataset(self, dataset: str) -> List[int]:
        """Returns the folds available in a dataset"""
        dataset_predictions = self.get_dataset(dataset=dataset)
        return sorted(list(dataset_predictions.keys()))

    @property
    def folds(self) -> List[int]:
        """
        Returns all folds that appear at least once in any dataset (sparse)
        """
        folds = set()
        for dataset in self.datasets:
            for f in self.folds_available_in_dataset(dataset=dataset):
                folds.add(f)
        return sorted(list(folds))

    def get_datasets_with_folds(self, folds: List[int]) -> List[str]:
        """
        Get list of datasets that have results for all input folds
        """
        datasets = self.datasets
        valid_datasets = []
        for dataset in datasets:
            folds_in_dataset = self.folds_available_in_dataset(dataset=dataset)
            if all(f in folds_in_dataset for f in folds):
                valid_datasets.append(dataset)
        return valid_datasets

    def get_datasets_with_models(self, models: List[str]) -> List[str]:
        """
        Get list of datasets that have results for all input models
        """
        datasets = self.datasets
        configs = set(models)
        valid_datasets = []
        for d in datasets:
            models_in_dataset = set(self.models_available_in_dataset(dataset=d, present_in_all=True))
            is_valid = True
            for m in configs:
                if m not in models_in_dataset:
                    is_valid = False
            if is_valid:
                valid_datasets.append(d)
        return valid_datasets

    def restrict_models(self, models: List[str]):
        """
        :param models: restricts the predictions to contain only the list of models given in arguments, useful to
        reduce memory footprint. The behavior depends on the data structure used. For pickle/full data structure,
        the data is immediately sliced. For lazy representation, the data is sliced on the fly when querying predictions.
        """
        models_present = self.models
        for m in models:
            assert m in models_present, f"cannot restrict {m} which is not in available models {models_present}."
        self._restrict_models(models)

    def _restrict_models(self, models: List[str]):
        raise NotImplementedError()

    def restrict_datasets(self, datasets: List[str]):
        for d in datasets:
            assert d in self.datasets, f"cannot restrict {d} which is not in available datasets {self.datasets}."
        self._restrict_datasets(datasets)

    def _restrict_datasets(self, datasets: List[str]):
        for dataset in self.datasets:
            if dataset not in datasets:
                self.remove_dataset(dataset)

    def restrict_folds(self, folds: List[int]):
        folds_cur = self.folds
        for f in folds:
            assert f in folds_cur, f"Trying to restrict to a fold {f} that does not exist! Valid folds: {folds_cur}."
        return self._restrict_folds(folds=folds)

    def _restrict_folds(self, folds: List[int]):
        raise NotImplementedError()

    def force_to_dense(self,
                       first_prune_method: str = 'task',
                       second_prune_method: str = 'dataset',
                       assert_not_empty: bool = True):
        """
        Force to be dense in all dimensions.
        This means all models will be present in all tasks, and all folds will be present in all datasets.
        # TODO: Not guaranteed to be dense if first_prune_method = 'dataset'
        """
        if first_prune_method in ['dataset', 'fold']:
            first_method = self.force_to_dense_folds
            second_method = self.force_to_dense_models
        else:
            first_method = self.force_to_dense_models
            second_method = self.force_to_dense_folds
        print(
            f'Forcing {self.__class__.__name__} to dense representation via two-stage filtering using '
            f'`first_prune_method="{first_prune_method}"`, `second_prune_method="{second_prune_method}"`...')
        first_method(prune_method=first_prune_method, assert_not_empty=assert_not_empty)
        second_method(prune_method=second_prune_method, assert_not_empty=assert_not_empty)

        print(f'The {self.__class__.__name__} object is now guaranteed to be dense.')
        assert self.is_dense()

    def force_to_dense_folds(self, prune_method: str = 'dataset', assert_not_empty: bool = True):
        """
        Force the pred dict to contain only dense fold results (no missing folds for any dataset)
        :param prune_method:
            If 'dataset', prunes any dataset that doesn't contain results for all folds
            If 'fold', prunes any fold that doesn't exist for all datasets
        """
        print(f'Forcing {self.__class__.__name__} to dense fold representation using `prune_method="{prune_method}"`...')
        valid_prune_methods = ['dataset', 'fold']
        if prune_method not in valid_prune_methods:
            raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
        pre_num_models = len(self.models)
        pre_num_datasets = len(self.datasets)
        pre_num_folds = len(self.folds)
        if prune_method == 'dataset':
            datasets_dense = self.get_datasets_with_folds(folds=self.folds)
            self.restrict_datasets(datasets=datasets_dense)
        elif prune_method == 'fold':
            folds_dense = self.get_folds_dense()
            self.restrict_folds(folds=folds_dense)
        else:
            raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
        post_num_models = len(self.models)
        post_num_datasets = len(self.datasets)
        post_num_folds = len(self.folds)

        print(f'\tPre : datasets={pre_num_datasets} | models={pre_num_models} | folds={pre_num_folds}')
        print(f'\tPost: datasets={post_num_datasets} | models={post_num_models} | folds={post_num_folds}')
        assert self.is_dense_folds()
        if assert_not_empty:
            assert not self.is_empty()

    def force_to_dense_models(self, prune_method: str = 'task', assert_not_empty: bool = True):
        """
        Force the pred dict to contain only dense results (no missing result for any task/model)
        :param prune_method:
            If 'task', prunes any task that doesn't contain results for all models
            If 'model', prunes any model that doesn't have results for all tasks
        """
        print(f'Forcing {self.__class__.__name__} to dense model representation using `prune_method="{prune_method}"`...')
        valid_prune_methods = ['task', 'model']
        if prune_method not in valid_prune_methods:
            raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
        datasets = self.datasets
        valid_models = self.get_models(present_in_all=False)
        pre_num_models = len(valid_models)
        pre_num_datasets = len(datasets)
        pre_num_folds = len(self.folds)
        if prune_method == 'task':
            valid_tasks = []
            for task in self.tasks:
                dataset = task[0]
                fold = task[1]
                models_in_task = self.models_available_in_task(dataset=dataset, fold=fold)
                models_in_task_set = set(models_in_task)
                if all(m in models_in_task_set for m in valid_models):
                    valid_tasks.append(task)
            self.restrict_tasks(tasks=valid_tasks)
        elif prune_method == 'model':
            valid_models = self.get_models(present_in_all=True)
            self.restrict_models(models=valid_models)
        else:
            raise ValueError(f'`prune_method={prune_method}` is invalid. Valid values: {valid_prune_methods}')
        post_num_models = len(self.models)
        post_num_datasets = len(self.datasets)
        post_num_folds = len(self.folds)

        print(f'\tPre : datasets={pre_num_datasets} | models={pre_num_models} | folds={pre_num_folds}')
        print(f'\tPost: datasets={post_num_datasets} | models={post_num_models} | folds={post_num_folds}')
        assert self.is_dense_models()
        if assert_not_empty:
            assert not self.is_empty()

    def restrict_tasks(self, tasks: List[Tuple[str, int]]):
        """
        Filter ta only tasks in `tasks`.

        tasks is a list of (dataset, fold) pairs, where (dataset, fold) represents a particular task.
        """
        valid_task_dict = defaultdict(set)
        for task in tasks:
            dataset = task[0]
            fold = task[1]
            valid_task_dict[dataset].add(fold)
        for task in self.tasks:
            dataset = task[0]
            fold = task[1]
            if fold not in valid_task_dict[dataset]:
                self.remove_task(dataset=dataset, fold=fold)

    def get_folds_dense(self) -> List[int]:
        """
        Returns folds that appear in all datasets
        """
        folds = []
        for dataset in self.datasets:
            folds_in_dataset = set(self.folds_available_in_dataset(dataset=dataset))
            folds.append(folds_in_dataset)
        if folds:
            return sorted(list(set.intersection(*map(set, folds))))
        else:
            return []

    def remove_dataset(self, dataset: str):
        raise NotImplementedError()

    def remove_task(self, dataset: str, fold: int, error_if_missing=True):
        raise NotImplementedError()

    @property
    def datasets(self) -> List[str]:
        raise NotImplementedError()

    @property
    def tasks(self) -> List[Tuple[str, int]]:
        """
        Returns list of all tasks in (dataset, fold) tuple form.
        """
        tasks = []
        for dataset in self.datasets:
            folds = self.folds_available_in_dataset(dataset)
            for fold in folds:
                tasks.append((dataset, fold))
        return tasks

    @staticmethod
    def _get_task_from_dataset(dataset_predictions: DatasetPredictionsDict, fold: int) -> TaskPredictionsDict:
        return dataset_predictions[fold]

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        """
        :param pred_dict: dictionary mapping dataset to fold to split to config name to predictions
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filename: str):
        raise NotImplementedError()

    def save(self, filename: str):
        raise NotImplementedError()

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        raise NotImplementedError()


class TabularPicklePredictions(TabularModelPredictions):
    def __init__(self, pred_dict: TabularPredictionsDict):
        self.pred_dict = pred_dict

    @classmethod
    def load(cls, filename: str):
        return cls(pred_dict=load_pkl.load(filename))

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        return cls(pred_dict=pred_dict)

    def save(self, filename: str):
        save_pkl(filename, self.pred_dict)

    def _get_model_results(self, model: str, model_pred_probas: dict) -> np.array:
        return model_pred_probas[model]

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        task = self.get_task(dataset=dataset, fold=fold)

        if models is None:
            models = self.models_available_in_task(task=task)

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = task[split_key]
            return np.array([self._get_model_results(model=model, model_pred_probas=model_results) for model in models])

        return [get_split(split, models) for split in splits]

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        return self.pred_dict[dataset]

    @property
    def models(self) -> List[str]:
        """
        Returns models that appear at least once
        """
        models = set()
        for dataset in self.datasets:
            models = models.union(set(self.models_available_in_dataset(dataset, present_in_all=False)))
        return sorted(list(models))

    def _restrict_models(self, models: List[str]):
        configs = set(models)
        tasks = self.tasks
        for task_tuple in tasks:
            task = self.get_task(dataset=task_tuple[0], fold=task_tuple[1])
            models_in_task = self.models_available_in_task(task=task)
            for m in models_in_task:
                if m not in configs:
                    for split in task:
                        task[split].pop(m, None)
            if not self.models_available_in_task(task=task):
                self.remove_task(dataset=task_tuple[0], fold=task_tuple[1])

    def _restrict_folds(self, folds: List[int]):
        folds_cur = self.folds
        valid_folds_set = set(folds)
        size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
        print(f'Restricting Folds... (Shrinking from {len(folds_cur)} -> {len(valid_folds_set)} folds)')
        print(f'OLD zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')
        folds_to_remove = [f for f in folds_cur if f not in valid_folds_set]
        for f in folds_to_remove:
            self.remove_fold(f)
        size_bytes = sys.getsizeof(pickle.dumps(self.pred_dict, protocol=4))
        print(f'NEW zeroshot_pred_proba Size: {round(size_bytes / 1e6, 3)} MB')

    @property
    def datasets(self) -> List[str]:
        return list(self.pred_dict.keys())

    def remove_fold(self, fold: int):
        for dataset in self.datasets:
            self.remove_task(dataset=dataset, fold=fold, error_if_missing=False)

    def remove_dataset(self, dataset: str):
        self.pred_dict.pop(dataset)

    def remove_task(self, dataset: str, fold: int, error_if_missing=True):
        if error_if_missing:
            self.pred_dict[dataset].pop(fold)
        else:
            self.pred_dict[dataset].pop(fold, None)
        if len(self.folds_available_in_dataset(dataset=dataset)) == 0:
            self.remove_dataset(dataset=dataset)

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        num_datasets = len(self.datasets)
        self.pred_dict = {rename_dict.get(k, k): v for k, v in self.pred_dict.items()}
        num_datasets_post = len(self.datasets)
        if num_datasets_post != num_datasets:
            raise AssertionError(f'Renaming caused a dataset name conflict! '
                                 f'Started with {num_datasets} datasets, ended with {num_datasets_post} datasets... '
                                 f'(rename_dict={rename_dict})')


class TabularPicklePerTaskPredictions(TabularModelPredictions):
    metadata_filename = 'metadata.pkl'
    # TODO: Consider saving/loading at the task level rather than the dataset level
    def __init__(self,
                 tasks_to_models: Dict[str, Dict[int, List[str]]],
                 output_dir: str,
                 rename_dict_inv: Dict[str, str] = None,
                 ):
        """
        Stores on pickle per task and load data in a lazy fashion which allows to reduce significantly the memory
        footprint.
        :param tasks_to_models:
        :param output_dir:
        """
        self.tasks_to_models = tasks_to_models
        self.output_dir = Path(output_dir)
        if rename_dict_inv is None:
            rename_dict_inv = {}
        self.rename_dict_inv = rename_dict_inv
        assert self.output_dir.is_dir()
        for f in self.folds:
            assert isinstance(f, int)

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        pred_dict = self._load_dataset(dataset)
        models_valid = self.models_available_in_dataset(dataset)
        if models is None:
            models = models_valid
        else:
            models_valid_set = set(models_valid)
            for m in models:
                assert m in models_valid_set, f"Model {m} is not valid for dataset {dataset} on fold {fold}! " \
                                              f"Valid models: {models_valid}"

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = pred_dict[fold][split_key]
            return np.array([model_results[model] for model in models])

        available_model_mask = np.array([i for i, model in enumerate(models)])
        return [get_split(split, models)[available_model_mask] for split in splits]

    def models_available_in_dataset(self, dataset: str, present_in_all=True) -> List[str]:
        models = []
        dataset_task_models = self.tasks_to_models[dataset]
        for fold in dataset_task_models:
            task_models = dataset_task_models[fold]
            models.append(set(task_models))
        if present_in_all:
            models = sorted(list(set.intersection(*map(set, models))))
        else:
            all_models = set()
            for model_set in models:
                all_models = all_models.union(model_set)
            models = sorted(list(all_models))
        return models

    def folds_available_in_dataset(self, dataset: str) -> List[int]:
        """Returns the folds available in a dataset"""
        dataset_fold_dict = self.tasks_to_models[dataset]
        return sorted(list(dataset_fold_dict.keys()))

    def get_dataset(self, dataset: str) -> DatasetPredictionsDict:
        return self._load_dataset(dataset=dataset)

    def _load_dataset(self, dataset: str, enforce_folds: bool = True) -> DatasetPredictionsDict:
        dataset_file_name = self.rename_dict_inv.get(dataset, dataset)
        filename = str(self.output_dir / f'{dataset_file_name}.pkl')
        out = load_pkl.load(filename)
        if enforce_folds:
            valid_folds = set(self.tasks_to_models[dataset])
            folds = list(out.keys())
            for f in folds:
                if f not in valid_folds:
                    out.pop(f)
        return out

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        task_to_models = pred_proba.models_available_in_task_dict()
        print(f"saving .pkl files in folder {output_dir}")
        for dataset in tqdm(datasets):
            filename = str(output_dir / f'{dataset}.pkl')
            save_pkl(filename, pred_dict[dataset])
        cls._save_metadata(output_dir=output_dir, tasks_to_models=task_to_models)
        return cls(tasks_to_models=task_to_models, output_dir=output_dir)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"copy .pkl files from {self.output_dir} to {output_dir}")
        # FIXME: Only copy the required files, not all files
        for file in self.output_dir.glob("*.pkl"):
            shutil.copyfile(file, output_dir / file.name)
        self._save_metadata(output_dir=output_dir,
                            tasks_to_models=self.tasks_to_models,
                            rename_dict_inv=self.rename_dict_inv)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        tasks_to_models = metadata["tasks_to_models"]
        rename_dict_inv = metadata["rename_dict_inv"]

        return cls(
            tasks_to_models=tasks_to_models,
            output_dir=filename,
            rename_dict_inv=rename_dict_inv,
        )

    @property
    def datasets(self):
        return list(self.tasks_to_models.keys())

    def _restrict_models(self, models: List[str]):
        models_to_keep = set(models)
        datasets = self.datasets
        for dataset in datasets:
            folds = list(self.tasks_to_models[dataset].keys())
            for fold in folds:
                models_in_task = self.tasks_to_models[dataset][fold]
                models_in_task = [m for m in models_in_task if m in models_to_keep]
                if not models_in_task:
                    self.remove_task(dataset=dataset, fold=fold)
                else:
                    self.tasks_to_models[dataset][fold] = models_in_task

    def _restrict_folds(self, folds: List[int]):
        valid_folds_set = set(folds)
        datasets = self.datasets
        for dataset in datasets:
            folds_in_dataset = self.folds_available_in_dataset(dataset)
            for fold in folds_in_dataset:
                if fold not in valid_folds_set:
                    self.remove_task(dataset=dataset, fold=fold)

    def remove_dataset(self, dataset: str):
        self.tasks_to_models.pop(dataset)
        self.rename_dict_inv.pop(dataset, None)

    def remove_task(self, dataset: str, fold: int, error_if_missing=True):
        if error_if_missing:
            self.tasks_to_models[dataset].pop(fold)
        else:
            self.tasks_to_models[dataset].pop(fold, None)
        if len(self.folds_available_in_dataset(dataset=dataset)) == 0:
            self.remove_dataset(dataset=dataset)

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        num_datasets = len(self.datasets)
        datasets = self.datasets

        for k_new, v_new in rename_dict.items():
            if k_new not in self.rename_dict_inv:
                self.rename_dict_inv[k_new] = k_new
        self.rename_dict_inv = {
            rename_dict.get(dataset, dataset): file_name for dataset, file_name in self.rename_dict_inv.items()
        }
        self.tasks_to_models = {
            rename_dict.get(dataset, dataset): self.tasks_to_models[dataset] for dataset in datasets
        }
        num_datasets_post = len(self.datasets)
        if num_datasets_post != num_datasets:
            raise AssertionError(f'Renaming caused a dataset name conflict! '
                                 f'Started with {num_datasets} datasets, ended with {num_datasets_post} datasets... '
                                 f'(rename_dict={rename_dict})')

    @classmethod
    def _save_metadata(cls, output_dir: Path, tasks_to_models: dict, rename_dict_inv: Dict[str, str] = None):
        # FIXME: Cant use json because json will store keys as string, even when they were integer (for example, folds)
        metadata = {
            "tasks_to_models": tasks_to_models,
            "rename_dict_inv": rename_dict_inv,
        }
        save_pkl(path=str(Path(output_dir) / cls.metadata_filename), object=metadata)

    @classmethod
    def _load_metadata(cls, output_dir: Path) -> dict:
        return load_pkl.load(path=str(Path(output_dir) / cls.metadata_filename))

    @property
    def models(self) -> List[str]:
        res = set()
        for d in self.tasks_to_models.keys():
            for f in self.tasks_to_models[d].keys():
                for model in self.tasks_to_models[d][f]:
                    res.add(model)
        return sorted(list(res))


# TODO: This might not work correctly. Haven't tested it.
class TabularNpyPerTaskPredictions(TabularModelPredictions):
    metadata_filename = 'metadata.pkl'

    def __init__(self, output_dir: str, dataset_shapes: Dict[str, Tuple[int, int, int]], models, folds):
        self._output_dir = output_dir
        self._dataset_shapes = dataset_shapes
        self._models = models
        self._folds = folds
        self.models_removed = set()

    def _predict_from_dataset(self, dataset: str) -> np.array:
        evals = np.load(Path(self._output_dir) / f"{dataset}.npy")
        num_val, num_test, output_dim = self._dataset_shapes[dataset]
        assert evals.shape[0] == num_val + num_test
        assert evals.shape[-1] == output_dim
        # (num_train/num_test, n_folds, n_models, output_dim)
        return evals[:num_val], evals[num_val:]

    def search_index(self, l, x):
        for i, y in enumerate(l):
            if y == x:
                return i

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        """
        :return: for each split, a tensor with shape (num_models, num_points) for regression and
        (num_models, num_points, num_classes) for classification corresponding the predictions of the model.
        """
        model_indices = {model: i for i, model in enumerate(self.models)}
        res = []
        # (num_train/num_test, n_folds, n_models, output_dim)
        val_evals, test_evals = self._predict_from_dataset(dataset)
        for split in splits:
            tensor = val_evals if split == "val" else test_evals
            if models is None:
                res.append(tensor[:, fold, :])
            else:
                res.append(tensor[:, fold, [model_indices[m] for m in models]])

        res = [np.swapaxes(x, 0, 1) for x in res]
        # squeeze last dim to be uniform with other part of the code
        res = [
            np.squeeze(x, axis=-1) if x.shape[-1] == 1 else x
            for x in res
        ]
        return res

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        models = pred_proba.models
        print(f"saving .pkl files in folder {output_dir}")
        dataset_shapes = {}
        for dataset in datasets:
            # (num_samples, n_folds, n_models, output_dim)
            val_evals, test_evals = cls._stack_pred(pred_dict[dataset], models)
            dataset_shapes[dataset] = (len(val_evals), len(test_evals), val_evals.shape[-1])
            evals = np.concatenate([val_evals, test_evals], axis=0)
            np.save(output_dir / f"{dataset}.npy", evals)
        cls._save_metadata(
            output_dir=output_dir,
            dataset_shapes=dataset_shapes,
            models=models,
            folds=pred_proba.folds,
        )

        return cls(
            dataset_shapes=dataset_shapes,
            output_dir=output_dir,
            models=models,
            folds=pred_proba.folds,
        )

    @staticmethod
    def _stack_pred(fold_dict: Dict[int, Dict[str, Dict[str, np.array]]], models):
        """
        :param fold_dict: dictionary mapping fold to split to config name to predictions
        :return:
        """
        # split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
        num_samples_val = min(len(config_evals) for config_evals in fold_dict[0]["pred_proba_dict_val"].values())
        num_samples_test = min(len(config_evals) for config_evals in fold_dict[0]["pred_proba_dict_test"].values())
        output_dims = set(
            config_evals.shape[1] if config_evals.ndim > 1 else 1
            for fold in fold_dict.values()
            for split in fold.values()
            for config_evals in split.values()
        )
        assert len(output_dims) == 1
        output_dim = next(iter(output_dims))
        n_folds = len(fold_dict)
        n_models = len(fold_dict[0]["pred_proba_dict_val"])
        val_res = np.zeros((num_samples_val, n_folds, n_models, output_dim))
        test_res = np.zeros((num_samples_test, n_folds, n_models, output_dim))
        def expand_if_scalar(x):
            return x if output_dim > 1 else np.expand_dims(x, axis=-1)

        for n_fold in range(n_folds):
            for n_model, model in enumerate(models):
                val_res[:, n_fold, n_model, :] = expand_if_scalar(
                    fold_dict[n_fold]["pred_proba_dict_val"][model][:num_samples_val]
                )
                test_res[:, n_fold, n_model, :] = expand_if_scalar(
                    fold_dict[n_fold]["pred_proba_dict_test"][model][:num_samples_test]
                )
        return val_res, test_res

    @classmethod
    def _save_metadata(cls, output_dir, dataset_shapes, models, folds):
        metadata = {
            "dataset_shapes": dataset_shapes,
            "models": models,
            "folds": folds,
        }
        save_pkl(path=str(Path(output_dir) / cls.metadata_filename), object=metadata)

    @classmethod
    def _load_metadata(cls, output_dir: Path) -> dict:
        return load_pkl.load(path=str(Path(output_dir) / cls.metadata_filename))

    @property
    def datasets(self) -> List[str]:
        return list(self._dataset_shapes.keys())

    def models_available_in_dataset(self, dataset: str) -> List[str]:
        return [m for m in self._models if m not in self.models_removed]

    @property
    def folds(self) -> List[int]:
        return self._folds

    @property
    def models(self) -> List[int]:
        return [m for m in self._models if m not in self.models_removed]

    def _restrict_models(self, models: List[str]):
        for model in self.models:
            if model not in models:
                self.models_removed.add(model)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, dataset_shapes=self._dataset_shapes, models=self._models, folds=self._folds)
        print(f"copy .npy files from {self._output_dir} to {output_dir}")
        for file in self._output_dir.glob("*.npy"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        return cls(
            output_dir=filename,
            **metadata
        )


class TabularPicklePredictionsOpt(TabularPicklePredictions):
    """
    A model predictions data representation optimized for `ray.put(self)` operations to minimize overhead.
    Ray has a large overhead when using a shared object with many numpy arrays (such as 500,000).
    This class converts many smaller numpy arrays into fewer larger numpy arrays,
    eliminating the vast majority of the overhead.

    """
    def __init__(self, pred_dict_opt: Dict[str, Dict[int, Dict[str, TaskModelPredictionsOpt]]]):
        self.pred_dict = pred_dict_opt

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        pred_dict_opt = cls.stack_pred_dict(pred_dict=pred_dict)
        return cls(pred_dict_opt=pred_dict_opt)

    @classmethod
    def stack_pred_dict(cls, pred_dict: TabularPredictionsDict) -> Dict[str, Dict[int, Dict[str, TaskModelPredictionsOpt]]]:
        pred_dict = copy.deepcopy(pred_dict)  # TODO: Avoid the deep copy, create from scratch to min mem usage
        datasets = list(pred_dict.keys())
        for dataset in datasets:
            folds = list(pred_dict[dataset].keys())
            for fold in folds:
                splits = list(pred_dict[dataset][fold].keys())
                for split in splits:
                    model_pred_probas: ConfigPredictionsDict = pred_dict[dataset][fold][split]
                    pred_dict[dataset][fold][split] = TaskModelPredictionsOpt.from_config_predictions(
                        config_predictions=model_pred_probas
                    )
        return pred_dict

    @classmethod
    def load(cls, filename: str):
        return cls(pred_dict_opt=load_pkl.load(filename))

    def save(self, filename: str):
        save_pkl(filename, self.pred_dict)

    def _get_model_results(self, model: str, model_pred_probas: TaskModelPredictionsOpt) -> np.array:
        return model_pred_probas.get_model_predictions(model=model)

    def _restrict_models(self, models: List[str]):
        task_names = self.datasets
        for t in task_names:
            available_folds = list(self.pred_dict[t].keys())
            for f in available_folds:
                available_splits = list(self.pred_dict[t][f].keys())
                for s in available_splits:
                    self.pred_dict[t][f][s] = self.pred_dict[t][f][s].subset(models=models, inplace=True)
                    if not self.pred_dict[t][f][s].models:
                        # If no models, then pop the entire task and go to the next task
                        self.pred_dict[t].pop(f)
                        break
            if not self.pred_dict[t]:
                # If no folds, then pop the entire dataset
                self.pred_dict.pop(t)

    def models_available_in_task(self,
                                 *,
                                 task: Optional[TaskPredictionsDict] = None,
                                 dataset: Optional[str] = None,
                                 fold: Optional[int] = None,
                                 split: str = None) -> List[str]:
        """
        Get list of valid models for a given task

        Either task must be specified or dataset & fold must be specified.

        If 'split' is not None, will only check for the given split.
        If 'split' is None, will return models that are present in every split (dense).
        """
        if task is None:
            assert dataset is not None
            assert fold is not None
            if self._check_task_exists(dataset=dataset, fold=fold):
                task = self.get_task(dataset=dataset, fold=fold)
            else:
                return []
        else:
            assert dataset is None
            assert fold is None
        if split is not None:
            models = list(task[split].models)
        else:
            splits = task.keys()
            models = [set(task[split].models) for split in splits]
            models = list(set.intersection(*map(set, models)))
        return models
