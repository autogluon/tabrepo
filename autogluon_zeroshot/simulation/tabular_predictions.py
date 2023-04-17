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


def filter_empty(dataset_dict: TabularPredictionsDict):
    # remove all possibly empty collections from the nested dictionary
    for dataset, folds in dataset_dict.items():
        for fold, splits in folds.items():
            dataset_dict[dataset][fold] = {split: models for split, models in splits.items() if models}
        dataset_dict[dataset] = {fold: splits for fold, splits in folds.items() if splits}
    return {dataset: folds for dataset, folds in dataset_dict.items() if folds}


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
            splits = self.splits
        for split in splits:
            assert split in self.splits
        assert models is None or len(models) > 0
        return self._predict(dataset=dataset, fold=fold, splits=splits, models=models)

    def predict_dataset(self, dataset: str) -> DatasetPredictionsDict:
        """
        :return: all the predictions associated to a dataset
        """
        raise NotImplementedError()

    def predict_task(self, dataset: str, fold: int) -> TaskPredictionsDict:
        """
        :return: all the predictions associated to a task
        """
        return self.predict_dataset(dataset=dataset)[fold]

    @property
    def splits(self) -> List[str]:
        return ['val', 'test']

    @property
    def folds(self) -> List[int]:
        """
        Returns all folds that appear at least once in any dataset (sparse)
        """
        return self.list_folds_available(present_in_all=False)

    @property
    def datasets(self) -> List[str]:
        raise NotImplementedError()

    @property
    def models(self) -> List[str]:
        return self.list_models_available(present_in_all=False)

    def list_folds_available(self, datasets: List[str] = None, present_in_all: bool = True) -> List[int]:
        """
        :return: the list of folds available in the datasets provided, if no dataset is given consider the list of
        all available datasets. Return the folds present in all datasets if `present_in_all` and otherwise the ones
        present in any dataset.
        """
        raise NotImplementedError()

    def list_models_available(
            self,
            datasets: Optional[List[str]] = None,
            folds: Optional[List[int]] = None,
            splits: Optional[List[str]] = None,
            present_in_all: bool = False,
    ) -> List[str]:
        """
        :return: the list of models available on the datasets/folds/splits specified. If a field is not specified
        then the list is computed over all elements of the collection. If `present_in_all` is set to True, then
        only models appearing in all datasets/folds/splits combinations are returned and else models appearing in any
        combination are returned.
        """
        res = []
        for dataset in datasets if datasets else self.datasets:
            for fold in folds if folds else self.list_folds_available(datasets=[dataset], present_in_all=False):
                for split in splits if splits else self.splits:
                    res.append(self.models_available_for_dataset_fold_split(dataset, fold, split))
        agg_fun = set.intersection if present_in_all else set.union
        return sorted(list(agg_fun(*map(set, res)))) if res else []

    def models_available_for_dataset_fold_split(self, dataset, fold, split) -> List[str]:
        """
        :return: the list of models available for a given dataset/fold/split 
        """
        raise NotImplementedError()

    def is_empty(self) -> bool:
        """
        Return True if no models exists
        """
        return len(self.models) == 0

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

    def remove_dataset(self, dataset: str):
        raise NotImplementedError()

    def restrict_tasks(self, tasks: List[Tuple[str, int]]):
        """
        Filter ta only tasks in `tasks`.

        tasks is a list of (dataset, fold) pairs, where (dataset, fold) represents a particular task.
        """
        valid_task_dict = defaultdict(set)
        for (dataset, fold) in tasks:
            valid_task_dict[dataset].add(fold)
        for (dataset, fold) in self.tasks:
            if fold not in valid_task_dict[dataset]:
                self.remove_task(dataset=dataset, fold=fold)

    def remove_task(self, dataset: str, fold: int, error_if_missing=True):
        raise NotImplementedError()

    @property
    def tasks(self) -> List[Tuple[str, int]]:
        """
        Returns list of all tasks in (dataset, fold) tuple form.
        """
        tasks = []
        for dataset in self.datasets:
            folds = self.list_folds_available(datasets=[dataset])
            for fold in folds:
                tasks.append((dataset, fold))
        return tasks

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
        task = self.predict_task(dataset=dataset, fold=fold)

        if models is None:
            models = self.models

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = task[split_key]
            return np.array([self._get_model_results(model=model, model_pred_probas=model_results) for model in models])

        return [get_split(split, models) for split in splits]

    def predict_dataset(self, dataset: str) -> DatasetPredictionsDict:
        return self.pred_dict[dataset]

    def models_available_for_dataset_fold_split(self, dataset, fold, split) -> List[str]:
        split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
        try:
            return self.pred_dict[dataset][fold][split_key].keys()
        except KeyError:
            return []

    def list_folds_available(self, datasets: List[str] = None, present_in_all: bool = True) -> List[int]:
        datasets = datasets if datasets else self.datasets
        all_folds = []
        for dataset in datasets:
            if dataset in self.pred_dict:
                if dataset in self.pred_dict:
                    all_folds.append(self.pred_dict[dataset].keys())

        agg_fun = set.intersection if present_in_all else set.union
        return sorted(list(agg_fun(*map(set, all_folds)))) if all_folds else []


    def _restrict_models(self, models: List[str]):
        models_to_keep = set(models)
        for (dataset, folds) in self.pred_dict.items():
            for (fold, splits) in folds.items():
                for (split, models) in splits.items():
                    self.pred_dict[dataset][fold][split] = {k: v for k, v in models.items() if k in models_to_keep}

        # remove collections that may now be empty
        self.pred_dict = filter_empty(self.pred_dict)


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
        if len(self.list_folds_available(datasets=[dataset])) == 0:
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
                 tasks_to_models: Dict[str, Dict[int, Dict[str, List[str]]]],
                 output_dir: str,
                 rename_dict_inv: Dict[str, str] = None,
                 ):
        """
        Stores on pickle per task and load data in a lazy fashion which allows to reduce significantly the memory
        footprint.
        :param tasks_to_models: dictionary mapping dataset to fold to split to model names
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

    def list_folds_available(self, datasets: List[str] = None, present_in_all: bool = True) -> List[int]:
        datasets = datasets if datasets else self.datasets
        all_folds = [self.tasks_to_models[dataset].keys() for dataset in datasets]
        agg_fun = set.intersection if present_in_all else set.union
        return sorted(list(agg_fun(*map(set, all_folds)))) if all_folds else []

    def models_available_for_dataset_fold_split(self, dataset, fold, split) -> List[str]:
        try:
            return self.tasks_to_models[dataset][fold][split]
        except KeyError:
            return []

    def _predict(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        pred_dict = self._load_dataset(dataset)
        models_valid = self.list_models_available(datasets=[dataset], present_in_all=True)
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

    def folds_available_in_dataset(self, dataset: str) -> List[int]:
        """Returns the folds available in a dataset"""
        dataset_fold_dict = self.tasks_to_models[dataset]
        return sorted(list(dataset_fold_dict.keys()))

    def predict_dataset(self, dataset: str) -> DatasetPredictionsDict:
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
        rename_split = lambda split : 'test' if split == "pred_proba_dict_test" else 'val'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        tasks_to_models = {
            dataset: {
                fold: {
                    rename_split(split): list(models.keys())
                    for split, models in splits.items()
                }
                for fold, splits in folds.items()
            }
            for dataset, folds in pred_dict.items()
        }
        print(f"saving .pkl files in folder {output_dir}")
        for dataset in tqdm(datasets):
            filename = str(output_dir / f'{dataset}.pkl')
            save_pkl(filename, pred_dict[dataset])
        cls._save_metadata(output_dir=output_dir, tasks_to_models=tasks_to_models)
        return cls(tasks_to_models=tasks_to_models, output_dir=output_dir)

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
        for dataset, folds in self.tasks_to_models.items():
            for fold, splits in folds.items():
                for split, models in splits.items():
                    self.tasks_to_models[dataset][fold][split] = list(models_to_keep.intersection(self.tasks_to_models[dataset][fold][split]))

        # remove collections that may now be empty
        self.tasks_to_models = filter_empty(self.tasks_to_models)

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

    def models_available_for_dataset_fold_split(self, dataset, fold, split) -> List[str]:
        split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
        try:
            return self.pred_dict[dataset][fold][split_key].model_index.keys()
        except KeyError:
            return []

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