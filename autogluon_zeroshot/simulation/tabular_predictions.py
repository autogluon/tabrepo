import json
import shutil
from typing import List, Dict, Tuple

import numpy as np
from pathlib import Path

from autogluon.common.loaders import load_pkl
from autogluon.common.savers.save_pkl import save

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = Dict[str, Dict[int, Dict[str, Dict[str, np.array]]]]


class TabularModelPredictions:
    """
    Class that allows to query offline predictions.
    """

    def score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        """
        :param dataset:
        :param fold:
        :param splits: split to consider values must be in 'val' or 'test'
        :param models: list of models to be evaluated, by default uses all models available
        :return: for each split, a tensor with shape (num_models, num_points) for regression and
        (num_models, num_points, num_classes) for classification.
        """
        if splits is None:
            splits = ['val', 'test']
        for split in splits:
            assert split in ['val', 'test']
        assert models is None or len(models) > 0
        return self._score(dataset, fold, splits, models)

    def models_available_per_dataset(self, dataset: str, fold: int) -> List[str]:
        """:returns the models available on both validation and test splits"""
        raise NotImplementedError()

    @property
    def models(self) -> List[str]:
        raise NotImplementedError()

    @property
    def datasets(self) -> List[str]:
        raise NotImplementedError()

    @property
    def folds(self) -> List[int]:
        raise NotImplementedError()

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

    def _score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
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
        save(filename, self.pred_dict)

    def _score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = self.pred_dict[dataset][fold][split_key]
            if models is None:
                models = model_results.keys()
            return np.array([model_results[model] for model in models])

        return [get_split(split, models) for split in splits]

    def models_available_per_dataset(self, dataset: str, fold: int = 0) -> List[str]:
        models = []
        for fold in self.folds:
            for split in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                models.append(set(self.pred_dict[dataset][fold][split].keys()))
        # returns models that appears in all lists, eg that are available for all folds and splits
        return list(set.intersection(*map(set, models)))

    @property
    def models(self) -> List[str]:
        models = []
        for dataset in self.datasets:
            for fold in self.folds:
                for split in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                    models.append(set(self.pred_dict[dataset][fold][split].keys()))
        # returns models that appears in all lists, eg that are available for all datasets, folds and splits
        return list(set.intersection(*map(set, models)))

    @property
    def datasets(self) -> List[str]:
        return list(self.pred_dict.keys())

    def remove_dataset(self, dataset: str):
        self.pred_dict.pop(dataset)

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.pred_dict = {rename_dict.get(k, k): v for k, v in self.pred_dict.items()}

    @property
    def folds(self) -> List[int]:
        # todo we could assert that the same number of folds exists in all cases
        first = next(iter(self.pred_dict.values()))
        return list(first.keys())


class TabularPicklePerTaskPredictions(TabularModelPredictions):
    def __init__(self, dataset_to_models: Dict[str, List[str]], output_dir: str):
        """
        Stores on pickle per task and load data in a lazy fashion which allows to reduce significantly the memory
        footprint.
        :param dataset_to_models:
        :param output_dir:
        """
        self.dataset_to_models = dataset_to_models
        self.output_dir = Path(output_dir)
        self.rename_dict_inv = {}
        assert self.output_dir.is_dir()

    def _score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        dataset = self.rename_dict_inv.get(dataset, dataset)
        pred_dict = self._load_dataset(dataset)
        if models is None:
            models = self.models_available_per_dataset(dataset, fold)

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = pred_dict[fold][split_key]
            return np.array([model_results[model] for model in models])
        return [get_split(split, models) for split in splits]

    def _load_dataset(self, dataset: str) -> Dict:
        filename = str(self.output_dir / f'{dataset}.pkl')
        return load_pkl.load(filename)

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        dataset_to_models = {
            dataset: pred_proba.models_available_per_dataset(dataset)
            for dataset in datasets
        }
        print(f"saving .pkl files in folder {output_dir}")
        for dataset in datasets:
            filename = str(output_dir / f'{dataset}.pkl')
            print(filename)
            save(filename, pred_dict[dataset])
        cls._save_metadata(output_dir=output_dir, dataset_to_models=dataset_to_models)
        return cls(dataset_to_models=dataset_to_models, output_dir=output_dir)

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, self.dataset_to_models)
        print(f"copy .pkl files from {self.output_dir} to {output_dir}")
        for file in self.output_dir.glob("*.pkl"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        filename = Path(filename)
        metadata = cls._load_metadata(filename)
        dataset_to_models = metadata["dataset_to_models"]

        return cls(
            dataset_to_models=dataset_to_models,
            output_dir=filename,
        )

    def models_available_per_dataset(self, dataset: str, fold: int) -> List[str]:
        return self.dataset_to_models[self.rename_dict_inv.get(dataset, dataset)]

    @property
    def folds(self) -> List[int]:
        # TODO
        return list(range(10))

    @property
    def datasets(self):
        rename_dict_inv = {v: k for k, v in self.rename_dict_inv.items()}
        return [rename_dict_inv.get(d, d) for d in self.dataset_to_models.keys()]

    def remove_dataset(self, dataset: str):
        if dataset in self.datasets:
            self.dataset_to_models.pop(self.rename_dict_inv.get(dataset, dataset))

    def rename_datasets(self, rename_dict: dict):
        for key in rename_dict:
            assert key in self.datasets
        self.rename_dict_inv = {v: k for k, v in rename_dict.items()}

    @staticmethod
    def _save_metadata(output_dir, dataset_to_models):
        with open(output_dir / "metadata.json", "w") as f:
            metadata = {
                "dataset_to_models": dataset_to_models,
            }
            json.dump(metadata, f)

    @staticmethod
    def _load_metadata(output_dir):
        with open(output_dir / "metadata.json", "r") as f:
            return json.load(f)
