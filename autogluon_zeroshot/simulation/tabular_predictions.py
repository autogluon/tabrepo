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

    @property
    def folds(self) -> List[int]:
        # todo we could assert that the same number of folds exists in all cases
        first = next(iter(self.pred_dict.values()))
        return list(first.keys())


class TabularPicklePerTaskPredictions(TabularModelPredictions):
    def __init__(self, dataset_to_models: Dict[str, List[str]], output_dir: str):
        self.dataset_to_models = dataset_to_models
        self.output_dir = Path(output_dir)
        assert self.output_dir.is_dir()

    def _score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
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
        return self.dataset_to_models[dataset]

    @property
    def datasets(self):
        return list(self.dataset_to_models.keys())

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


class TabularNpyPredictions(TabularModelPredictions):
    def __init__(self, dataset_to_models: Dict[str, List[str]], output_dir: str):
        self.dataset_to_models = dataset_to_models
        self.output_dir = Path(output_dir)
        assert self.output_dir.is_dir()

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str):
        def _stack_and_slice(arrays):
            num_points_splits = set([arr.shape[1] for arr in arrays])
            if len(num_points_splits) > 1:
                # some splits may have different number of points slice to uniform
                min_num_points = min(num_points_splits)
                print(
                    f"Folds have different number of points ({num_points_splits}), keeping {min_num_points} in each fold.")
                arrays = np.array([arr[:, :min_num_points, ...] for arr in arrays])
                return np.stack(arrays)
            else:
                return np.stack(arrays)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba = TabularPicklePredictions.from_dict(pred_dict=pred_dict)
        datasets = pred_proba.datasets
        dataset_to_models = {
            dataset: pred_proba.models_available_per_dataset(dataset)
            for dataset in datasets
        }
        models = pred_proba.models
        print(f"saving .npy files in folder {output_dir}")
        for dataset in datasets:
            filename = output_dir / f'{dataset}.npy'
            print(filename)
            folds = pred_proba.folds
            with open(filename, 'wb') as f:
                # two tensor with shape (num_models, output_dim) or (num_models,) if
                # the problem is unidimensional
                val_array = _stack_and_slice([
                    pred_proba.score(dataset=dataset, fold=fold, splits=['val'], models=models)[0]
                    for fold in folds
                ])
                test_array = _stack_and_slice([
                    pred_proba.score(dataset=dataset, fold=fold, splits=['test'], models=models)[0]
                    for fold in folds
                ])
                for arr in val_array, test_array:
                    assert arr.shape[0] == len(folds)
                    assert arr.shape[1] == len(models)
                np.save(f, val_array.astype("float16"))
                np.save(f, test_array.astype("float16"))
        cls._save_metadata(output_dir=output_dir, dataset_to_models=dataset_to_models)
        return cls(dataset_to_models=dataset_to_models, output_dir=output_dir)

    def _load_dataset(self, dataset: str) -> Tuple[np.array, np.array]:
        filename = self.output_dir / f'{dataset}.npy'
        with open(filename, 'rb') as f:
            val_array = np.load(f)
            test_array = np.load(f)
        return val_array, test_array

    def _score(self, dataset: str, fold: int, splits: List[str] = None, models: List[str] = None) -> List[np.array]:
        # two tensor with shape (num_folds, num_models, output_dim) or (num_folds, num_models,)
        val_array, test_array = self._load_dataset(dataset)
        split_dict = {
            "val": val_array,
            "test": test_array,
        }
        # tensors with shape (num_models, num_points, num_classes) for each split
        return [split_dict[split][fold] for split in splits]

    def save(self, output_dir: str):
        print(f"saving into {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata(output_dir, self.dataset_to_models)
        print(f"copy .npy files from {self.output_dir} to {output_dir}")
        for file in self.output_dir.glob("*.npy"):
            shutil.copyfile(file, output_dir / file.name)

    @classmethod
    def load(cls, filename: str):
        # 1) load dataset to models in json
        # 2) load .npy
        filename = Path(filename)

        metadata = cls._load_metadata(filename)
        dataset_to_models = metadata["dataset_to_models"]

        return cls(
            dataset_to_models=dataset_to_models,
            output_dir=filename,
        )

    def models_available_per_dataset(self, dataset: str, fold: int) -> List[str]:
        return self.dataset_to_models[dataset]

    @property
    def datasets(self):
        return list(self.dataset_to_models.keys())

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
