import copy
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union

import pickle
from pathlib import Path
import numpy as np
import json


# dictionary mapping the config name to predictions for a given dataset fold split
ConfigPredictionsDict = Dict[str, np.array]

# dictionary mapping a particular fold of a dataset (a task) to split to config name to predictions
TaskPredictionsDict = Dict[str, ConfigPredictionsDict]

# dictionary mapping the folds of a dataset to split to config name to predictions
DatasetPredictionsDict = Dict[int, TaskPredictionsDict]

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = Dict[str, DatasetPredictionsDict]


class TabularModelPredictions:
    def __init__(self, datasets: List[str] = None):
        """
        Contains a collection of model evaluations, can be instantiated by either `TabularPredictionsInMemory`
        which contains all evaluations in memory with a dictionary or `TabularPredictionsMemmap` which stores
        all model evaluations on disk and retrieve them on the fly with memmap.
        :param datasets: if specified, only consider the given list of dataset
        """
        if datasets:
            self.restrict_datasets(datasets)

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None, **kwargs):
        """
        Instantiates from a dictionary of predictions
        :param pred_dict:
        :param output_dir: directory where files are written (used for `TabularPredictionsMemmap`)
        :return:
        """
        raise NotImplementedError()

    def to_dict(self) -> TabularPredictionsDict:
        """
        :return: the whole dictionary of predictions
        """
        raise NotImplementedError()

    def predict_val(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        """
        Obtains validation predictions on a given dataset and fold for a list of models
        :return: predictions with shape (num_models, num_rows, num_classes) for classification and
        (num_models, num_rows, ) for regression
        """
        raise NotImplementedError()

    def predict_test(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        """
        Obtains test predictions on a given dataset and fold for a list of models
        :return: predictions with shape (num_models, num_rows, num_classes) for classification and
        (num_models, num_rows, ) for regression
        """
        raise NotImplementedError()

    @property
    def datasets(self) -> List[str]:
        """
        :return: list of datasets that are present in the collection
        """
        return list(self.model_available_dict().keys())

    def restrict_datasets(self, datasets: List[str]):
        raise NotImplementedError()

    @property
    def folds(self) -> List[int]:
        """
        :return: list of folds that are present in all datasets
        """
        all_folds = []
        for dataset, fold_dict in self.model_available_dict().items():
            all_folds.append(fold_dict.keys())
        return list(set.intersection(*map(set, all_folds))) if all_folds else []

    def restrict_folds(self, folds: List[int]):
        raise NotImplementedError()

    @property
    def models(self) -> List[str]:
        """
        :return: list of models that are present in all datasets and folds
        """
        all_models = []
        for dataset, fold_dict in self.model_available_dict().items():
            for fold, models in fold_dict.items():
                all_models.append(models)
        return list(set.intersection(*map(set, all_models))) if all_models else []

    def restrict_models(self, models: List[str]):
        raise NotImplementedError()

    def model_available_dict(self) -> Dict[str, Dict[int, List[str]]]:
        """
        :return: a dictionary listing all evaluations available mapping dataset to fold to list of models available.
        """
        model_available_dict = self._model_available_dict()
        return self._filter_empty(model_available_dict)

    def _filter_empty(self, model_available_dict):
        # remove all possibly empty collections from the given nested dictionaries
        res = model_available_dict
        for dataset, folds in model_available_dict.items():
            for fold, models in folds.items():
                if models:
                    res[dataset][fold] = models
            res[dataset] = {fold: models for fold, models in folds.items() if models}
        return {dataset: folds for dataset, folds in res.items() if folds}

    def _model_available_dict(self) -> Dict[str, Dict[int, List[str]]]:
        raise NotImplementedError()


class TabularPredictionsInMemory(TabularModelPredictions):
    def __init__(self, pred_dict: TabularPredictionsDict, datasets: Optional[List[str]] = None):
        # TODO assert that models are all both in validation and test set
        self.pred_dict = pred_dict
        super(TabularPredictionsInMemory, self).__init__(datasets=datasets)

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None, datasets: Optional[List[str]] = None):
        # Optional, avoids changing passed object
        pred_dict = copy.deepcopy(pred_dict)
        return cls(pred_dict=pred_dict, datasets=datasets)

    def to_dict(self) -> TabularPredictionsDict:
        return self.pred_dict

    def predict_val(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="val")

    def predict_test(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="test")

    def _load_pred(self, dataset: str, split: str, fold: int, models: List[str] = None):
        if models is None:
            models = self.models

        def get_split(split, models):
            split_key = 'pred_proba_dict_test' if split == "test" else 'pred_proba_dict_val'
            model_results = self.pred_dict[dataset][fold][split_key]
            return np.array([self._get_model_results(model=model, model_pred_probas=model_results) for model in models])

        return get_split(split, models)

    def _get_model_results(self, model: str, model_pred_probas: dict) -> np.array:
        return model_pred_probas[model]

    def restrict_datasets(self, datasets: List[str]):
        self.pred_dict = {
            dataset: fold_dict for dataset, fold_dict in self.pred_dict.items() if dataset in datasets
        }

    def restrict_folds(self, folds: List[int]):
        for dataset, fold_dict in self.pred_dict.items():
            self.pred_dict[dataset] = {
                fold: fold_info for fold, fold_info in fold_dict.items() if fold in folds
            }

    def restrict_models(self, models: List[str]):
        selected_models = set(models)
        for dataset, fold_dict in self.pred_dict.items():
            for fold, fold_info in fold_dict.items():
                for split, model_dict in fold_info.items():
                    self.pred_dict[dataset][fold][split] = {
                        model: v for model, v in model_dict.items() if model in selected_models
                    }

    def _model_available_dict(self) -> Dict[str, Dict[int, List[str]]]:
        return {
            dataset: {
                fold: list(fold_info['pred_proba_dict_val'].keys()) for fold, fold_info in fold_dict.items()
            }
            for dataset, fold_dict in self.pred_dict.items()
        }


def path_memmap(folder_memmap: Path, dataset: str, fold: int):
    return folder_memmap / dataset / str(fold)


class TabularPredictionsMemmap(TabularModelPredictions):
    def __init__(self, data_dir: Union[str, Path], datasets: Optional[List[str]] = None):
        """
        :param data_dir: data where the predictions has been saved
        :param datasets: if specified, the predictions only contains those datasets
        """
        self.data_dir = Path(data_dir)
        self.metadata_dict = self._load_metadatas(data_dir)
        super(TabularPredictionsMemmap, self).__init__(datasets=datasets)

    @classmethod
    def from_dict(cls, pred_dict: TabularPredictionsDict, output_dir: str = None, datasets: Optional[List[str]] = None, dtype: str = "float32"):
        assert dtype in ["float16", "float32"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset, folds_dict in pred_dict.items():
            for fold, folds in folds_dict.items():
                target_folder = path_memmap(output_dir, dataset, fold)
                target_folder.mkdir(exist_ok=True, parents=True)

                # print(f"Converting {dataset} fold {fold} to {target_folder}")

                models = list(folds["pred_proba_dict_val"].keys())
                assert set(list(folds["pred_proba_dict_test"].keys())) == set(models), \
                    "different models available on validation and testing"

                def get_split(split_key, models):
                    model_results = pred_dict[dataset][fold][split_key]
                    return np.array([model_results[model] for model in models])

                pred_val = get_split("pred_proba_dict_val", models)
                pred_test = get_split("pred_proba_dict_test", models)

                # Save metadata that are required to retrieve the predictions, in particular the shape of model
                # predictions and the model list
                with open(target_folder / "metadata.json", "w") as f:
                    metadata_dict = {
                        "models": models,
                        "dataset": dataset,
                        "fold": fold,
                        "pred_val_shape": pred_val.shape,
                        "pred_test_shape": pred_test.shape,
                        "dtype": dtype,
                    }

                    f.write(json.dumps(metadata_dict))

                # Dumps data to memmap tensors, alternatively could use .npy but it would make model loading much
                # slower in cases when only some models are loaded
                fp = np.memmap(str(target_folder / "pred-val.dat"), dtype=dtype, mode='w+', shape=pred_val.shape)
                fp[:] = pred_val[:]
                fp.flush()
                fp = np.memmap(str(target_folder / "pred-test.dat"), dtype=dtype, mode='w+', shape=pred_test.shape)
                fp[:] = pred_test[:]
                fp.flush()

        return cls(data_dir=output_dir, datasets=datasets)

    def to_dict(self) -> TabularPredictionsDict:
        model_available_dict = self.model_available_dict()
        return {
            dataset: {
                fold: {
                    "pred_proba_dict_val": {
                        model: self.predict_val(dataset, fold, [model]).squeeze() for model in models
                    },
                    "pred_proba_dict_test": {
                        model: self.predict_test(dataset, fold, [model]).squeeze() for model in models
                    }
                } for fold, models in fold_dict.items()
            } for dataset, fold_dict in model_available_dict.items()
        }

    @staticmethod
    def _load_metadatas(data_dir):
        res = defaultdict(dict)
        metadata_files = list(Path(data_dir).rglob("*metadata.json"))
        for metadata_file in metadata_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            dataset = metadata.pop("dataset")
            fold = metadata.pop("fold")
            res[dataset][fold] = metadata
        return res

    def predict_val(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        pred = self._load_pred(dataset=dataset, fold=fold, models=models, split="val")
        return pred

    def predict_test(self, dataset: str, fold: int, models: List[str] = None) -> np.array:
        pred = self._load_pred(dataset=dataset, fold=fold, models=models, split="test")
        return pred

    def _load_pred(self, dataset: str, split: str, fold: int, models: List[str] = None):
        assert dataset in self.metadata_dict, f"{dataset} not available."
        assert fold in self.metadata_dict[dataset], f"Fold {fold} of {dataset} not available."

        assert split in ["val", "test"]
        task_folder = path_memmap(self.data_dir, dataset, fold)
        metadata = self.metadata_dict[dataset][fold]
        models_available = {m: i for i, m in enumerate(metadata['models'])}
        models_indices = [models_available[m] for m in models]
        dtype = metadata["dtype"]
        pred = np.memmap(
            str(task_folder / f"pred-{split}.dat"),
            dtype=dtype,
            mode='r',
            shape=tuple(metadata[f"pred_{split}_shape"])
        )
        pred = pred[models_indices]
        return pred

    def restrict_datasets(self, datasets: List[str]):
        datasets = set(datasets)
        self.metadata_dict = {
            dataset: folds
            for dataset, folds in self.metadata_dict.items()
            if dataset in datasets
        }

    def restrict_folds(self, folds: List[int]):
        folds = set(folds)
        self.metadata_dict = {
            dataset: {
                fold: fold_metadata
                for fold, fold_metadata in fold_dict.items() if fold in folds
            }
            for dataset, fold_dict in self.metadata_dict.items()
        }

    def restrict_models(self, models: List[str]):
        selected_models = set(models)
        for dataset, fold_dict in self.metadata_dict.items():
            for fold, fold_metadata in fold_dict.items():
                self.metadata_dict[dataset][fold]["models"] = [
                    m for m in self.metadata_dict[dataset][fold]["models"]
                    if m in selected_models
                ]

    def _model_available_dict(self) -> Dict[str, Dict[int, List[str]]]:
        return {
            dataset: {
                fold: fold_info["models"] for fold, fold_info in fold_dict.items()
            }
            for dataset, fold_dict in self.metadata_dict.items()
        }
