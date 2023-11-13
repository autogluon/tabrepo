import tempfile
from typing import Dict, List

import pandas as pd
from pathlib import Path


class GroundTruth:
    def __init__(self, label_val_dict: Dict[str, Dict[int, pd.Series]], label_test_dict: Dict[str, Dict[int, pd.Series]]):
        """

        :param label_val_dict: dictionary from tid to fold to labels series where the index are openml rows and
        values are the labels
        :param label_test_dict: same as `label_val_dict`
        """
        assert set(label_val_dict.keys()) == set(label_test_dict.keys())
        self._label_val_dict = label_val_dict
        self._label_test_dict = label_test_dict

    @property
    def datasets(self) -> List[str]:
        return sorted(list(self._label_val_dict.keys()))

    # FIXME: Add restrict instead, same as tabular_predictions
    def remove_dataset(self, dataset: str):
        self._label_val_dict.pop(dataset)
        self._label_test_dict.pop(dataset)

    def labels_val(self, dataset: str, fold: int):
        # Note we could also expose the series index (original row of OpenML)
        return self._label_val_dict[dataset][fold].values.flatten()

    def labels_test(self, dataset: str, fold: int):
        return self._label_test_dict[dataset][fold].values.flatten()

    # TODO: Unit test
    @classmethod
    def from_dict(cls, label_dict):
        label_val_dict = dict()
        label_test_dict = dict()
        for dataset in label_dict.keys():
            label_val_dict[dataset] = dict()
            label_test_dict[dataset] = dict()
            for fold in label_dict[dataset].keys():
                labels_val = label_dict[dataset][fold]["y_val"]
                labels_test = label_dict[dataset][fold]["y_test"]
                label_val_dict[dataset][fold] = labels_val
                label_test_dict[dataset][fold] = labels_test
        return cls(label_val_dict=label_val_dict, label_test_dict=label_test_dict)

    # TODO: Unit test
    def to_data_dir(self, data_dir: str):
        if data_dir[:2] == "s3":
            from autogluon.common.utils.s3_utils import is_s3_url, upload_s3_folder, s3_path_to_bucket_prefix
            if is_s3_url(str(data_dir)):
                s3_bucket, s3_prefix = s3_path_to_bucket_prefix(data_dir)
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.to_data_dir(data_dir=temp_dir)
                    upload_s3_folder(bucket=s3_bucket, prefix=s3_prefix, folder_to_upload=temp_dir)
                return

        datasets = self.datasets
        for dataset in datasets:
            for fold in self._label_val_dict[dataset]:
                target_folder = Path(data_dir) / dataset / str(fold)
                target_folder.mkdir(exist_ok=True, parents=True)
                labels_val = self._label_val_dict[dataset][fold]
                labels_test = self._label_test_dict[dataset][fold]
                labels_val.to_csv(target_folder / "label-val.csv.zip", index=True)
                labels_test.to_csv(target_folder / "label-test.csv.zip", index=True)
