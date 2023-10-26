from typing import Dict, List

import pandas as pd


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
