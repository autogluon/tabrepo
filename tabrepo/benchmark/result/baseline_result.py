
import copy

import pandas as pd

from tabrepo.benchmark.result.abstract_result import AbstractResult


class BaselineResult(AbstractResult):
    def __init__(self, result: dict, convert_format: bool = True, inplace: bool = False):
        super().__init__(result=result, inplace=inplace)
        if convert_format:
            if inplace:
                self.result = copy.deepcopy(self.result)
            self.result = self._align_result_input_format()

        required_keys = [
            "framework",
            "dataset",
            "fold",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        for key in required_keys:
            assert key in self.result, f"Missing {key} in result dict!"

    @property
    def framework(self):
        return self.result["framework"]

    def _align_result_input_format(self) -> dict:
        """
        Converts results in old format to new format
        Keeps results in new format as-is.

        This enables the use of results in the old format alongside results in the new format.

        Returns
        -------

        """
        if "metric_error_val" in self.result:
            self.result["metric_error_val"] = float(self.result["metric_error_val"])
        if "df_results" in self.result:
            self.result.pop("df_results")
        return self.result

    def compute_df_result(self) -> pd.DataFrame:
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "time_train_s",
            "time_infer_s",
            "problem_type",
        ]

        optional_columns = [
            "metric_error_val",
            "tid",
        ]

        columns_to_use = copy.deepcopy(required_columns)

        for c in required_columns:
            assert c in self.result
        for c in optional_columns:
            if c in self.result:
                columns_to_use.append(c)

        df_result = pd.DataFrame([{c: self.result[c] for c in columns_to_use}])

        return df_result
