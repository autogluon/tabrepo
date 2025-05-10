
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
            "task_metadata",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        for key in required_keys:
            assert key in self.result, f"Missing {key} in result dict!"

    @property
    def framework(self) -> str:
        return self.result["framework"]

    @property
    def dataset(self) -> str:
        return self.result["task_metadata"]["name"]

    @property
    def problem_type(self) -> str:
        return self.result["problem_type"]

    @property
    def split_idx(self) -> int:
        return self.result["task_metadata"]["split_idx"]

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
        if "task_metadata" not in self.result:
            self.result["task_metadata"] = dict(
                fold=self.result["fold"],
                repeat=0,
                sample=0,
                split_idx=self.result["fold"],
                tid=self.result["tid"],
                name=self.result["dataset"],
            )
            self.result.pop("fold")
            self.result.pop("tid")
            self.result.pop("dataset")
        return self.result

    def compute_df_result(self) -> pd.DataFrame:
        required_columns = [
            "framework",
            "metric_error",
            "metric",
            "time_train_s",
            "time_infer_s",
            "problem_type",
        ]

        data = {
            "dataset": self.dataset,
            "fold": self.split_idx,
        }

        optional_columns = [
            "metric_error_val",
        ]

        columns_to_use = copy.deepcopy(required_columns)

        for c in required_columns:
            assert c in self.result
        for c in optional_columns:
            if c in self.result:
                columns_to_use.append(c)

        data.update({c: self.result[c] for c in columns_to_use})

        if "tid" in self.result["task_metadata"]:
            data.update({"tid": self.result["task_metadata"]["tid"]})

        df_result = pd.DataFrame([data])

        return df_result
