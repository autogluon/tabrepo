from __future__ import annotations

import copy

import numpy as np
from typing import Any
from typing_extensions import Self

from tabrepo.benchmark.result.baseline_result import BaselineResult


class ConfigResult(BaselineResult):
    def __init__(self, result: dict, convert_format: bool = True, inplace: bool = False):
        super().__init__(result=result, convert_format=convert_format, inplace=inplace)

        required_keys = [
            "simulation_artifacts",
            "metric_error_val",
        ]
        for key in required_keys:
            assert key in self.result, f"Missing {key} in result dict!"

    @property
    def simulation_artifacts(self) -> dict:
        return self.result["simulation_artifacts"]

    def _align_result_input_format(self) -> dict:
        self.result = super()._align_result_input_format()
        dataset = self.result["dataset"]
        fold = self.result["fold"]
        framework = self.result["framework"]

        if list(self.result["simulation_artifacts"].keys()) == [self.result["dataset"]]:
            # if old format
            new_sim_artifacts = self.result["simulation_artifacts"][dataset][fold]
            self.result["simulation_artifacts"] = new_sim_artifacts
        if "pred_proba_dict_test" in self.result["simulation_artifacts"]:
            pred_proba_dict_test = self.result["simulation_artifacts"].pop("pred_proba_dict_test")
            pred_proba_test = pred_proba_dict_test[framework]
            self.result["simulation_artifacts"]["pred_test"] = pred_proba_test
        if "pred_proba_dict_val" in self.result["simulation_artifacts"]:
            pred_proba_dict_val = self.result["simulation_artifacts"].pop("pred_proba_dict_val")
            pred_proba_val = pred_proba_dict_val[framework]
            self.result["simulation_artifacts"]["pred_val"] = pred_proba_val
        if "eval_metric" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["eval_metric"] == self.result["metric"]
            self.result["simulation_artifacts"].pop("eval_metric")
        if "metric" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["metric"] == self.result["metric"]
            self.result["simulation_artifacts"].pop("metric")
        if "problem_type_transform" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["problem_type_transform"] == self.result["problem_type"]
            self.result["simulation_artifacts"].pop("problem_type_transform")
        if "problem_type" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["problem_type"] == self.result["problem_type"]
            self.result["simulation_artifacts"].pop("problem_type")

        return self.result

    def _pred_val_from_children(self) -> np.ndarray:
        num_samples_val = len(self.simulation_artifacts["y_val_idx"])
        if len(self.bag_info["pred_val_per_child"][0].shape) == 1:
            pred_val = np.zeros(dtype=np.float64, shape=num_samples_val)
        else:
            pred_val = np.zeros(dtype=np.float64, shape=(num_samples_val, self.bag_info["pred_val_per_child"][0].shape[1]))
        val_child_count = np.zeros(dtype=int, shape=num_samples_val)
        for val_idx_child, pred_val_child in zip(self.bag_info["val_idx_per_child"], self.bag_info["pred_val_per_child"]):
            val_child_count[val_idx_child] += 1
            pred_val[val_idx_child] += pred_val_child
            pass
        pred_val = pred_val / val_child_count[:, None]
        pred_val = pred_val.astype(np.float32)
        return pred_val

    def _pred_test_from_children(self) -> np.ndarray:
        num_samples_test = len(self.simulation_artifacts["y_test_idx"])
        if len(self.bag_info["pred_val_per_child"][0].shape) == 1:
            pred_test = np.zeros(dtype=np.float64, shape=num_samples_test)
        else:
            pred_test = np.zeros(dtype=np.float64, shape=(num_samples_test, self.bag_info["pred_test_per_child"][0].shape[1]))
        num_children = len(self.bag_info["pred_test_per_child"])
        for pred_test_child in self.bag_info["pred_test_per_child"]:
            pred_test += pred_test_child
        pred_test = pred_test / num_children
        pred_test = pred_test.astype(np.float32)
        return pred_test

    # TODO: Maybe calibrating model binary pred proba will improve ensemble roc_auc?
    def temp_scale(self, y_val, y_pred_proba_val, method: str = "v2"):
        init_val = 1.0
        max_iter = 200
        lr = 0.1
        from tabrepo.utils.temp_scaling.calibrators import (
            AutoGluonTemperatureScalingCalibrator,
            TemperatureScalingCalibrator,
            AutoGluonTemperatureScalingCalibratorFixed,
            TemperatureScalingCalibratorFixed,
        )
        if method == "v1":
            calibrator = AutoGluonTemperatureScalingCalibrator(init_val=init_val, max_iter=max_iter, lr=lr)
        elif method == "v2":
            calibrator = TemperatureScalingCalibrator(max_iter=max_iter, lr=lr)
        elif method == "v1_fix":
            calibrator = AutoGluonTemperatureScalingCalibratorFixed(init_val=init_val, max_iter=max_iter, lr=lr)
        elif method == "v2_fix":
            calibrator = TemperatureScalingCalibratorFixed(max_iter=max_iter, lr=lr)
        else:
            raise ValueError(f"Unknown temp_scale method: {method}")
        calibrator.fit(X=y_pred_proba_val, y=y_val)
        return calibrator

    def generate_calibrated(self, method: str = "v2", name_suffix: str = "_CAL") -> Self:
        result = self.result
        sim_artifact = result["simulation_artifacts"]
        metric = result["metric"]
        from autogluon.core.metrics import get_metric
        problem_type = result["problem_type"]
        ag_metric = get_metric(metric=metric, problem_type=problem_type)
        y_test = sim_artifact["y_test"]

        y_val = sim_artifact["y_val"]
        y_pred_proba_val = sim_artifact["pred_val"]
        calibrator = self.temp_scale(y_val=y_val, y_pred_proba_val=y_pred_proba_val, method=method)
        y_pred_proba_test = sim_artifact["pred_test"]
        y_pred_proba_test_scaled = calibrator.predict_proba(y_pred_proba_test)
        y_pred_proba_val_scaled = calibrator.predict_proba(y_pred_proba_val)

        # metric_error_og = ag_metric.error(y_test, y_pred_proba_test)
        metric_error_cal = ag_metric.error(y_test, y_pred_proba_test_scaled)
        metric_error_val_og = ag_metric.error(y_val, y_pred_proba_val)
        metric_error_val_cal = ag_metric.error(y_val, y_pred_proba_val_scaled)

        if metric_error_val_cal > metric_error_val_og:
            print(f"WARNING:")
            print(metric_error_val_cal, metric_error_val_og)
            print(result["framework"], result["dataset"], result["fold"])

        result_calibrated = copy.deepcopy(self)
        result_calibrated.result["metric_error"] = metric_error_cal
        result_calibrated.result["metric_error_val"] = metric_error_val_cal
        result_calibrated.result["simulation_artifacts"]["pred_test"] = y_pred_proba_test_scaled
        result_calibrated.result["simulation_artifacts"]["pred_val"] = y_pred_proba_val_scaled
        result_calibrated.result["framework"] = result_calibrated.result["framework"] + name_suffix
        # FIXME: Fix bag children? Should they be calibrated?

        return result_calibrated

    def generate_old_sim_artifact(self) -> dict[str, dict[int, dict[str, Any]]]:
        sim_artifacts = copy.deepcopy(self.simulation_artifacts)
        framework = self.result["framework"]
        dataset = self.result["dataset"]
        fold = self.result["fold"]

        sim_artifacts["pred_proba_dict_test"] = {framework: sim_artifacts.pop("pred_test")}
        sim_artifacts["pred_proba_dict_val"] = {framework: sim_artifacts.pop("pred_val")}
        sim_artifacts["eval_metric"] = self.result["metric"]
        sim_artifacts["problem_type"] = self.result["problem_type"]
        sim_artifacts["problem_type_transform"] = self.result["problem_type"]
        sim_artifacts = {dataset: {fold: sim_artifacts}}
        return sim_artifacts

    @property
    def hyperparameters(self) -> dict:
        config_hyperparameters = None
        if "method_metadata" in self.result and "model_hyperparameters" in self.result["method_metadata"]:
            method_metadata = self.result["method_metadata"]
            model_hyperparameters = method_metadata["model_hyperparameters"]
            model_cls = method_metadata.get("model_cls", None)
            model_type = method_metadata.get("model_type", None)
            name_prefix = method_metadata.get("name_prefix", None)

            config_hyperparameters = dict(
                model_cls=model_cls,
                model_type=model_type,
                name_prefix=name_prefix,
                hyperparameters=model_hyperparameters,
            )
        return config_hyperparameters
