import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from tabrepo.simulation.ensemble_selection_config_scorer import EnsembleScorer
from tqdm.auto import trange

from autogluon.common.loaders import load_pkl
from autogluon.timeseries.models.ensemble import TimeSeriesGreedyEnsemble


def recursive_dd():
    return defaultdict(recursive_dd)


def dd_to_dict(dd):
    dd = dict(dd)
    for k, v in dd.items():
        if isinstance(v, defaultdict):
            dd[k] = dd_to_dict(v)
    return dd


def convert_simulation_artifacts_to_tabular_predictions_dict(
    simulation_artifacts: Dict[str, Dict[int, Dict[str, Any]]]
) -> Tuple[dict, dict]:
    aggregated_pred_proba = recursive_dd()
    aggregated_ground_truth = recursive_dd()
    for task_name in simulation_artifacts.keys():
        for fold in simulation_artifacts[task_name].keys():
            zeroshot_metadata = simulation_artifacts[task_name][fold]
            fold = int(fold)
            if fold not in aggregated_ground_truth[task_name]:
                for k in [
                    "y_val",
                    "y_test",
                    "target",
                    "prediction_length",
                    "eval_metric",
                    "eval_metric_seasonal_period",
                    "quantile_levels",
                ]:
                    aggregated_ground_truth[task_name][fold][k] = zeroshot_metadata[k]
                aggregated_ground_truth[task_name][fold]["task"] = task_name
            for k in ["pred_dict_val", "pred_dict_test"]:
                for m, pred_proba in zeroshot_metadata[k].items():
                    aggregated_pred_proba[task_name][fold][k][m] = pred_proba
    return dd_to_dict(aggregated_pred_proba), dd_to_dict(aggregated_ground_truth)


class TimeSeriesEnsembleScorer(EnsembleScorer):
    def __init__(self, zeroshot_pp: dict, zeroshot_gt: dict, **kwargs):
        self.zeroshot_pp = zeroshot_pp
        self.zeroshot_gt = zeroshot_gt

    def get_task_metadata(self, dataset: str, fold: int):
        return self.zeroshot_gt[dataset][fold]

    def evaluate_task(self, dataset: str, fold: int, models: List[str]) -> Tuple[float, np.array]:
        task_metadata = self.get_task_metadata(dataset, fold)
        pred_dict_val = self.zeroshot_pp[dataset][fold]["pred_dict_val"]
        pred_dict_test = self.zeroshot_pp[dataset][fold]["pred_dict_test"]

        ensemble = TimeSeriesGreedyEnsemble(
            name=None,
            target=task_metadata["target"],
            eval_metric=task_metadata["eval_metric"],
            prediction_length=task_metadata["prediction_length"],
            eval_metric_seasonal_period=task_metadata["eval_metric_seasonal_period"],
        )
        ensemble.fit_ensemble(
            predictions_per_window={m: pred_dict_val[m] for m in models},
            data_per_window=task_metadata["y_val"],
        )
        ensemble_weights = np.array([ensemble.model_to_weight.get(m, 0.0) for m in models])
        preds = ensemble.predict({m: pred_dict_test[m] for m in ensemble.model_names})
        err = -1.0 * ensemble._score_with_predictions(task_metadata["y_test"], preds)
        return err, ensemble_weights


def greedy_portfolio_selection(
    portfolio_size: int, models: list, scorer: EnsembleScorer, dataset: str = dataset, fold: int = 0
) -> list:
    """
    Toy greedy portfolio selection algorithm when only considering a single dataset and fold.
    """
    print(
        f"Performing greedy portfolio selection... (portfolio_size={portfolio_size}, dataset={dataset}, fold={fold})"
    )
    portfolio = []
    cur_models = models
    for p in trange(portfolio_size):
        if len(cur_models) == 0:
            break
        model_errs = dict()
        for model in cur_models:
            candidate_portfolio = portfolio + [model]
            err, ensemble_weights = scorer.evaluate_task(dataset=dataset, fold=fold, models=candidate_portfolio)
            model_errs[model] = err
        best_model = min(model_errs, key=model_errs.get)
        portfolio = portfolio + [best_model]
        cur_models = [m for m in cur_models if m != best_model]
        print(f"\t{model_errs[best_model]}\t{best_model}")
    return portfolio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="m4_hourly")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("-p", "--path", type=str, default="simulation_artifact.pkl")
    parser.add_argument("-s", "--size", type=int, default=3)
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold

    simulation_artifact = load_pkl.load(args.path)
    simulation_artifacts = {dataset: {fold: simulation_artifact}}

    zeroshot_pp, zeroshot_gt = convert_simulation_artifacts_to_tabular_predictions_dict(
        simulation_artifacts=simulation_artifacts
    )
    scorer = TimeSeriesEnsembleScorer(zeroshot_pp=zeroshot_pp, zeroshot_gt=zeroshot_gt)

    greedy_portfolio_selection(
        portfolio_size=args.size,
        models=list(zeroshot_pp[dataset][fold]["pred_dict_val"].keys()),
        scorer=scorer,
        dataset=dataset,
        fold=fold,
    )
