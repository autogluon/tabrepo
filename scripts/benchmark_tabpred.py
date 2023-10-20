from pathlib import Path

from tabrepo.simulation.convert_memmap import convert_memmap_pred_from_pickle
from tabrepo.predictions import TabularPredictionsMemmap, TabularPredictionsInMemory, TabularPredictionsInMemoryOpt
from tabrepo.utils import catchtime

filepath = Path(__file__)


def compute_all_with_load(data_dir, predictions_class, name: str):
    with catchtime(f"Load time with   {name}"):
        preds = predictions_class.from_data_dir(data_dir=data_dir)
    with catchtime(f"Compute sum with {name}"):
        res = compute_all(preds=preds)
    print(f"Sum obtained with {name}: {res}")
    return res


def compute_all(preds):
    datasets = preds.datasets
    res = 0
    for dataset in datasets:
        for fold in preds.folds:
            pred_val = preds.predict_val(dataset, fold, models)
            pred_test = preds.predict_test(dataset, fold, models)
            res += pred_val.mean() + pred_test.mean()

    return res


if __name__ == '__main__':

    """
    start: Compute sum with memmap
    Sum obtained with memmap: 1176878.8741704822
    Time for Compute sum with memmap: 0.0547 secs
        
    start: Compute sum with pickle per task
    Sum obtained with pickle per task: 1176878.874170535
    Time for Compute sum with pickle per task: 7.5385 secs

    """
    models = [f"CatBoost_r{i}_BAG_L1" for i in range(1, 10)]
    repeats = 1

    # Download predictions locally
    # load_context("BAG_D244_F3_C1416_micro", ignore_cache=False)

    pickle_path = Path(filepath.parent.parent / "data/results/2023_08_21_micro/zeroshot_metadata/")
    memmap_dir = Path(filepath.parent.parent / "data/results/2023_08_21_micro/model_predictions/")
    if not memmap_dir.exists():
        print("converting to memmap")
        convert_memmap_pred_from_pickle(pickle_path, memmap_dir)

    class_method_tuples = [
        (TabularPredictionsInMemory, "mem"),
        (TabularPredictionsInMemoryOpt, "memopt"),
        (TabularPredictionsMemmap, "memmap"),
    ]

    for (predictions_class, name) in class_method_tuples:
        compute_all_with_load(data_dir=memmap_dir, predictions_class=predictions_class, name=name)
