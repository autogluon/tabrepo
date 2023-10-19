from pathlib import Path

from scripts import load_context
from tabrepo.simulation.convert_memmap import convert_memmap_pred_from_pickle
from tabrepo.predictions import TabularPredictionsMemmap
from tabrepo.simulation.tabular_predictions_old import TabularPicklePerTaskPredictions
from tabrepo.utils import catchtime

filepath = Path(__file__)

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

    pickle_path = Path(filepath.parent.parent / "data/results/2023_08_21/zeroshot_metadata/")
    memmap_dir = Path(filepath.parent.parent / "data/results/2023_08_21/model_predictions/")
    if not memmap_dir.exists():
        print("converting to memmap")
        convert_memmap_pred_from_pickle(pickle_path, memmap_dir)

    with catchtime("Compute sum with memmap"):
        for _ in range(repeats):
            preds = TabularPredictionsMemmap(data_dir=memmap_dir)
            datasets = preds.datasets
            res = 0
            for dataset in datasets:
                for fold in preds.folds:
                    pred_val = preds.predict_val(dataset, fold, models)
                    pred_test = preds.predict_test(dataset, fold, models)
                    res += pred_val.mean() + pred_test.mean()
            print(f"Sum obtained with memmap: {res}")

    # Load previous format to compare performance
    paths = [x for x in list(pickle_path.rglob("*zeroshot_pred_proba.pkl"))]
    preds = TabularPicklePerTaskPredictions.from_paths(paths, output_dir=pickle_path)

    with catchtime("Compute sum with pickle per task"):
        for _ in range(repeats):
            res = 0
            for dataset in datasets:
                for fold in preds.folds:
                    pred_val, pred_test = preds.predict(dataset=dataset, fold=fold, models=models, splits=["val", "test"])
                    res += pred_val.mean() + pred_test.mean()
            print(f"Sum obtained with pickle per task: {res}")
