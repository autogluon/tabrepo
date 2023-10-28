from pathlib import Path

from tabrepo.simulation.convert_memmap import convert_memmap_pred_from_pickle
from tabrepo.predictions import TabularModelPredictions, TabularPredictionsMemmap, TabularPredictionsInMemory, TabularPredictionsInMemoryOpt
from tabrepo.utils import catchtime

filepath = Path(__file__)


def compute_all_with_load(data_dir, predictions_class, name: str, models: list, repeats: int = 1):
    with catchtime(f"Load time with   {name}"):
        preds: TabularModelPredictions = predictions_class.from_data_dir(data_dir=data_dir)
    with catchtime(f"Compute sum with repeats={repeats} : {name}"):
        for _ in range(repeats):
            res = compute_all(preds=preds, models=models)
    print(f"Sum obtained with {name}: {res}\n")
    return res


def compute_all(preds: TabularModelPredictions, models):
    datasets = preds.datasets
    res = 0
    for dataset in datasets:
        for fold in preds.folds:
            pred_val = preds.predict_val(dataset=dataset, fold=fold, models=models)
            pred_test = preds.predict_test(dataset=dataset, fold=fold, models=models)
            res += pred_val.mean() + pred_test.mean()

    return res


if __name__ == '__main__':

    """
    start: Load time with   mem
    Time for Load time with   mem: 31.4967 secs
    start: Compute sum with repeats=100 : mem
    Time for Compute sum with repeats=100 : mem: 3.5681 secs
    Sum obtained with mem: 1177631.0633783638
    
    start: Load time with   memopt
    Time for Load time with   memopt: 31.6524 secs
    start: Compute sum with repeats=100 : memopt
    Time for Compute sum with repeats=100 : memopt: 4.7095 secs
    Sum obtained with memopt: 1177631.0633783638
    
    start: Load time with   memmap
    Time for Load time with   memmap: 0.0245 secs
    start: Compute sum with repeats=100 : memmap
    Time for Compute sum with repeats=100 : memmap: 3.1743 secs
    Sum obtained with memmap: 1177631.0633783638
    """
    models = [f"CatBoost_r{i}_BAG_L1" for i in range(1, 50)]
    repeats = 100

    # Download predictions locally
    # load_context("D244_F3_C1416_30", ignore_cache=False)

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
        compute_all_with_load(data_dir=memmap_dir, predictions_class=predictions_class, name=name, repeats=repeats, models=models)
