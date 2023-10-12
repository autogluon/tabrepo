import argparse
from pathlib import Path

import yaml

from autogluon.common.savers import save_pkl
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def load_ds(name, dataset_dir=Path("/home/shchuro/data/datasets/")):
    data_dir = dataset_dir / name
    with open(data_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    prediction_length = metadata.get("default_horizon", 14)
    data_full = TimeSeriesDataFrame.from_path(str(data_dir / "data.parquet"))
    d_train, d_test = data_full.train_test_split(prediction_length)
    return prediction_length, d_train, d_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="m4_hourly")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("-p", "--path", type=str, default="simulation_artifact.pkl")
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold

    prediction_length, d_train, d_test = load_ds(dataset)
    predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(d_train, num_val_windows=2)
    simulation_artifact = predictor.get_simulation_artifact(d_test)
    save_pkl.save(args.path, simulation_artifact)
