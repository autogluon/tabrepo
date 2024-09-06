from __future__ import annotations

from collections import defaultdict

import autogluon.common.loaders.load_json
from tabrepo import EvaluationRepository
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from autogluon.features import AutoMLPipelineFeatureGenerator, AsTypeFeatureGenerator

import pandas as pd


def get_feature_info(repo: EvaluationRepository) -> pd.DataFrame:
    datasets = repo.datasets()

    dataset_infos = {}

    from_csv = False
    for dataset in datasets:
        print(dataset)
        task = repo.get_openml_task(dataset=dataset)

        X, y, X_test, y_test = task.get_train_test_split(fold=0)
        if from_csv:
            save_pd.save(path="tmp.csv", df=X)
            X = load_pd.load(path="tmp.csv")
            X.index = y.index

        # feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator_2 = AsTypeFeatureGenerator()

        # X_transform = feature_generator.fit_transform(X=X, y=y)
        X_transform_2 = feature_generator_2.fit_transform(X=X, y=y)

        # feature_metadata = feature_generator.feature_metadata
        feature_metadata_2 = feature_generator_2.feature_metadata
        feature_metadata = feature_metadata_2

        num_each_type_raw = defaultdict(int)
        num_each_type_special = defaultdict(int)

        type_map_raw = feature_metadata.type_map_raw
        for k, v in type_map_raw.items():
            num_each_type_raw[v] += 1
            type_special = feature_metadata.get_feature_types_special(feature=k)
            num_each_type_special[(v, tuple(sorted(type_special)))] += 1

        dataset_infos[dataset] = {
            "num_each_type_raw": num_each_type_raw,
            "num_each_type_special": num_each_type_special,
        }

    series_lst = []

    for dataset in dataset_infos:
        cur_info = dataset_infos[dataset]
        num_each_type_raw = cur_info["num_each_type_raw"]
        num_each_type_special = cur_info["num_each_type_special"]

        b = pd.Series(data=num_each_type_special, name=dataset)
        series_lst.append(b)

    df_out = pd.concat(series_lst, axis=1).fillna(0).astype(int).T

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(df_out)

    return df_out


if __name__ == '__main__':
    context_name = "D244_F3_C1530_10"
    repo: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)

    df_out = get_feature_info(repo)

    a = df_out[("int", ("bool",))]
    print(a)

    b = a[a > 0]
    print(b)

    # datasets = repo.datasets()
    #
    # dataset_infos = {}
    #
    # for dataset in datasets:
    #     print(dataset)
    #     task = repo.get_openml_task(dataset=dataset)
    #
    #     X, y, X_test, y_test = task.get_train_test_split(fold=0)
    #
    #     # feature_generator = AutoMLPipelineFeatureGenerator()
    #     feature_generator_2 = AsTypeFeatureGenerator()
    #
    #
    #     # X_transform = feature_generator.fit_transform(X=X, y=y)
    #     X_transform_2 = feature_generator_2.fit_transform(X=X, y=y)
    #
    #     # feature_metadata = feature_generator.feature_metadata
    #     feature_metadata_2 = feature_generator_2.feature_metadata
    #     feature_metadata = feature_metadata_2
    #
    #     num_each_type_raw = defaultdict(int)
    #     num_each_type_special = defaultdict(int)
    #
    #     type_map_raw = feature_metadata.type_map_raw
    #     for k, v in type_map_raw.items():
    #         num_each_type_raw[v] += 1
    #         type_special = feature_metadata.get_feature_types_special(feature=k)
    #         num_each_type_special[(v, tuple(sorted(type_special)))] += 1
    #
    #     dataset_infos[dataset] = {
    #         "num_each_type_raw": num_each_type_raw,
    #         "num_each_type_special": num_each_type_special,
    #     }
    #
    # series_lst = []
    #
    # for dataset in dataset_infos:
    #     cur_info = dataset_infos[dataset]
    #     num_each_type_raw = cur_info["num_each_type_raw"]
    #     num_each_type_special = cur_info["num_each_type_special"]
    #
    #     import pandas as pd
    #     b = pd.Series(data=num_each_type_special, name=dataset)
    #     series_lst.append(b)
    #
    # df_out = pd.concat(series_lst, axis=1).fillna(0).astype(int).T
    #
    # with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
    #     print(df_out)
    #


"""
                      int float     int object
                       ()    () (bool,)     ()
MIP-2016-regression     4   131       8      1
Moneyball               4     7       2      1
arcene               5969  3929     102      0
boston                  1    11       1      0
dresses-sales           0     1       0     11
fri_c3_500_50           0    50       0      0
pm10                    1     6       0      0
sensory                 7     0       4      0
socmob                  0     1       2      2
tecator                 0   124       0      0

                      int float     int category
                       ()    () (bool,)       ()
MIP-2016-regression     4   131       8        1
Moneyball               3     5       2        4
arcene               5969  3929     102        0
boston                  0    11       1        1
dresses-sales           0     1       0       11
fri_c3_500_50           0    50       0        0
pm10                    1     6       0        0
sensory                 0     0       4        7
socmob                  0     1       2        2
tecator                 0   124       0        0

"""