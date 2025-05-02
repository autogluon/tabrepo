import pytest
import openml
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.features import AutoMLPipelineFeatureGenerator
import numpy as np
from autogluon.core.models import BaggedEnsembleModel
from tabrepo.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
from autogluon.tabular.testing import FitHelper
from sklearn.metrics import accuracy_score
import shutil
import torch
import time
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

def test_tabicl():
    model_hyperparameters = {"n_estimators": 1}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabrepo.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
        model_cls = TabICLModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )

def run_bagging(task_id, fold, bagging=True):

    print('Task id', task_id, 'Fold', fold)

    bagged_custom_model = BaggedEnsembleModel(TabICLModel(path="", hyperparameters={"n_estimators": 1}))
    custom_model = TabICLModel(hyperparameters={"n_estimators": 1})
    bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local' 

    task = openml.tasks.get_task(task_id, download_splits=False)
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(fold=fold)
    x_train, x_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    n_class = len(np.unique(y_train.values))
    problem_type = 'multiclass' if n_class > 2 else 'binary'

    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    feature_generator = AutoMLPipelineFeatureGenerator()
    x_train = feature_generator.fit_transform(X=x_train, y=y_train)
    y_train = label_cleaner.transform(y_train)
    x_test = feature_generator.transform(X=x_test)
    y_test = label_cleaner.transform(y_test).values

    time1 = time.time()

    if bagging:
        bagged_custom_model.fit(X=x_train, y=y_train, k_fold=8) # Perform 8-fold bagging
    else:
        custom_model.fit(X=x_train, y=y_train)
    
    time2 = time.time()

    if bagging:
        out = bagged_custom_model.predict_proba(x_test)
    else:
        out = custom_model.predict_proba(x_test)
    
    if n_class == 2 and (out.ndim == 1 or out.shape[1] == 1):
        out = np.vstack([1 - out[:, 0], out[:, 0]]).T if out.ndim > 1 else np.vstack([1 - out, out]).T

    time3 = time.time()

    train_time = time2 - time1
    infer_time = time3 - time2

    accuracy = accuracy_score(y_test, out[:,:n_class].argmax(axis=-1))
    ce = log_loss(y_test, out[:,:n_class], labels=list(range(n_class)))
    if n_class == 2:
        roc = roc_auc_score(y_test, out[:,:2][:, 1])
    else:
        roc = roc_auc_score(y_test, out[:,:n_class], multi_class='ovo', labels=list(range(n_class)))

    print(f"accuracy: {accuracy}, ce: {ce}, roc: {roc}")

    if bagging:
        file_path = '/fsx/results/tabrepo10fold/tabicl_bagging_ft.csv'
    else:
        file_path = '/fsx/results/tabrepo10fold/tabicl_no_bagging.csv'

    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame({
        "roc": roc,
        "ce": ce,
        "accuracy": accuracy,
        "time_train_s": train_time,
        "time_infer_s": infer_time,
    }, index=[f'tabrepo_{task_id}' + f'_fold_{fold}'])
    df.to_csv(file_path, mode='a', index=True, header=not file_exists, float_format='%.4f')

if __name__ == "__main__":

    df = pd.read_csv("/home/ubuntu/tabular/mitra/tabpfnmix/evaluate/task_metadata_244.csv")
    df = df[df['ttid']=='TaskType.SUPERVISED_CLASSIFICATION']
    dids = df['tid'].values.tolist()
    names_to_filter = df['name'].values.tolist()
    filtered_data = list(zip(dids, names_to_filter))
    print(len(dids))

    # filter datasets with over 10 cls, over 100 features, over 3000 rows
    remove = [6, 26, 28, 30, 32, 41, 43, 45, 58, 219, 223, 2074, 2076, 2076, 3011, 3481, 3481, 3481, 3483, 3573, 3573, 3591, 3593, 3600, 3601, 3618, 3627, 3668, 3672, 3681, 3684, 3688, 3698, 3712, 3735, 3764, 3786, 3844, 3892, 3892, 3897, 3899, 3904, 3907, 3919, 3945, 3945, 3954, 3962, 3964, 3968, 3976, 3980, 3995, 4000, 7307, 7593, 9920, 9923, 9927, 9928, 9931, 9932, 9943, 9945, 9945, 9954, 9959, 9960, 9964, 9966, 9972, 9972, 9974, 9974, 9976, 9983, 14963, 14970, 14970, 125922, 125922, 125968, 125968, 146802, 146820, 146823, 146823, 167120, 167121, 167121, 167121, 167124, 167124, 168300, 168300, 168350, 168868, 168868, 168909, 168909, 168910, 168910, 168911, 189354, 189355, 189355, 189356, 189773, 189922, 189922, 190392, 190392, 190410, 190410, 190411, 190412, 211978, 211978, 211978, 211979, 211980, 211980, 211980, 211986, 359953, 359953, 359957, 359961, 359964, 359964, 359966, 359966, 359967, 359967, 359968, 359969, 359970, 359971, 359972, 359973, 359973, 359974, 359975, 359976, 359976, 359977, 359979, 359980, 359980, 359981, 359982, 359983, 359984, 359984, 359985, 359985, 359986, 359986, 359987, 359988, 359988, 359989, 359989, 359990, 359991, 359992, 359993, 359994, 360112, 360112, 360113, 360114, 360859, 360859, 360975, 360975, 361324, 361324, 361324, 361325, 361325, 361325, 361326, 361326, 361326, 361327, 361327, 361330, 361331, 361342, 361343, 361344]
    remove += [361332, 361334]

    filtered_data = [data for data in filtered_data if data[0] not in remove]
    dids, names_to_filter = zip(*filtered_data)
    dids = list(dids)
    names_to_filter = list(names_to_filter)
    print(len(dids))

    for did in dids:

        for fold in range(10):

            run_bagging(task_id=did, fold=fold, bagging=True)  

            shutil.rmtree("/home/ubuntu/tabular/tabrepo/AutogluonModels")