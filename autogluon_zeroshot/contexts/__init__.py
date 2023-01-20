def intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt):
    dataset_names = zeroshot_pred_proba.datasets
    dataset_names_set = set(dataset_names)
    # for d in zsc.unique_datasets:
    #     if d not in dataset_names_set:
    #         raise AssertionError(f'Missing expected dataset {d} in zeroshot_pred_proba!')
    #     folds_in_zs = list(zeroshot_pred_proba[d].keys())
    #     for f in zsc.folds:
    #         if f not in folds_in_zs:
    #             raise AssertionError(f'Missing expected fold {f} in dataset {d} in zeroshot_pred_proba! '
    #                                  f'Expected: {zsc.folds}, Actual: {folds_in_zs}')

    for d in dataset_names:
        if d not in zsc.unique_datasets:
            zeroshot_pred_proba.remove_dataset(d)
            if d in zeroshot_gt:
                zeroshot_gt.pop(d)
        else:
            # folds_in_zs = list(zeroshot_pred_proba[d].keys())
            for f in zeroshot_pred_proba.folds:
                if f not in zsc.folds:
                    zeroshot_pred_proba[d].pop(f)
                    zeroshot_gt[d].pop(f)
    datasets_in_zs = list(zeroshot_pred_proba.datasets)
    zsc.subset_datasets(datasets_in_zs)

