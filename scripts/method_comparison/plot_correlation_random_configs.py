"""
Plot correlation between train/test error, requires to run evaluate_random_configs.py before to generate evaluations.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

from autogluon_zeroshot.loaders import Paths

expname = "0lG55"
csv_filename = Paths.results_root / f"random-evals-{expname}.csv"

df = pd.read_csv(csv_filename)
# print(df.to_string())
all_n_splits = df.n_splits.unique()
# all_num_folds_fit = df.num_folds_fit.unique()

all_num_folds_fit = [10]
for n_splits in all_n_splits:
    for num_folds_fit in all_num_folds_fit:
        corrs = []
        for fold in range(n_splits):
            fold_errors = df.loc[
                (df.n_splits == n_splits) & (df.num_folds_fit == num_folds_fit) & (df.fold == fold),
                ['train-score', 'test-score']
            ]
            corrs.append(fold_errors.corr().min().min())
            print(f"Pearson correlation between train and test error on fold {fold} with {n_splits} splits and {num_folds_fit} folds used to fit: {corrs[-1]}")

        errors = df.loc[
            (df.n_splits == n_splits) & (df.num_folds_fit == num_folds_fit),
            ['train-score', 'test-score']
        ]
        fig = seaborn.regplot(errors, x='train-score', y='test-score').figure
        # fig.suptitle(f"Correlation between train and test error ({n_splits} splits and {num_folds_fit} folds used to fit)")
        fig.suptitle(f"Correlation between train and test error with {n_splits} splits")

        plt.tight_layout()
        plt.savefig(f"correlation-split-folds-{n_splits}-{num_folds_fit}.png")
        plt.show()

        avg_corr = np.mean(corrs)
        print(f"AVG Pearson correlation between train and test error on all folds with {n_splits} splits and {num_folds_fit} "
              f"folds used to fit: {avg_corr}")
