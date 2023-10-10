import os
from argparse import ArgumentParser

import pandas as pd

from tabrepo.loaders import Paths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--expname", type=str,
        required=True,
        help="The name that you set or that was generated when you evaluated \"run_method_comparison.py\""
    )
    input_args, _ = parser.parse_known_args()

    expname = input_args.expname

    csv_filename = Paths.results_root / f"{expname}.csv"

    df_results = pd.read_csv(csv_filename).drop("selected_configs", axis=1)

    print(df_results.pivot_table(values=["train-score", "test-score"], columns="fold", index='searcher').to_string())

    df_results.groupby("fold").mean()

    print(df_results.groupby("searcher").agg(['mean', 'std'])[['train-score', 'test-score']].to_string(float_format="%.2f"))
    
    import seaborn
    import matplotlib.pyplot as plt
    errors = df_results.loc[:, ['train-score', 'test-score']]
    fig = seaborn.regplot(errors, x='train-score', y='test-score').figure
    # fig.suptitle(f"Correlation between train and test error ({n_splits} splits and {num_folds_fit} folds used to fit)")
    fig.suptitle(f"Correlation between train and test error")
    plt.tight_layout()
    plt.show()