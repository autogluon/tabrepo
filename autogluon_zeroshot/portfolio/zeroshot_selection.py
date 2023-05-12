from typing import List
import pandas as pd
import numpy as np

def zeroshot_configs(val_scores: np.array, output_size: int) -> List[int]:
    """
    :param val_scores: a tensor with shape (n_task, n_configs) that contains evaluations to consider
    :param output_size: number of configurations to return, in some case where no configuration helps anymore,
    the portfolio can have smaller length.
    :return: a list of index configuration for the portfolio where all indices are in [0, `n_configs` - 1]
    """
    assert val_scores.ndim == 2

    df_val_scores = pd.DataFrame(val_scores)
    ranks = pd.DataFrame(df_val_scores).rank(axis=1)
    res = []
    for _ in range(output_size):
        # Select greedy-best configuration considering all others
        best_idx = ranks.mean(axis=0).idxmin()

        # Update ranks for choosing each configuration considering the previously chosen ones
        ranks.clip(upper=ranks[best_idx], axis=0, inplace=True)
        if best_idx not in df_val_scores:
            break
        # Drop the chosen configuration as a future candidate
        df_val_scores.drop(columns=best_idx, inplace=True)
        res.append(best_idx)
    return res
