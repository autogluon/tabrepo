import pandas as pd
from scripts.baseline_comparison.evaluate_baselines import list_artificial_experiments
from scripts.baseline_comparison.plot_utils import show_latex_table, show_cdf


def test_evaluate_baselines():
    experiments = list_artificial_experiments()
    df = pd.concat([
        experiment.data(ignore_cache=True) for experiment in experiments
    ])
    show_latex_table(df)
    show_cdf(df)