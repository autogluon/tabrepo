"""
Writes latex file that can be copied to the latex directory into automl_conf/tables/ag_and_portfolio.tex

Requires to have run run_eval_tabrepo_v1.py before which will generate the csv file.
* Repo: https://github.com/Innixma/autogluon-benchmark
* Install via README.md: https://github.com/Innixma/autogluon-benchmark?tab=readme-ov-file#development-installation
* Script to run: https://github.com/Innixma/autogluon-benchmark/blob/master/scripts/run_eval/tabrepo/run_eval_tabrepo_v1.py
"""
from pathlib import Path

from autogluon.common.loaders.load_pd import load


if __name__ == "__main__":
    path = Path(__file__).parent.parent.parent / 'autogluon-benchmark/scripts/run_eval/tabrepo/data/results/output/openml' \
                                          '/autogluon_v1/4h8c_fillna/all/pairwise/AutoGluon with Portfolio (Best, 4h8c).csv'
    df = load(str(path))
    cols = ['framework', 'Winrate', '% Loss Reduction']
    df_sub = df.loc[:, cols]
    df_sub.framework = df_sub.framework.apply(lambda s: s.split(" (")[0])
    df_sub.framework = df_sub.framework.apply(lambda s: s.replace("AutoGluon", "AG"))
    df_sub.framework = df_sub.framework.apply(lambda s: s.replace("with Portfolio", "+ portfolio"))

    df_sub = df_sub.sort_values(by=["Winrate", "% Loss Reduction"])
    df_sub["Winrate"] = df_sub["Winrate"] * 100
    df_sub["Winrate"] = df_sub["Winrate"].apply(lambda x: f"{int(round(x, 0))}\%")
    df_sub["% Loss Reduction"] = df_sub["% Loss Reduction"].apply(lambda x: f"{round(x, 1)}\%")
    df_sub.rename({
        "framework": "method",
        "Winrate": "win-rate",
        "% Loss Reduction": "loss reduc."
    }, axis=1, inplace=True)

    s = df_sub.to_latex(index=False, column_format="lrr")

    print(s)
    save_path = Path(__file__).parent / "tables/ag_and_portfolio.tex"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(s)
