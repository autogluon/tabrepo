"""
Writes latex file that can be copied to the latex directory into automl_conf/tables/ag_and_portfolio.tex

Requires to have run run_eval_tabrepo_v1.py before which will generate the csv file.
* Repo: https://github.com/Innixma/autogluon-benchmark
* Install via README.md: https://github.com/Innixma/autogluon-benchmark?tab=readme-ov-file#development-installation
* Script to run: https://github.com/Innixma/autogluon-benchmark/blob/master/scripts/run_eval/tabrepo/run_eval_tabrepo_v1.py
"""
from pathlib import Path

from autogluon.common.loaders.load_pd import load
path = Path(__file__).parent.parent / 'autogluon-benchmark/scripts/run_eval/tabrepo/data/results/output/openml' \
                                      '/autogluon_v1/4h8c/all/pairwise/AutoGluon with Portfolio (Best, 4h8c).csv '
df = load(str(path))
cols = ['framework', 'Winrate', '% Loss Reduction']
df_sub = df.loc[:, cols]
df_sub.framework = df_sub.framework.apply(lambda s: s.split(" (")[0])
df_sub.framework = df_sub.framework.apply(lambda s: s.replace("AutoGluon", "AG"))
df_sub.framework = df_sub.framework.apply(lambda s: s.replace("with Portfolio", "+ portfolio"))

df_sub.rename({
    "framework": "framework",
    "Winrate": "win-rate (\%)",
    "% Loss Reduction": "loss reduc. (\%)"
}, axis=1, inplace=True)
s = df_sub.to_latex(float_format="%.2f", index=False)
print(s)
with open(Path(__file__).parent / "tables/ag_and_portfolio.tex", "w") as f:
    f.write(s)
