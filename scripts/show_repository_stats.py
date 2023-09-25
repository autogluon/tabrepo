"""
\newcommand{\realnumdatasets}{130}
\newcommand{\realnumbagged}{8}
\newcommand{\realnumhps}{608}
\newcommand{\realnumseeds}{10}
\newcommand{\realnumtasks}{1300}
\newcommand{\realnumensemble}{10}
\newcommand{\realnumautomlbaseline}{6}
\newcommand{\realnumframeworks}{5}
\newcommand{\realnumevaluations}{790004} % numhps x numtasks x numbagged

\newcommand{\realnumhourstabrepo}{2854}
\newcommand{\realnumcpuhourstabrepo}{22832} % realnumhourstabrepo x 8, depends on machine type update when 2x large is used
\newcommand{\numhoursnosimu}{73403} % shown with "Total time of experiments:" when running evaluations
\newcommand{\numcpuhoursnosimu}{587224} % numhoursnosimu x 8
"""
from scripts import load_context, output_path
from scripts.baseline_comparison.baselines import framework_types

repo = load_context()

automl_frameworks = repo._zeroshot_context.df_results_by_dataset_automl.framework.unique().tolist()

automl_frameworks = set(x.split("_")[0] for x in automl_frameworks)

n_datasets = repo.n_datasets()
n_bagged = 8
n_hps = repo.n_models()
n_seeds = repo.n_folds()
n_automl = len(automl_frameworks)

n_cpus = 8

realnumhourstabrepo = int(repo._zeroshot_context.df_results_by_dataset_vs_automl['time_train_s'].sum() / 3600)
realnumcpuhourstabrepo = realnumhourstabrepo * n_cpus
numhoursnosimu = 96698  # % shown with "Total time of experiments:" when running evaluations
numcpuhoursnosimu = numhoursnosimu * n_cpus

stats = {
    "realnumdatasets": n_datasets,
    "realnumtasks": n_datasets * n_seeds,
    "realnumbagged": n_bagged,
    "realnumhps": n_hps,
    "realnumseeds": n_seeds,
    "realnumautomlbaseline": n_automl,
    "realnumframeworks": len(framework_types),
    # "realnumevaluations": n_datasets * (n_hps + n_automl) * n_seeds,
    "realnumevaluations": n_datasets * n_hps * n_seeds,  # drop automl systems as we dont store their predictions
    "realnumhourstabrepo": realnumhourstabrepo,
    "realnumcpuhourstabrepo": realnumcpuhourstabrepo,
    "numhoursnosimu": numhoursnosimu,
    "numcpuhoursnosimu": numcpuhoursnosimu * n_cpus,
    "ratiosaving": "{:.1f}".format(round(numhoursnosimu / realnumhourstabrepo, 1)),
}

with open(output_path / "tables" / "tab_repo_constants.tex", "w") as f:

    for name, value in stats.items():
        bracket = lambda s: "{" + str(s) + "}"
        make_latex_command = lambda name, value: "{\\" + str(name) + "}" + "{" + str(value) + "}"
        print(f"\\newcommand{make_latex_command(name, value)}")
        f.write(f"\\newcommand{make_latex_command(name, value)}\n")