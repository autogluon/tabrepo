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
from pathlib import Path

from scripts import load_context
from scripts.baseline_comparison.plot_utils import table_path


def get_stats(expname_outdir: str, context: str = None, repo=None):
    if repo is None:
        assert context is not None, f"Either repo or context must be specified."
        repo = load_context(context)

    automl_frameworks = repo._zeroshot_context.df_baselines.framework.unique().tolist()

    automl_frameworks = set(x.split("_")[0] for x in automl_frameworks)

    n_datasets = repo.n_datasets()
    n_bagged = 8
    n_hps = repo.n_configs()
    n_seeds = repo.n_folds()
    n_automl = len(automl_frameworks)

    config_names = repo.configs()
    config_names_defaults = [c for c in config_names if "_c1_" in c]
    realnumframeworks = len(config_names_defaults)

    n_cpus = 8

    realnumhourstabrepo = int(repo._zeroshot_context.df_configs['time_train_s'].sum() / 3600)
    realnumcpuhourstabrepo = realnumhourstabrepo * n_cpus

    save_path_tables = table_path(prefix=expname_outdir)
    runtime_file = save_path_tables / "runtime.txt"
    assert runtime_file.exists(), "You should run `evaluate_baselines.py` first to compute total time of experiments"

    print(f"read runtime from {runtime_file}")
    with open(runtime_file, "r") as f:
        # % shown with "Total time of experiments:" when running evaluations
        numhoursnosimu = int(float(f.readline().strip()))

    stats = {
        "realnumdatasets": n_datasets,
        "realnumtasks": n_datasets * n_seeds,
        "realnumbagged": n_bagged,
        "realnumhps": n_hps,
        "realnumseeds": n_seeds,
        "realnumautomlbaseline": n_automl,
        "realnumframeworks": realnumframeworks,
        # "realnumevaluations": n_datasets * (n_hps + n_automl) * n_seeds,
        "realnumevaluations": n_datasets * n_hps * n_seeds,  # drop automl systems as we dont store their predictions
        "realnumhourstabrepo": realnumhourstabrepo,
        "realnumcpuhourstabrepo": realnumcpuhourstabrepo,
        "numhoursnosimu": numhoursnosimu,
        "numcpuhoursnosimu": numhoursnosimu * n_cpus,
        "ratiosaving": "{:.1f}".format(round(numhoursnosimu / realnumhourstabrepo, 1)),
    }

    with open(save_path_tables / "tab_repo_constants.tex", "w") as f:
        for name, value in stats.items():
            bracket = lambda s: "{" + str(s) + "}"
            make_latex_command = lambda name, value: "{\\" + str(name) + "}" + "{" + str(value) + "}"
            print(f"\\newcommand{make_latex_command(name, value)}")
            f.write(f"\\newcommand{make_latex_command(name, value)}\n")


if __name__ == "__main__":
    context = "D244_F3_C1530_200"
    expname = "D244_F3_C1530_200"
    expname_outdir = str(Path("output") / expname)
    get_stats(context=context, expname_outdir=expname_outdir)
