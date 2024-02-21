import os
import math
import seaborn as sns
import pandas as pd
import matplotlib
from pathlib import Path

from autogluon.common.savers import save_pd

from tabrepo import EvaluationRepository
from tabrepo.loaders import Paths
from tabrepo.utils import catchtime

from scripts import load_context, show_repository_stats
from scripts.baseline_comparison.baselines import (
    automl_results,
    framework_default_results,
    framework_best_results,
    zeroshot_results,
    zeroshot_name,
    framework_name,
    time_suffix,
    default_ensemble_size,
    n_portfolios_default,
)
from scripts.baseline_comparison.compare_results import winrate_comparison
from scripts.baseline_comparison.evaluate_utils import (
    Experiment,
    rename_dataframe,
    generate_sensitivity_plots,
    plot_figure,
    save_total_runtime_to_file,
    make_scorers,
)
from scripts.baseline_comparison.plot_utils import (
    MethodStyle,
    show_latex_table,
    show_scatter_performance_vs_time, show_scatter_performance_vs_time_lower_budgets,
    figure_path,
    plot_critical_diagrams,
)


def run_evalute_baselines(
    repo,
    ignore_cache: bool = False,
    ray_process_ratio: float = None,
    engine: str = "ray",
    expname: str = None,
    all_configs: bool = False,
    n_datasets: int = None,
    n_eval_folds: int = -1,
):
    repo_version = repo
    expname = repo_version if expname is None else expname
    as_paper = not all_configs

    if n_datasets:
        expname += f"-{n_datasets}"

    if engine == "ray" and ray_process_ratio is not None:
        assert (ray_process_ratio <= 1) and (ray_process_ratio > 0)
        num_cpus = os.cpu_count()
        num_ray_processes = math.ceil(num_cpus*ray_process_ratio)

        print(f'NOTE: To avoid OOM, we are limiting ray processes to {num_ray_processes} (Total Logical Cores: {num_cpus})\n'
              f'\tThis is based on ray_process_ratio={ray_process_ratio}')

        # FIXME: The large-scale 3-fold 244-dataset 1416-config runs OOM on m6i.32x without this limit
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=num_ray_processes)

    n_portfolios = [5, 10, 25, 50, 100, n_portfolios_default]
    max_runtimes = [300, 600, 1800, 3600, 3600 * 4, 24 * 3600]
    # n_training_datasets = list(range(10, 210, 10))
    # n_training_configs = list(range(10, 210, 10))
    n_training_datasets = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 199]
    n_training_configs = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
    n_seeds = 20
    n_training_folds = [1, 2, 5, 10]
    n_ensembles = [10, 20, 40, 80]
    linestyle_ensemble = "--"
    linestyle_default = "-"
    linestyle_tune = "dotted"

    # Number of digits to show in table
    n_digits = {
        "normalized-error": 3,
        "rank": 1,
        "time fit (s)": 1,
        "time infer (s)": 3,
    }

    if not as_paper:
        expname += "_ALL"

    repo: EvaluationRepository = load_context(version=repo_version, ignore_cache=ignore_cache, as_paper=as_paper)
    # use predictions from this method in case of model failures
    repo.print_info()
    repo.set_config_fallback("ExtraTrees_c1_BAG_L1")

    if n_eval_folds == -1:
        n_eval_folds = repo.n_folds()

    problem_types = None
    if problem_types is not None:
        expname += f"-{'_'.join(sorted(problem_types))}"
        repo = repo.subset(problem_types=problem_types)

    rank_scorer, normalized_scorer = make_scorers(repo)
    dataset_names = repo.datasets()
    if n_datasets:
        dataset_names = dataset_names[:n_datasets]

    # TODO: This is a hack, in future repo should know the framework_types via the configs.json input
    configs_default = [c for c in repo.configs() if "_c1_" in c]
    framework_types_with_gpu = [c.rsplit('_c1_', 1)[0] for c in configs_default]
    framework_types = [f for f in framework_types_with_gpu if f not in ["FTTransformer"]]
    configs_total = repo.configs()
    framework_counts = [c.split('_', 1)[0] for c in configs_total]
    framework_counts_unique = set(framework_counts)
    framework_counts_dict = {
        f: len([f2 for f2 in framework_counts if f2 == f]) for f in framework_counts_unique
    }
    print(framework_counts_dict)

    expname_outdir = Path("output") / expname

    experiment_common_kwargs = dict(
        repo=repo,
        dataset_names=dataset_names,
        framework_types=framework_types,
        rank_scorer=rank_scorer,
        normalized_scorer=normalized_scorer,
        n_eval_folds=n_eval_folds,
        engine=engine,
    )

    experiment_gpu = experiment_common_kwargs.copy()
    experiment_gpu["framework_types"] = framework_types_with_gpu
    experiment_gpu["method_prefix"] = "-gpu"

    experiments = [
        Experiment(
            expname=expname, name=f"framework-default-{expname}",
            run_fun=lambda: framework_default_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"framework-best-{expname}",
            run_fun=lambda: framework_best_results(max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        ),
        # Experiment(
        #     expname=expname, name=f"framework-all-best-{expname}",
        #     run_fun=lambda: framework_best_results(framework_types=[None], max_runtimes=[3600, 3600 * 4, 3600 * 24], **experiment_common_kwargs),
        # ),
        # Automl baselines such as Autogluon best, high, medium quality
        Experiment(
            expname=expname, name=f"automl-baselines-{expname}",
            run_fun=lambda: automl_results(**experiment_common_kwargs),
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-maxruntimes",
            run_fun=lambda: zeroshot_results(max_runtimes=max_runtimes, **experiment_common_kwargs)
        ),
        Experiment(
            expname=expname, name=f"zeroshot-gpu-{expname}",
            run_fun=lambda: zeroshot_results(**experiment_gpu)
        ),
        # Experiment(
        #     expname=expname, name=f"zeroshot-{expname}-num-folds",
        #     run_fun=lambda: zeroshot_results(n_training_folds=n_training_folds, ** experiment_common_kwargs)
        # ),
        Experiment(
            expname=expname, name=f"zeroshot-{expname}-num-portfolios",
            run_fun=lambda: zeroshot_results(n_portfolios=n_portfolios, n_ensembles=[1, default_ensemble_size], **experiment_common_kwargs)
        ),
    ]

    # Use more seeds
    seeds = [s for s in range(n_seeds)]
    experiments.append(Experiment(
        expname=expname, name=f"zeroshot-{expname}-num-configs-n-seeds-{n_seeds}",
        run_fun=lambda: zeroshot_results(
            n_training_configs=n_training_configs,
            n_ensembles=[1, default_ensemble_size],
            seeds=seeds,
            **experiment_common_kwargs,
        )
    ))

    experiments.append(Experiment(
        expname=expname, name=f"zeroshot-{expname}-num-training-datasets-n-seeds-{n_seeds}",
        run_fun=lambda: zeroshot_results(
            n_training_datasets=n_training_datasets,
            n_ensembles=[1, default_ensemble_size],
            seeds=seeds,
            **experiment_common_kwargs,
        )
    ))

    with catchtime("total time to generate evaluations"):
        df = pd.concat([
            experiment.data(ignore_cache=ignore_cache) for experiment in experiments
        ])
    # De-duplicate in case we ran a config multiple times
    df = rename_dataframe(df)
    framework_types.remove("NeuralNetTorch")
    framework_types.append("MLP")
    df = df.drop_duplicates(subset=["method", "dataset", "fold", "seed"])

    print(f"Obtained {len(df)} evaluations on {len(df.dataset.unique())} datasets for {len(df.method.unique())} methods.")
    print(f"Methods available:" + "\n".join(sorted(df.method.unique())))
    total_time_h = df.loc[:, "time fit (s)"].sum() / 3600
    print(f"Total time of experiments: {total_time_h} hours")
    save_total_runtime_to_file(total_time_h, save_prefix=expname_outdir)

    generate_sensitivity_plots(df, show=True, save_prefix=expname_outdir)

    # Drop multiple seeds after generating sensitivity plots
    df = df.drop_duplicates(subset=["method", "dataset", "fold"])

    # Save results
    save_pd.save(path=str(Paths.data_root / "simulation" / expname / "results.csv"), df=df)

    # df = time_cutoff_baseline(df)

    show_latex_table(df, "all", show_table=True, n_digits=n_digits, save_prefix=expname_outdir)
    ag_styles = [
        # MethodStyle("AutoGluon best (1h)", color="black", linestyle="--", label_str="AG best (1h)"),
        MethodStyle("AutoGluon best (4h)", color="black", linestyle="-.", label_str="AG best (4h)", linewidth=2.5),
        # MethodStyle("AutoGluon high quality (ensemble)", color="black", linestyle=":", label_str="AG-high"),
        # MethodStyle("localsearch (ensemble) (ST)", color="red", linestyle="-")
    ]

    method_styles = ag_styles.copy()
    show_latex_table(df[df.method.isin([m.name for m in method_styles])], "frameworks", n_digits=n_digits, save_prefix=expname_outdir)

    for i, framework_type in enumerate(framework_types):
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, tuned=False),
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_default,
                label=True,
                label_str=framework_type,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, ensemble_size=1, tuned=True),
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_tune,
                label=False,
            )
        )
        method_styles.append(
            MethodStyle(
                framework_name(framework_type, max_runtime=4 * 3600, tuned=True),
                color=sns.color_palette('bright', n_colors=20)[i],
                linestyle=linestyle_ensemble,
                label=False,
                label_str=framework_type
            )
        )

    plot_figure(df, method_styles, figname="cdf-frameworks", save_prefix=expname_outdir)

    plot_figure(
        df, [x for x in method_styles if "ensemble" not in x.name], figname="cdf-frameworks-tuned",
        title="Effect of tuning configurations",
        save_prefix=expname_outdir,
    )

    plot_figure(
        df,
        [x for x in method_styles if any(pattern in x.name for pattern in ["tuned", "AutoGluon"])],
        figname="cdf-frameworks-ensemble",
        title="Effect of tuning & ensembling",
        # title="Comparison of frameworks",
        save_prefix=expname_outdir,
    )

    cmap = matplotlib.colormaps["viridis"]
    # Plot effect number of training datasets
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_dataset=size),
            color=cmap(i / (len(n_training_datasets) - 1)), linestyle="-", label_str=r"$\mathcal{D}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_datasets)
    ]
    plot_figure(df, method_styles, title="Effect of number of training tasks", figname="cdf-n-training-datasets", save_prefix=expname_outdir)

    # # Plot effect number of training fold
    # method_styles = ag_styles + [
    #     MethodStyle(
    #         zeroshot_name(n_training_fold=size),
    #         color=cmap(i / (len(n_training_folds) - 1)), linestyle="-", label_str=f"S{size}",
    #     )
    #     for i, size in enumerate(n_training_folds)
    # ]
    # plot_figure(df, method_styles, title="Effect of number of training folds", figname="cdf-n-training-folds")

    # Plot effect number of portfolio size
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_portfolio=size),
            color=cmap(i / (len(n_portfolios) - 1)), linestyle="-", label_str=r"$\mathcal{N}~=~" + f"{size}$",
        )
        for i, size in enumerate(n_portfolios)
    ]
    plot_figure(df, method_styles, title="Effect of number of portfolio configurations", figname="cdf-n-configs", save_prefix=expname_outdir)

    # Plot effect of number of training configurations
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(n_training_config=size),
            color=cmap(i / (len(n_training_configs) - 1)), linestyle="-", label_str=r"$\mathcal{M}'~=~" + f"{size}$",
        )
        for i, size in enumerate(n_training_configs)
    ]
    plot_figure(df, method_styles, title="Effect of number of offline configurations", figname="cdf-n-training-configs", save_prefix=expname_outdir)

    # Plot effect of training time limit
    method_styles = ag_styles + [
        MethodStyle(
            zeroshot_name(max_runtime=size),
            color=cmap(i / (len(max_runtimes) - 1)), linestyle="-",
            label_str=time_suffix(size).replace("(", "").replace(")", "").strip(),
        )
        for i, size in enumerate(max_runtimes)
    ]
    plot_figure(df, method_styles, title="Effect of training time limit", figname="cdf-max-runtime", save_prefix=expname_outdir)

    automl_frameworks = ["Autosklearn2", "Flaml", "Lightautoml", "H2oautoml"]
    for budget in ["1h", "4h"]:
        budget_suffix = f"\({budget}\)"
        # df = df[~df.method.str.contains("All")]
        df_selected = df[
            (df.method.str.contains(f"AutoGluon .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(automl_frameworks) + f").*{budget_suffix}")) |
            (df.method.str.contains(f"Portfolio-N{n_portfolios_default} .*{budget_suffix}")) |
            (df.method.str.contains(".*(" + "|".join(framework_types) + ")" + f".*{budget_suffix}")) |
            (df.method.str.contains(".*default.*"))
        ].copy()
        df_selected.method = df_selected.method.str.replace(f" {budget_suffix}", "").str.replace(f"\-N{n_portfolios_default}", "")
        show_latex_table(
            df_selected,
            title=f"selected-methods-{budget}",
            show_table=True,
            n_digits=n_digits,
            save_prefix=expname_outdir,
        )

    show_latex_table(
        df[(df.method.str.contains("Portfolio") | (df.method.str.contains("AutoGluon ")))],
        title="zeroshot",
        n_digits=n_digits,
        save_prefix=expname_outdir,
    )

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path(prefix=expname_outdir) / f"scatter-perf-vs-time.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    fig, _, bbox_extra_artists = show_scatter_performance_vs_time_lower_budgets(df, metric_cols=["normalized-error", "rank"])
    fig_save_path = figure_path(prefix=expname_outdir) / f"scatter-perf-vs-time-lower-budget.pdf"
    fig.savefig(fig_save_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.show()

    plot_critical_diagrams(df, save_prefix=expname_outdir)

    winrate_comparison(df=df, repo=repo, save_prefix=expname_outdir)

    show_repository_stats.get_stats(expname_outdir=expname_outdir, repo=repo)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--repo", type=str, help="Name of the repo to load", default="D244_F3_C1416_200")
    parser.add_argument("--n_folds", type=int, default=-1, required=False,
                        help="Number of folds to consider when evaluating all baselines. Uses all if set to -1.")
    parser.add_argument("--n_datasets", type=int, required=False, help="Number of datasets to consider when evaluating all baselines.")
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore previously generated results and recompute them from scratch.")
    parser.add_argument("--all_configs", action="store_true", help="If True, will use all configs rather than filtering out NeuralNetFastAI. If True, results will differ from the paper.")
    parser.add_argument("--expname", type=str, help="Name of the experiment. If None, defaults to the value specified in `repo`.", required=False, default=None)
    parser.add_argument("--engine", type=str, required=False, default="ray", choices=["sequential", "ray", "joblib"],
                        help="Engine used for embarrassingly parallel loop.")
    parser.add_argument("--ray_process_ratio", type=float,
                        help="The ratio of ray processes to logical cpu cores. Use lower values to reduce memory usage. Only used if engine == 'ray'",)
    args = parser.parse_args()

    print(args.__dict__)
    run_evalute_baselines(
        repo=args.repo,
        n_eval_folds=args.n_folds,
        n_datasets=args.n_datasets,
        ignore_cache=args.ignore_cache,
        all_configs=args.all_configs,
        expname=args.expname,
        engine=args.engine,
        ray_process_ratio=args.ray_process_ratio,
    )
