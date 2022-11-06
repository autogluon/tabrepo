import numpy as np
import pandas as pd

from autogluon_zeroshot.simulation.config_generator import ZeroshotConfigGeneratorCV
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.contexts.context_2022_10_13 import load_context_2022_10_13


def get_zeroshot_config_simulation(zeroshot_configs, config_scorer, df_raw, zeroshot_sim_name):
    df_pivot_time_train_s = config_scorer.df_results_by_dataset_with_score_val.pivot_table(index='framework', columns='dataset', values=['time_train_s'])
    zeroshot_raw_df = config_scorer.get_configs_df(zeroshot_configs)
    zeroshot_raw_df['tid_new'] = zeroshot_raw_df['dataset']
    zeroshot_raw_df['model'] = zeroshot_raw_df['framework']
    df_raw_zeroshot = df_raw.merge(zeroshot_raw_df[['tid_new', 'model']], on=['tid_new', 'model'])
    df_raw_zeroshot['model'] = zeroshot_sim_name
    df_raw_zeroshot['framework'] = zeroshot_sim_name
    df_raw_zeroshot['framework_parent'] = ''

    df_zs_train_time_s_per_dataset_ = df_pivot_time_train_s.loc[zeroshot_configs]['time_train_s'].sum()
    df_zs_train_time_s_per_dataset_.name = 'time_train_s'
    df_zs_train_time_s_per_dataset_ = df_zs_train_time_s_per_dataset_.rename_axis('tid_new')
    df_zs_train_time_s_per_dataset_ = df_zs_train_time_s_per_dataset_.to_frame().reset_index(drop=False)

    # TODO: Still not technically correct train_time_s, can optimize order of zero-shot training and report the earliest at which the score is generated
    df_raw_zeroshot = df_raw_zeroshot.drop(columns=['time_train_s']).merge(df_zs_train_time_s_per_dataset_, on=['tid_new'])  # Get the correct train_time_s
    return df_raw_zeroshot


# FIXME: Ideally this should be easier, but not really possible with current logic.
def metric_error_to_score(metric_error: float, metric: str):
    if metric.startswith('neg_'):
        metric_score = -metric_error
    elif metric == 'auc':
        metric_score = 1 - metric_error
    else:
        raise AssertionError(f'Unknown metric: {metric}')
    return metric_score


if __name__ == '__main__':
    out_name = 'EnsembleTrueCV'

    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_10_13(load_zeroshot_pred_proba=True)
    zsc.print_info()

    from autogluon.common.loaders import load_pkl
    results = load_pkl.load(path='ensemble_result.pkl')

    num_folds = len(results)

    for i in range(num_folds):
        print(f'Fold {results[i]["fold"]} Selected Configs: {results[i]["selected_configs"]}')


    for i in range(num_folds):
        print(f'Fold {results[i]["fold"]} Test Score: {results[i]["score"]}')

    score = np.mean([r['score'] for r in results])
    print(f'Final Test Score: {score}')
    # zeroshot_configs_dict = {config: configs_full[config] for config in zeroshot_final}

    df_results = zsc.df_results_by_dataset_vs_automl
    df_raw = zsc.df_raw

    df_raw_all = []
    # FIXME: can speed up via ray
    for f in range(num_folds):
        print(f)
        results_1 = results[f]
        zeroshot_final = results_1['selected_configs']
        zeroshot_configs_dict = {config: configs_full[config] for config in zeroshot_final}
        datasets_test = results_1['X_test_fold']

        config_scorer_test = EnsembleSelectionConfigScorer.from_zsc(
            datasets=datasets_test,
            zeroshot_simulator_context=zsc,
            zeroshot_gt=zeroshot_gt,
            zeroshot_pred_proba=zeroshot_pred_proba,
            ensemble_size=100,
        )

        # config_scorer_test = SingleBestConfigScorer.from_zsc(
        #     zeroshot_simulator_context=zsc,
        #     datasets=datasets_test,
        # )

        df_raw_subset = df_raw[df_raw['tid_new'].isin(datasets_test)]
        df_raw_subset = df_raw_subset.drop_duplicates(subset=['tid_new'])

        # score_per_dataset = config_scorer_test.score_per_dataset(zeroshot_final, score_col='metric_error')
        score_per_dataset = config_scorer_test.compute_errors(zeroshot_final)

        df_raw_subset['metric_error'] = [score_per_dataset[row[0]] for row in zip(df_raw_subset['tid_new'])]
        df_raw_subset['metric_score'] = [metric_error_to_score(row[0], row[1]) for row in zip(df_raw_subset['metric_error'], df_raw_subset['metric'])]
        df_raw_subset['framework'] = out_name
        df_raw_subset['framework_parent'] = ''
        df_raw_all.append(df_raw_subset)
        # FIXME: accurate fit time
        # FIXME: accurate inference time
        # FIXME: accurate score val

    df_raw_all = pd.concat(df_raw_all)

    from autogluon.common.savers import save_pd
    # Use this file to compare theoretical performance to AutoGluon in separate analysis repo
    save_pd.save(path=f'zeroshot_results/zeroshot_{out_name}.csv', df=df_raw_all)
    s3_prefix = 's3://autogluon-zeroshot/config_results'
    save_pd.save(path=f'{s3_prefix}/zeroshot_{out_name}.csv', df=df_raw_all)

    print('yo')

