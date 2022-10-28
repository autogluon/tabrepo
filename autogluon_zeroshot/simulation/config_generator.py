import copy

import numpy as np
from sklearn.model_selection import KFold


class ZeroshotConfigGenerator:
    def __init__(self, config_scorer, configs: list):
        self.config_scorer = config_scorer
        # self.all_configs = list(self.config_scorer.df_pivot_val.index)
        self.all_configs = configs

    def select_zeroshot_configs(self,
                                num_zeroshot: int,
                                zeroshot_configs: list = None,
                                removal_stage=True,
                                removal_threshold=0,
                                config_scorer_test=None,
                                ) -> list:
        if zeroshot_configs is None:
            zeroshot_configs = []
        else:
            zeroshot_configs = copy.deepcopy(zeroshot_configs)

        iteration = 0
        while len(zeroshot_configs) < num_zeroshot:
            iteration += 1
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.
            best_next_config = None
            best_score = 999999999
            for config in self.all_configs:
                # print(config)
                if config in zeroshot_configs:
                    continue
                else:
                    config_selected = zeroshot_configs + [config]
                    config_score = self.config_scorer.score(config_selected)
                    if config_score < best_score:
                        best_score = config_score
                        best_next_config = config
            if best_next_config is None:
                break

            zeroshot_configs.append(best_next_config)
            msg = f'{iteration}\t: {round(best_score, 2)}'
            if config_scorer_test:
                score_test = config_scorer_test.score(zeroshot_configs)
                msg += f'\tTest: {round(score_test, 2)}'
            msg += f'\t{best_next_config}'
            print(msg)

        if removal_stage:
            zeroshot_configs = self.prune_zeroshot_configs(zeroshot_configs, removal_threshold=removal_threshold)
        print(f"selected {zeroshot_configs}")
        return zeroshot_configs

    def prune_zeroshot_configs(self, zeroshot_configs: list, removal_threshold=0) -> list:
        zeroshot_configs = copy.deepcopy(zeroshot_configs)
        best_score = self.config_scorer.score(zeroshot_configs)
        finished_removal = False
        while not finished_removal:
            best_remove_config = None
            for config in zeroshot_configs:
                config_selected = [c for c in zeroshot_configs if c != config]
                config_score = self.config_scorer.score(config_selected)

                if best_remove_config is None:
                    if config_score <= (best_score + removal_threshold):
                        best_score = config_score
                        best_remove_config = config
                else:
                    if config_score <= best_score:
                        best_score = config_score
                        best_remove_config = config
            if best_remove_config is not None:
                print(f'REMOVING: {best_score} | {best_remove_config}')
                zeroshot_configs.remove(best_remove_config)
            else:
                finished_removal = True
        return zeroshot_configs


class ZeroshotEnsembleConfigCV:
    def __init__(self, n_splits, df_results_by_dataset, zeroshot_sim_name, config_scorer, configs: list = None):

        self.n_splits = n_splits
        self.zeroshot_sim_name = zeroshot_sim_name
        self.config_scorer = config_scorer

        # df_results_by_dataset_with_score_val = combine_results_with_score_val(self.df_raw, self.df_results_by_dataset)
        self.unique_datasets = np.array(config_scorer.datasets)

        if configs is None:
            configs = list(df_results_by_dataset['framework'].unique())
        self.configs = configs

        self.kf = KFold(n_splits=self.n_splits, random_state=0, shuffle=True)

    def run(self):
        df_raw_zeroshots = []
        for i, (train_index, test_index) in enumerate(self.kf.split(self.unique_datasets)):
            print(f'Fitting Fold {i+1}...')
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = list(self.unique_datasets[train_index]), list(self.unique_datasets[test_index])
            df_raw_zeroshots.append(self.run_fold(X_train, X_test))
        # df_raw_zeroshots = pd.concat(df_raw_zeroshots)
        return df_raw_zeroshots

    def run_fold(self, X_train, X_test):
        config_scorer_train = self.config_scorer.subset(datasets=X_train)
        config_scorer_test = self.config_scorer.subset(datasets=X_test)

        zs_config_generator = ZeroshotConfigGenerator(config_scorer=config_scorer_train, configs=self.configs)

        zeroshot_configs = zs_config_generator.select_zeroshot_configs(10, removal_stage=False, config_scorer_test=config_scorer_test)
        # deleting
        zeroshot_configs = zs_config_generator.prune_zeroshot_configs(zeroshot_configs, removal_threshold=0)

        score = config_scorer_test.score(zeroshot_configs)
        print(f'score: {score}')

        return score
        # df_raw_zeroshot = get_zeroshot_config_simulation(zeroshot_configs=zeroshot_configs, config_scorer=config_scorer_test, df_raw=df_raw_test, zeroshot_sim_name=self.zeroshot_sim_name)
        # return df_raw_zeroshot
