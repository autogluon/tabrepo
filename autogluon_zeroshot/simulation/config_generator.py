import copy
import time

import numpy as np
import ray
from sklearn.model_selection import KFold


@ray.remote
def score_config_ray(config_scorer, existing_configs, new_config) -> float:
    configs = existing_configs + [new_config]
    score = config_scorer.score(configs)
    return score


class ZeroshotConfigGenerator:
    def __init__(self, config_scorer, configs: list, backend='ray'):
        self.config_scorer = config_scorer
        self.all_configs = configs
        self.backend = backend

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
        if self.backend == 'ray':
            if not ray.is_initialized():
                ray.init()
            config_scorer = ray.put(self.config_scorer)
            selector = self._select_ray
        else:
            config_scorer = self.config_scorer
            selector = self._select_sequential
        while len(zeroshot_configs) < num_zeroshot:
            iteration += 1
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.

            valid_configs = [c for c in self.all_configs if c not in zeroshot_configs]
            if not valid_configs:
                break


            time_start = time.time()
            best_next_config, best_score = selector(valid_configs, zeroshot_configs, config_scorer)
            time_end = time.time()

            zeroshot_configs.append(best_next_config)
            msg = f'{iteration}\t: {round(best_score, 2)} | {round(time_end-time_start, 2)}s | {self.backend}'
            if config_scorer_test:
                score_test = config_scorer_test.score(zeroshot_configs)
                msg += f'\tTest: {round(score_test, 2)}'
            msg += f'\t{best_next_config}'
            print(msg)

        if removal_stage:
            zeroshot_configs = self.prune_zeroshot_configs(zeroshot_configs, removal_threshold=removal_threshold)
        print(f"selected {zeroshot_configs}")
        return zeroshot_configs

    @staticmethod
    def _select_sequential(configs: list, prior_configs: list, config_scorer):
        best_next_config = None
        best_score = 999999999
        for config in configs:
            config_selected = prior_configs + [config]
            config_score = config_scorer.score(config_selected)
            if config_score < best_score:
                best_score = config_score
                best_next_config = config
        return best_next_config, best_score

    @staticmethod
    def _select_ray(configs: list, prior_configs: list, config_scorer):
        # Create and execute all tasks in parallel
        results = []
        for i in range(len(configs)):
            results.append(score_config_ray.remote(
                config_scorer,
                prior_configs,
                configs[i],
            ))
        result = ray.get(results)
        result_idx_min = result.index(min(result))
        best_next_config = configs[result_idx_min]
        best_score = result[result_idx_min]
        return best_next_config, best_score

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


class ZeroshotConfigGeneratorCV:
    def __init__(self,
                 n_splits,
                 df_results_by_dataset,
                 zeroshot_sim_name,
                 config_scorer,
                 unique_datasets_map,
                 configs: list = None):

        self.n_splits = n_splits
        self.zeroshot_sim_name = zeroshot_sim_name
        self.config_scorer = config_scorer
        self.unique_datasets_fold = np.array(config_scorer.datasets)
        self.unique_datasets_map = unique_datasets_map
        self.unique_datasets = set()
        self.dataset_parent_to_fold_map = dict()
        for d in self.unique_datasets_fold:
            dataset_parent = self.unique_datasets_map[d]
            self.unique_datasets.add(dataset_parent)
            if dataset_parent in self.dataset_parent_to_fold_map:
                self.dataset_parent_to_fold_map[dataset_parent].append(d)
            else:
                self.dataset_parent_to_fold_map[dataset_parent] = [d]
        for d in self.dataset_parent_to_fold_map:
            self.dataset_parent_to_fold_map[d] = sorted(self.dataset_parent_to_fold_map[d])
        self.unique_datasets = np.array((sorted(list(self.unique_datasets))))

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
            X_train_fold = []
            X_test_fold = []
            for d in X_train:
                X_train_fold += self.dataset_parent_to_fold_map[d]
            for d in X_test:
                X_test_fold += self.dataset_parent_to_fold_map[d]
            # print(X_train_fold)
            # print(X_test_fold)
            df_raw_zeroshots.append(self.run_fold(X_train_fold, X_test_fold))
        # df_raw_zeroshots = pd.concat(df_raw_zeroshots)
        return df_raw_zeroshots

    def run_fold(self, X_train, X_test):
        config_scorer_train = self.config_scorer.subset(datasets=X_train)
        config_scorer_test = self.config_scorer.subset(datasets=X_test)

        zs_config_generator = ZeroshotConfigGenerator(config_scorer=config_scorer_train, configs=self.configs)

        zeroshot_configs = zs_config_generator.select_zeroshot_configs(10,
                                                                       removal_stage=False,
                                                                       # config_scorer_test=config_scorer_test
                                                                       )
        # deleting
        # FIXME: SPEEDUP WITH RAY
        # zeroshot_configs = zs_config_generator.prune_zeroshot_configs(zeroshot_configs, removal_threshold=0)

        score = config_scorer_test.score(zeroshot_configs)
        print(f'score: {score}')

        return score
        # df_raw_zeroshot = get_zeroshot_config_simulation(zeroshot_configs=zeroshot_configs, config_scorer=config_scorer_test, df_raw=df_raw_test, zeroshot_sim_name=self.zeroshot_sim_name)
        # return df_raw_zeroshot
