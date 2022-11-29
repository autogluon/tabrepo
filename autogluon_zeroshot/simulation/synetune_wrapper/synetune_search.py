import numpy as np
from typing import Optional, List

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialSuggestion, TrialScheduler


class RandomSearch(TrialScheduler):
    def __init__(self, models: List[dict], metric: str, num_base_models: int, train_datasets, test_datasets,
                 num_folds: int, ensemble_size: int, backend: str, initial_suggestions=None):
        """Search configurations by random sampling."""
        super(RandomSearch, self).__init__(config_space={
            "configs": None,
            "train_datasets": None,
            "test_datasets": None,
            "num_folds": None,
            "ensemble_size": None,
            "backend": None,
        })
        self.backend = backend
        self.metric = metric
        self.models = models
        self.num_base_models = num_base_models
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.num_folds = num_folds
        self.ensemble_size = ensemble_size
        self.mode = "min"
        self.initial_suggestions = initial_suggestions if initial_suggestions is not None else []

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        if len(self.initial_suggestions) > 0:
            configs = self.initial_suggestions.pop()
        else:
            configs = [self.models[i] for i in np.random.permutation(len(self.models))[:self.num_base_models]]
        config = {
            "configs": configs,
            "train_datasets": self.train_datasets,
            "test_datasets": self.test_datasets,
            "num_folds": self.num_folds,
            "ensemble_size": self.ensemble_size,
            "backend": self.backend,
        }
        return TrialSuggestion.start_suggestion(config)

    def metric_names(self) -> List[str]:
        return [self.metric]


class LocalSearch(TrialScheduler):
    def __init__(self, models: List[dict], metric: str, num_base_models: int, train_datasets, test_datasets,
                 num_folds: int, ensemble_size: int, backend: str, initial_suggestions=None):
        """Search ensemble configurations by mutating the top configuration."""
        super(LocalSearch, self).__init__(config_space={
            "configs": None,
            "train_datasets": None,
            "test_datasets": None,
            "num_folds": None,
            "ensemble_size": None,
            "backend": None
        })
        self.backend = backend
        self.metric = metric
        self.models = models
        self.num_base_models = num_base_models
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.num_folds = num_folds
        self.ensemble_size = ensemble_size
        self.mode = "min"
        self.initial_suggestions = initial_suggestions if initial_suggestions is not None else []
        self.config_scores = {}

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        if len(self.initial_suggestions) > 0:
            configs = self.initial_suggestions.pop()
        elif len(self.config_scores) == 0:
            # no config sampled yet, sample at random
            configs = [self.models[i] for i in np.random.permutation(len(self.models))[:self.num_base_models]]
        else:
            configs = self.mutate_top_config()
        config = {
            "configs": configs,
            "train_datasets": self.train_datasets,
            "test_datasets": self.test_datasets,
            "num_folds": self.num_folds,
            "ensemble_size": self.ensemble_size,
            "backend": self.backend,
        }
        self.config_scores[tuple(configs)] = None
        return TrialSuggestion.start_suggestion(config)

    def mutate_top_config(self):
        # pick the best configuration and mutate it.
        best_configs = sorted(self.config_scores.items(), key=lambda x: x[1] if x[1] else np.inf)
        best_configs = [x[0] for x in best_configs]
        best_config = list(best_configs[0])

        # mutate the configuration until one that was not seen before is obtained
        def mutate(models):
            res = models.copy()
            res[np.random.randint(0, len(models))] = str(np.random.choice(self.models))
            return res
        num_tries = 10
        for i in range(num_tries):
            new_models = mutate(best_config)
            if tuple(new_models) not in self.config_scores:
                break
        if i == num_tries:
            # in the case where we could not obtain a new configuration in `num_tries`, sample one at random
            new_models = [self.models[i] for i in np.random.permutation(len(self.models))[:self.num_base_models]]

        return new_models

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        self.config_scores[tuple(trial.config['configs'])] = result[self.metric]

    def metric_names(self) -> List[str]:
        return [self.metric]





