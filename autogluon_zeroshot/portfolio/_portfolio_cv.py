from typing import List


# TODO: Consider including more, such as metric, metric_error, so that we don't need to calculate that later
class Portfolio:
    def __init__(self,
                 configs: List[str],
                 train_score: float = None,
                 test_score: float = None,
                 train_datasets: List[str] = None,
                 test_datasets: List[str] = None,
                 train_datasets_fold: List[str] = None,
                 test_datasets_fold: List[str] = None,
                 fold: int = None,
                 split: int = None,
                 repeat: int = None,
                 step: int = None,
                 n_configs_avail: int = None):
        self.configs = configs
        self.train_score = train_score
        self.test_score = test_score
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.train_datasets_fold = train_datasets_fold
        self.test_datasets_fold = test_datasets_fold
        self.fold = fold
        self.split = split
        self.repeat = repeat

        # Optional, the step in which this portfolio was generated
        # (relevant when portfolio is generated through iterative steps)
        self.step = step

        self.n_configs_avail = n_configs_avail  # Number of configs to choose from at fit time


class PortfolioCV:
    """
    Contains a list of Portfolios generated during cross-validation.
    """
    def __init__(self, portfolios: List[Portfolio]):
        self.portfolios = portfolios

    def get_test_score_overall(self):
        total_num_datasets = 0
        total_test_score = 0
        for portfolio in self.portfolios:
            test_score = portfolio.test_score
            num_datasets = len(portfolio.test_datasets_fold)
            total_num_datasets += num_datasets
            total_test_score += test_score*num_datasets
        test_score = total_test_score / total_num_datasets
        return test_score

    def get_train_score_overall(self):
        total_num_datasets = 0
        total_train_score = 0
        for portfolio in self.portfolios:
            train_score = portfolio.train_score
            num_datasets = len(portfolio.train_datasets_fold)
            total_num_datasets += num_datasets
            total_train_score += train_score*num_datasets
        train_score = total_train_score / total_num_datasets
        return train_score

    def are_test_folds_unique(self) -> bool:
        """
        Return True if each test dataset is only present in one fold.
        An example of when this would return False is repeated k-fold.
        """
        seen_datasets_folds = set()
        for p in self.portfolios:
            for d in p.test_datasets_fold:
                if d not in seen_datasets_folds:
                    seen_datasets_folds.add(d)
                else:
                    return False
        return True

    @classmethod
    def combine(cls, portfolio_cv_list: list):
        """
        Combine a list of PortfolioCV objects into one
        """
        portfolios = []
        for p in portfolio_cv_list:
            portfolios += p.portfolios
        return PortfolioCV(portfolios=portfolios)

    def get_test_train_rank_diff(self) -> float:
        """
        Returns the amount of overfitting that has occurred.
        AKA: How overoptimistic the portfolio is on training compared to test
        Lower = Better
        """
        return self.get_test_score_overall() - self.get_train_score_overall()
