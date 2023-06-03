import pandas as pd

import pytest

from autogluon_zeroshot.contexts import get_subcontext
from autogluon_zeroshot.repository import EvaluationRepositoryZeroshot


def verify_result_df(repo: EvaluationRepositoryZeroshot, result_df: pd.DataFrame, name: str):
    assert list(result_df.columns) == [
        'dataset', 'fold', 'framework', 'metric_error', 'time_train_s', 'time_infer_s', 'metric', 'problem_type', 'tid'
    ]

    assert len(result_df) == repo.n_folds() * repo.n_datasets()
    assert len(result_df['framework'].unique()) == 1
    assert result_df['framework'].iloc[0] == name

    assert result_df['metric_error'].min() >= 0

    tids = repo.get_datasets()
    for tid in tids:
        dataset = repo.taskid_to_dataset(tid)
        result_df_dataset = result_df[result_df['dataset'] == dataset]
        assert len(result_df_dataset) == repo.n_folds()
        assert set(list(result_df_dataset['fold'].unique())) == set(list(range(repo.n_folds())))
        assert list(result_df_dataset['tid'].unique()) == [tid]


# TODO: Update so that anyone can run this test without needing private data first
@pytest.mark.skip("skipping for now as takes 20s and requires downloading data from s3, "
                  "we can add it back once we have infrastructure for integration test")
def test_subcontext():
    """
    Tests subcontext and EvaluationRepository logic by loading an EvaluationRepository
    from a subcontext and running a variety of simulations.
    """
    n_folds = 2
    n_models = 20
    n_datasets = 10
    num_zeroshot = 3
    subcontext_name = 'BAG_D244_F10_C608_FULL'
    name = 'test_sim'

    repo = get_subcontext(name=subcontext_name).load_subset(
        n_folds=n_folds,
        n_models=n_models,
        n_datasets=n_datasets,
    ).to_zeroshot()

    assert isinstance(repo, EvaluationRepositoryZeroshot)
    assert repo.n_folds() == n_folds
    assert repo.n_datasets() == n_datasets
    assert repo.n_models() == n_models

    for config_scorer_type in ['single', 'ensemble']:
        for backend in ['seq', 'ray']:
            portfolio_cv = repo.simulate_zeroshot(num_zeroshot=num_zeroshot,
                                                  config_scorer_type=config_scorer_type,
                                                  backend=backend)
            for config_scorer_type_output in ['single', 'ensemble']:
                result_df = repo.generate_output_from_portfolio_cv(portfolio_cv=portfolio_cv,
                                                                   name=name,
                                                                   config_scorer_type=config_scorer_type_output)
                verify_result_df(repo=repo, result_df=result_df, name=name)

    # TODO: Verify test score equivalence to expectation
    # TODO: Verify selected config equivalence too expectation
