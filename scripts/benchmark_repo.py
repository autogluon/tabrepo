from pathlib import Path

from tabrepo.repository.evaluation_repository_zeroshot import EvaluationRepositoryZeroshot
from tabrepo.utils import catchtime
from tabrepo.contexts import get_subcontext

filepath = Path(__file__)


def compute_all_with_load(subcontext, prediction_format: str):
    with catchtime(f"Load time with   {prediction_format}") as time_load:
        repo = subcontext.load_from_parent(prediction_format=prediction_format)
        repo: EvaluationRepositoryZeroshot = repo.to_zeroshot()
    time_load = time_load()

    with catchtime(f"Simulate with {prediction_format}") as time_sim:
        results_cv = repo.simulate_zeroshot(
            num_zeroshot=5,
            n_splits=2,
            backend="ray",
        )
    time_sim = time_sim()
    results_cv.print_summary()
    return results_cv, time_load, time_sim


if __name__ == '__main__':
    """
    instance_type = "m6i.32xlarge"
    context_name = "BAG_D244_F3_C1416_micro"
    
    mem:
        Time  Load: 54.78s
        Time   Sim: 446.59s
        Train  Err: 6.98507
        Test   Err: 11.54601
    memopt:
        Time  Load: 55.70s
        Time   Sim: 28.82s
        Train  Err: 6.98507
        Test   Err: 11.54601
    memmap:
        Time  Load: 23.09s
        Time   Sim: 35.47s
        Train  Err: 6.98507
        Test   Err: 11.54601
    """
    context_name = "BAG_D244_F3_C1416_micro"
    subcontext = get_subcontext(context_name)

    prediction_formats = [
        "mem",
        "memopt",
        "memmap",
    ]

    time_load_dict = {}
    time_sim_dict = {}
    results_cv_dict = {}

    for prediction_format in prediction_formats:
        results_cv, time_load, time_sim = compute_all_with_load(subcontext=subcontext, prediction_format=prediction_format)
        time_load_dict[prediction_format] = time_load
        time_sim_dict[prediction_format] = time_sim
        results_cv_dict[prediction_format] = results_cv

    for prediction_format in prediction_formats:
        print(f"{prediction_format}:\n"
              f"\tTime  Load: {time_load_dict[prediction_format]:.2f}s\n"
              f"\tTime   Sim: {time_sim_dict[prediction_format]:.2f}s\n"
              f"\tTrain  Err: {results_cv_dict[prediction_format].get_train_score_overall():.5f}\n"
              f"\tTest   Err: {results_cv_dict[prediction_format].get_test_score_overall():.5f}")
