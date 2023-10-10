from pathlib import Path

from autogluon.common.savers import save_pkl

from tabrepo.simulation.sim_runner import run_zs_sim_end_to_end


if __name__ == '__main__':
    subcontext_name = 'BAG_D244_F3_C1416_small'
    results_cv, repo = run_zs_sim_end_to_end(subcontext_name=subcontext_name,
                                             config_scorer_type='single')

    save_pkl.save(path=str(Path(__file__).parent / 'sim_results' / 'single_best_result_bagged.pkl'), object=results_cv)
