from pathlib import Path

from autogluon.common.savers import save_pkl

from autogluon_zeroshot.simulation.sim_runner import run_zs_sim_end_to_end


if __name__ == '__main__':
    subcontext_name = 'BAG_D104_F10_C608_FULL'
    results_cv, repo = run_zs_sim_end_to_end(subcontext_name=subcontext_name,
                                             config_scorer_type='ensemble')

    save_pkl.save(path=str(Path(__file__).parent / 'sim_results' / 'ensemble_result_bagged.pkl'), object=results_cv)
