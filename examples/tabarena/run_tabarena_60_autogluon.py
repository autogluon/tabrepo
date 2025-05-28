from __future__ import annotations

from tabrepo import EvaluationRepository
from tabrepo.nips2025_utils.sanity_check import sanity_check
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.script_constants import tabarena_data_root
from tabrepo.nips2025_utils.generate_repo import generate_repo


"""
# INSTRUCTIONS: First run the below commands in the folder 1 above the tabrepo root folder to fetch the data from S3 (ex: `workspace/code`)
# NOTE: These files are non-public, and cant be accessed without credentials

S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-tabarena60_big
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}

"""


if __name__ == '__main__':
    experiment_name = "neerick-exp-tabarena60_autogluon"

    # Load Context
    expname = f"{tabarena_data_root}/{experiment_name}"  # folder location of results, need to point this to the correct folder
    repo_dir = "repos/tabarena61/AutoGluon"  # location of local cache for fast script running
    load_repo = False  # ensure this is False for the first time running

    task_metadata = load_task_metadata()

    if not load_repo:
        repo: EvaluationRepository = generate_repo(experiment_path=expname, task_metadata=task_metadata)
        repo = repo.subset(baselines=[
            "AutoGluon_bq_4h8c",
            "AutoGluon_bq_1h8c",
            "AutoGluon_bq_5m8c",
        ])
        repo.print_info()
        repo.to_dir(repo_dir)
    else:
        repo = EvaluationRepository.from_dir(repo_dir)

    sanity_check(repo=repo, fillna=False)
