from __future__ import annotations

from tabrepo import EvaluationRepository
from nips2025_utils.sanity_check import sanity_check
from nips2025_utils.fetch_metadata import load_task_metadata
from nips2025_utils.script_constants import tabarena_data_root
from nips2025_utils.generate_repo import generate_repo, generate_repo_from_paths
from tabrepo.utils.pickle_utils import fetch_all_pickles
from autogluon.common.savers import save_pkl
from autogluon.common.loaders import load_pkl
from nips2025_utils.load_artifacts import load_and_check_if_valid


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

S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-tabarena61_reruns
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}





S3_BUCKET=prateek-ag
EXP_NAME=neerick-exp-tabarena61_modernnca_cpu_seq_regression_fix
EXCLUDE=(--exclude "*.log" --exclude "*.json")

EXP_DATA_PATH=${EXP_NAME}/data/
S3_DIR=s3://${S3_BUCKET}/${EXP_DATA_PATH}
USER_DIR=../data/${EXP_DATA_PATH}
echo "${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}"
aws s3 cp --recursive ${S3_DIR} ${USER_DIR} ${EXCLUDE[@]}

"""


def get_file_paths():
    experiment_name = "neerick-exp-tabarena60_big"
    experiment_name_v2 = "neerick-exp-tabarena60_big_v2"
    experiment_name_v3 = "neerick-exp-tabarena61_tabm_modernnca_cpu_seq"
    experiment_name_reruns = "neerick-exp-tabarena61_reruns"

    # Load Context
    expname = f"{tabarena_data_root}/{experiment_name}"  # folder location of results, need to point this to the correct folder
    expname_v2 = f"{tabarena_data_root}/{experiment_name_v2}"
    expname_tabm_modernnca = f"{tabarena_data_root}/{experiment_name_v3}"

    expname_reruns = f"{tabarena_data_root}/{experiment_name_reruns}"

    # file_paths_v1 = fetch_all_pickles(dir_path=expname)
    # file_paths_v2 = fetch_all_pickles(dir_path=expname_v2)
    #
    # file_paths = file_paths_v1 + file_paths_v2
    # file_paths = sorted([str(f) for f in file_paths])
    #
    # save_pkl.save(path="./file_paths_fix.pkl", object=file_paths)
    file_paths = load_pkl.load(path="./file_paths_fix.pkl")

    file_paths_tabm_modernnca = fetch_all_pickles(dir_path=expname_tabm_modernnca)
    file_paths_tabm_modernnca = sorted([str(f) for f in file_paths_tabm_modernnca])

    file_paths_reruns = fetch_all_pickles(dir_path=expname_reruns)
    file_paths_reruns = sorted([str(f) for f in file_paths_reruns])

    file_paths = file_paths + file_paths_tabm_modernnca + file_paths_reruns

    print(len(file_paths))

    file_paths_invalid = [f for f in file_paths if not f.endswith(".pkl")]

    # Fix invalid pkl paths
    file_paths = [f if f.endswith(".pkl") else f.rsplit(".pkl", 1)[0] + ".pkl" for f in file_paths]
    file_paths = sorted(list(set(file_paths)))

    file_paths = remove_duplicate_paths(file_paths=file_paths)
    save_pkl.save(path="./file_paths_full_fix.pkl", object=file_paths)

    expname_tabicl = f"{tabarena_data_root}/2025-05-11-lennart-TabICL"
    expname_tabdpt = f"{tabarena_data_root}/2025-05-11-lennart-TabDPT"
    expname_tabpfnv2 = f"{tabarena_data_root}/2025-05-11-lennart-TabPFNv2"

    file_paths_tabicl = fetch_all_pickles(dir_path=expname_tabicl)
    file_paths_tabicl = sorted([str(f) for f in file_paths_tabicl])

    file_paths_tabdpt = fetch_all_pickles(dir_path=expname_tabdpt)
    file_paths_tabdpt = sorted([str(f) for f in file_paths_tabdpt])

    file_paths_tabpfnv2 = fetch_all_pickles(dir_path=expname_tabpfnv2)
    file_paths_tabpfnv2 = sorted([str(f) for f in file_paths_tabpfnv2])

    file_paths = file_paths + file_paths_tabicl + file_paths_tabdpt + file_paths_tabpfnv2

    file_paths = [f if f.endswith(".pkl") else f.rsplit(".pkl", 1)[0] + ".pkl" for f in file_paths]
    file_paths = sorted(list(set(file_paths)))

    save_pkl.save(path="./file_paths_full_fix_w_gpu.pkl", object=file_paths)
    return file_paths


def remove_duplicate_paths(file_paths: list[str]) -> list[str]:
    file_paths_set = set(file_paths)

    file_paths_clean = []
    for f in file_paths:
        duplicate = False
        if "_big/data/" not in f and "_big/" in f:
            if "_big_v2/data/" in f:
                duplicate = False
            else:
                try:
                    left, right = f.rsplit("_big/", 1)
                except Exception as err:
                    print(f)
                    raise err
                alt_path = left + "_big/data/" + right
                if alt_path in file_paths_set:
                    duplicate = True
                    print(f"DUPLICATE: {duplicate}\t{f}")
                else:
                    duplicate = False

        if not duplicate:
            file_paths_clean.append(f)
    return file_paths_clean


def save_holdout_repo(method: str, file_paths: list[str], repo_dir_prefix: str):
    repo_dir = f"{repo_dir_prefix}/holdout/{method}"

    print(f"Processing holdout for method {method} | {len(file_paths)} files...")
    repo_holdout: EvaluationRepository = generate_repo_from_paths(
        result_paths=file_paths,
        task_metadata=task_metadata,
        as_holdout=True,
    )
    if repo_holdout is not None:
        repo_holdout.to_dir(repo_dir)


if __name__ == '__main__':
    recompute_paths = False

    if recompute_paths:
        file_paths = get_file_paths()
    else:
        file_paths = load_pkl.load("./file_paths_full_fix_w_gpu.pkl")

    repo_dir_prefix = "repos/tabarena61"

    task_metadata = load_task_metadata()

    method_families = [
        # --- Completed ---
        "Dummy",
        "KNeighbors",
        "RealMLP",
        "CatBoost",
        "XGBoost",
        "NeuralNetTorch",
        "RandomForest",
        "ExtraTrees",

        "ExplainableBM",
        "LightGBM",
        "LinearModel",
        "NeuralNetFastAI",  # FIXME: Extra files?

        "TabM",
        "ModernNCA",

        # "TabICL",
        # "TabDPT",
        # "TabPFNv2",

        # # "FTTransformer",  # Excluded
    ]

    file_paths_per_method = {}
    for method in method_families:
        path_contains = f"/{method}_"
        file_paths_per_method[method] = [f for f in file_paths if path_contains in f]

    print("hi")
    for f in file_paths_per_method:
        print(f"{f}: {len(file_paths_per_method[f])}")

    # for method in method_families:
    #     repo_dir = f"{repo_dir_prefix}/{method}"
    #     file_paths_method = file_paths_per_method[method]
    #
    #     print(f"Processing method {method} | {len(file_paths_method)} files...")
    #     repo: EvaluationRepository = generate_repo_from_paths(result_paths=file_paths_method, task_metadata=task_metadata)
    #     repo.to_dir(repo_dir)

    for method in method_families:
        file_paths_method = file_paths_per_method[method]
        save_holdout_repo(
            method=method,
            file_paths=file_paths_method,
            repo_dir_prefix=repo_dir_prefix,
        )

    # if not load_repo:
    #     repo: EvaluationRepository = generate_repo(experiment_path=expname, task_metadata=task_metadata)
    #     # repo = repo.subset(datasets=["abalone", "airfoil_self_noise"])
    #     repo.to_dir(repo_dir)
    # else:
    #     repo = EvaluationRepository.from_dir(repo_dir)

    # sanity_check(repo=repo, fillna=False)
