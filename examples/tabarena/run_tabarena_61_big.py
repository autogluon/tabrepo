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

"""


if __name__ == '__main__':
    experiment_name = "neerick-exp-tabarena60_big"
    experiment_name_v2 = "neerick-exp-tabarena60_big_v2"

    # Load Context
    expname = f"{tabarena_data_root}/{experiment_name}"  # folder location of results, need to point this to the correct folder
    expname_v2 = f"{tabarena_data_root}/{experiment_name_v2}"
    repo_dir = "repos/tabarena60_big"  # location of local cache for fast script running
    repo_dir_prefix = "repos/tabarena61"
    load_repo = False  # ensure this is False for the first time running

    task_metadata = load_task_metadata()
    # file_paths_v1 = fetch_all_pickles(dir_path=expname)
    # file_paths_v2 = fetch_all_pickles(dir_path=expname_v2)
    #
    # file_paths = file_paths_v1 + file_paths_v2
    # file_paths = sorted([str(f) for f in file_paths])
    #
    # save_pkl.save(path="./file_paths.pkl", object=file_paths)
    file_paths = load_pkl.load(path="./file_paths.pkl")

    print(len(file_paths))

    file_paths_invalid = [f for f in file_paths if not f.endswith(".pkl")]

    # Fix invalid pkl paths
    file_paths = [f if f.endswith(".pkl") else f.rsplit(".pkl", 1)[0] + ".pkl" for f in file_paths]
    file_paths = sorted(list(set(file_paths)))


    # for f in file_paths_invalid:
    #     str_split = f.split("results.pkl")[0]
    #     str_split = str_split.split(expname + "/")[1]
    #     print()

    method_families = [
        # --- Completed ---
        # "Dummy",
        # "FTTransformer",
        # "KNeighbors",
        # "RealMLP",
        # "CatBoost",
        # "XGBoost",
        # "NeuralNetTorch",
        # "RandomForest",
        # "ExtraTrees",
        # "ExplainableBM",
        # "LightGBM",
        # "LinearModel",
        # "NeuralNetFastAI",  # FIXME: Extra files?
    ]

    file_paths_per_method = {}
    for method in method_families:
        path_contains = f"/{method}_"
        file_paths_per_method[method] = [f for f in file_paths if path_contains in f]

    print("hi")
    for f in file_paths_per_method:
        print(f"{f}: {len(file_paths_per_method[f])}")

    # invalid_files = {}
    # missing_lst = []
    # for method in method_families:
    #     file_paths_method = file_paths_per_method[method]
    #     # file_paths_method = file_paths_method[231300:]
    #     print(f"Processing method {method} | {len(file_paths_method)} files...")
    #     for file_path in file_paths_method:
    #         is_valid = load_and_check_if_valid(file_path)
    #         # print(is_valid)
    #         if not is_valid:
    #             print(file_path)
    #             missing_lst.append(file_path)
    #             # raise AssertionError
    # print(len(missing_lst))

    for method in method_families:
        file_paths_method = file_paths_per_method[method]
        file_paths_method_set = set(file_paths_method)

        file_paths_method_clean = []
        for f in file_paths_method:
            duplicate = False
            if "_big/data/" not in f:
                if "_big_v2/data/" in f:
                    continue
                try:
                    left, right = f.rsplit("_big/", 1)
                except Exception as err:
                    print(f)
                    raise err
                alt_path = left + "_big/data/" + right
                if alt_path in file_paths_method_set:
                    duplicate = True
                    print(f"DUPLICATE: {duplicate}\t{f}")
                else:
                    duplicate = False

            if not duplicate:
                file_paths_method_clean.append(f)
        file_paths_per_method[method] = file_paths_method_clean

    # dataset = "abalone"
    # tid = task_metadata[task_metadata["name"] == dataset].iloc[0]["tid"]
    # # tid = 363628
    # name = "NeuralNetFastAI_c1_BAG_L1"
    # combined = f"{name}/{tid}/1_0/"

    for method in method_families:
        repo_dir = f"{repo_dir_prefix}/{method}"
        file_paths_method = file_paths_per_method[method]

        print(f"Processing method {method} | {len(file_paths_method)} files...")
        repo: EvaluationRepository = generate_repo_from_paths(result_paths=file_paths_method, task_metadata=task_metadata)
        repo.to_dir(repo_dir)




    # if not load_repo:
    #     repo: EvaluationRepository = generate_repo(experiment_path=expname, task_metadata=task_metadata)
    #     # repo = repo.subset(datasets=["abalone", "airfoil_self_noise"])
    #     repo.to_dir(repo_dir)
    # else:
    #     repo = EvaluationRepository.from_dir(repo_dir)

    # sanity_check(repo=repo, fillna=False)
