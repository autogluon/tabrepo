import math
from pathlib import Path

import yaml

from autogluon.common.loaders import load_json


def get_amlb_yaml_ag_baseline() -> dict:
    return dict(
        AutoGluon=dict(
            version="latest",
        )
    )


def get_amlb_yaml_template_bag(name, hyperparameters: dict, max_time_limit=None) -> dict:
    yaml_dict = {
        name: dict(
            extends="AutoGluon",
            version="latest",
            params=dict(
                _save_artifacts=['leaderboard', 'zeroshot', 'model_failures'],
                _leaderboard_test=True,
                calibrate=False,
                fit_weighted_ensemble=False,
                num_bag_folds=8,
                num_bag_sets=1,
                hyperparameters=hyperparameters
            )
        )
    }
    if max_time_limit is not None:
        yaml_dict[name]['params']['ag_args_ensemble'] = {'ag.max_time_limit': max_time_limit}
    return yaml_dict


def construct_hyperparameters_dict(configs) -> dict:
    hyperparameters_dict = {}

    for name in configs:
        config = configs[name]
        hyperparameters = config['hyperparameters']
        model_type = config['model_type']

        if model_type not in hyperparameters_dict:
            hyperparameters_dict[model_type] = []

        hyperparameters_dict[model_type].append(hyperparameters)

    return hyperparameters_dict


def bash_dict_to_str(d: dict) -> str:
    name = d['name']
    if 'frameworks_suffix' in d:
        name = f'{name}:{d["frameworks_suffix"]}'
    benchmark = d['benchmark']
    constraint = d['constraint']
    extra_args = d['extra_args']
    bash_str = f"python ../../code/automlbenchmark/runbenchmark.py {name} {benchmark} {constraint} {extra_args}"
    return bash_str


if __name__ == '__main__':
    json_root = Path(__file__).parent.parent / 'data' / 'configs'

    hyperparameters_dict_list = []
    bash_str_list = []

    benchmark = "ag_244"
    constraint = "60h8c"
    extra_args = "-u custom_configs -f 0 1 2 -m aws -p 3000"  # TODO: custom user dir

    resume_path = "s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/results_valid.csv"
    resume_path = None
    if resume_path is not None:
        extra_args += f" --resume --jobhistory {resume_path}"

    frameworks_suffix = 'zeroshot'

    batch_size = 16
    max_time_limit_per_model = 3600

    config_families = [
        'catboost',
        'lightgbm',
        'xgboost',
        'fastai',
        'nn_torch',
        'rf',
        'xt',
    ]
    for family in config_families:
        print(f'\n\n\n==================== {family} ====================')
        config_file_name = f'configs_{family}.json'
        configs = load_json.load(path=str(json_root / config_file_name))
        config_names = list(configs.keys())
        print(config_names)
        hyperparameters_dict = construct_hyperparameters_dict(configs=configs)

        num_configs = len(configs.keys())
        num_batches = math.ceil(num_configs / batch_size)
        print(num_configs)
        print(num_batches)
        for batch in range(num_batches):
            print(batch)
            config_names_batch = config_names[batch_size*batch:batch_size*(batch+1)]
            print(config_names_batch)
            print(len(config_names_batch))
            configs_batch = {k: configs[k] for k in config_names_batch}
            hyperparameters_dict = construct_hyperparameters_dict(configs=configs_batch)
            name = f'ZS_BAG_{family}_b{batch}'

            # print(hyperparameters_dict)

            yaml_dict = get_amlb_yaml_template_bag(
                name=name,
                hyperparameters=hyperparameters_dict,
                max_time_limit=max_time_limit_per_model,
            )

            # print(yaml_dict)

            yaml_string = yaml.dump(yaml_dict)
            # print(yaml_string)

            hyperparameters_dict_list.append(yaml_string)

            bash_dict = dict(
                name=name,
                frameworks_suffix=frameworks_suffix,
                benchmark=benchmark,
                constraint=constraint,
                extra_args=extra_args,
            )
            bash_str = bash_dict_to_str(d=bash_dict)
            bash_str_list.append(bash_str)

    baselines = [
        "AutoGluon_bq:latest",
        "AutoGluon_hq:latest",
        "AutoGluon_mq:latest",
        "AutoGluon_bq_simple:latest",
        "AutoGluon_ezs:latest",
        "AutoGluon_ezsh:latest",
        "H2OAutoML:2023Q2",
        "autosklearn:2023Q2",
        "autosklearn2:2023Q2",
        "AutoWEKA:2023Q2",
        "flaml:2023Q2",
        "NaiveAutoML:2023Q2",
        "GAMA_benchmark:2023Q2",
        "lightautoml:2023Q2",
        "mljarsupervised_benchmark:2023Q2",
        "TPOT:2023Q2",
        "mlr3automl:2023Q2",
        "constantpredictor:2023Q2",
        "RandomForest:2023Q2",
        "TunedRandomForest:2023Q2",
    ]

    baseline_constraints = ["1h8c", "4h8c"]

    bash_str_list_baselines = []
    for baseline_constraint in baseline_constraints:
        for baseline in baselines:
            bash_dict = dict(
                name=baseline,
                benchmark=benchmark,
                constraint=baseline_constraint,
                extra_args=extra_args,
            )
            bash_str = bash_dict_to_str(d=bash_dict)
            bash_str_list_baselines.append(bash_str)

    bash_str_list = bash_str_list_baselines + bash_str_list

    baseline_autogluon_yaml_dict = get_amlb_yaml_ag_baseline()
    yaml_baseline_string = yaml.dump(baseline_autogluon_yaml_dict)

    combined_str = '\n'.join([yaml_baseline_string] + hyperparameters_dict_list)
    # print(combined_str)

    # data = yaml.load(yaml_string, Loader=yaml.Loader)
    # print(data)

    frameworks_zeroshot_yaml_filename = json_root / f'frameworks_{frameworks_suffix}.yaml'

    with open(frameworks_zeroshot_yaml_filename, "w") as text_file:
        text_file.write(combined_str)

    for s in bash_str_list:
        print(f'"{s}"')

    print('')
    print(f'Saved frameworks yaml file to "{frameworks_zeroshot_yaml_filename}"')

    # TODO: Split into batches of 50 per family?
    # TODO: Write the AMLB run code
    "python ../$REPO/runbenchmark.py $FRAMEWORK $BENCHMARK $CONSTRAINT $CUSTOM_USER_DIR $EXTRA_ARGS"
    "cp -r ../../code/automlbenchmark/custom_configs/ ./"
