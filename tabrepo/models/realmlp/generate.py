import numpy as np


def generate_single_config_realmlp(rng, is_classification: bool):
    # common search space
    params = {
        'hidden_sizes': 'rectangular',
        'hidden_width': rng.choice([256, 384, 512]),
        'p_drop': rng.uniform(0.0, 0.5),
        'act': 'mish',
        'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
        'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
        'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
        'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
        'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
        'ls_eps_sched': 'coslog4'
    }

    if is_classification:
        # based on alt9
        params.update({
            'n_hidden_layers': rng.integers(1, 3, endpoint=True),
            'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
            'wd': np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),

            'p_drop_sched': 'constant',
            'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(2e-1))),
        })
    else:
        # based on alt6
        params.update({
            'n_hidden_layers': rng.integers(2, 4, endpoint=True),
            'lr': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
            'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),

            'p_drop_sched': 'flat_cos',
        })

    if rng.uniform(0.0, 1.0) > 0.5:
        # large configs
        # params['plr_hidden_1'] = round(np.exp(rng.uniform(np.log(8), np.log(64))))
        # params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(8), np.log(64))))
        params['plr_hidden_1'] = rng.choice([8, 16, 32, 64])
        params['plr_hidden_2'] = rng.choice([8, 16, 32, 64])
        params['n_epochs'] = rng.choice([256, 512])
        params['use_early_stopping'] = True
    else:
        # default values, used here to always set the same set of parameters
        params['plr_hidden_1'] = 16
        params['plr_hidden_2'] = 4
        params['n_epochs'] = 256
        params['use_early_stopping'] = False

    return params


def generate_configs_realmlp_classification(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_realmlp(rng, is_classification=True) for _ in range(num_random_configs)]


def generate_configs_realmlp_regression(num_random_configs=200, seed=1234):
    rng = np.random.default_rng(seed)
    return [generate_single_config_realmlp(rng, is_classification=False) for _ in range(num_random_configs)]


if __name__ == '__main__':
    for config in generate_configs_realmlp_classification(20):
        print(config)
