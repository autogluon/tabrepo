from __future__ import annotations

import io
import os
import pickle
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from tabrepo.benchmark.models.ag.beta.deps.tabpfn import encoders, transformer
from torch.utils.checkpoint import checkpoint


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.0 - num / x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    """Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(
        torch.where(mask, mean_broadcast - x, torch.full_like(x, 0))
    )
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(
        x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare
    )


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + 0.000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + 0.000001
    data = (data - mean) / std
    return torch.clip(data, min=-100, max=100)


def normalize_by_used_features_f(
    x, num_features_used, num_features, normalize_with_sqrt=False
):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    return torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)


def load_model_only_inference(path, filename, device):
    """Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """
    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location="cpu", weights_only=False,
    )

    if (
        (
            "nan_prob_no_reason" in config_sample
            and config_sample["nan_prob_no_reason"] > 0.0
        )
        or (
            "nan_prob_a_reason" in config_sample
            and config_sample["nan_prob_a_reason"] > 0.0
        )
        or (
            "nan_prob_unknown_reason" in config_sample
            and config_sample["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    n_out = config_sample["max_num_classes"]
    device = device if torch.cuda.is_available() else "cpu:0"
    encoder = encoder(config_sample["num_features"], config_sample["emsize"])

    nhid = config_sample["emsize"] * config_sample["nhid_factor"]
    y_encoder_generator = (
        encoders.get_Canonical(config_sample["max_num_classes"])
        if config_sample.get("canonical_y_encoder", False)
        else encoders.Linear
    )

    assert config_sample["max_num_classes"] > 2
    loss = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
    )

    model = transformer.TransformerModel(
        encoder,
        n_out,
        config_sample["emsize"],
        config_sample["nhead"],
        nhid,
        config_sample["nlayers"],
        y_encoder=y_encoder_generator(1, config_sample["emsize"]),
        dropout=config_sample["dropout"],
        efficient_eval_masking=config_sample["efficient_eval_masking"],
    )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float("inf"), float("inf"), model), config_sample  # no loss measured


def load_model_only_inference_regression(path, filename, device):
    """Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """
    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location="cpu"
    )
    # file_path = '/data1/Benchmark/T1/model/models/models_diff/prior_diff_real_checkpoint_multiclass_12_30_2024_23_42_32_n_0_epoch_52.cpkt'
    # model_state, optimizer_state, config_sample = torch.load(file_path, map_location='cpu')
    if (
        (
            "nan_prob_no_reason" in config_sample
            and config_sample["nan_prob_no_reason"] > 0.0
        )
        or (
            "nan_prob_a_reason" in config_sample
            and config_sample["nan_prob_a_reason"] > 0.0
        )
        or (
            "nan_prob_unknown_reason" in config_sample
            and config_sample["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)
    n_out = config_sample["max_num_classes"]
    device = device if torch.cuda.is_available() else "cpu:0"
    encoder = encoder(config_sample["num_features"], config_sample["emsize"])

    nhid = config_sample["emsize"] * config_sample["nhid_factor"]
    y_encoder_generator = (
        encoders.get_Canonical(config_sample["max_num_classes"])
        if config_sample.get("canonical_y_encoder", False)
        else encoders.Linear
    )

    assert config_sample["max_num_classes"] > 2
    loss = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
    )

    model = transformer.TransformerModel(
        encoder,
        n_out,
        config_sample["emsize"],
        config_sample["nhead"],
        nhid,
        config_sample["nlayers"],
        y_encoder=y_encoder_generator(1, config_sample["emsize"]),
        dropout=config_sample["dropout"],
        efficient_eval_masking=config_sample["efficient_eval_masking"],
    )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")
    # y_encoder_generator = y_encoder_generator(1, config_sample['emsize'])
    model.criterion = loss
    module_prefix = "module."
    # model.y_encoder = encoders.get_Regression(1)

    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)

    # model.decoder = nn.Sequential(nn.Linear(512, nhid), nn.GELU(), nn.Linear(nhid, 1))
    model.to(device)
    model.eval()

    return (float("inf"), float("inf"), model), config_sample  # no loss measured


def fix_loaded_config_sample(loaded_config_sample, config):
    def copy_to_sample(*k):
        t, s = loaded_config_sample, config
        for k_ in k[:-1]:
            t = t[k_]
            s = s[k_]
        t[k[-1]] = s[k[-1]]

    copy_to_sample("num_features_used")
    copy_to_sample("num_classes")
    copy_to_sample(
        "differentiable_hyperparameters", "prior_mlp_activations", "choice_values"
    )


def load_config_sample(path, template_config):
    model_state, optimizer_state, loaded_config_sample = torch.load(
        path, map_location="cpu"
    )
    fix_loaded_config_sample(loaded_config_sample, template_config)
    return loaded_config_sample


def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [
        1000,
        2000,
        3000,
        4000,
        5000,
    ]  # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max(
        [X.shape[1] for (_, X, _, _, _, _) in test_datasets]
        + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets]
    )
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits


def get_mlp_prior_hyperparameters(config):
    from tabpfn.priors.utils import gamma_sampler_f

    config = {
        hp: (next(iter(config[hp].values())))
        if type(config[hp]) is dict
        else config[hp]
        for hp in config
    }

    if "random_feature_rotation" not in config:
        config["random_feature_rotation"] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(
            config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"]
        )
        config["init_std"] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(
            config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"]
        )
        config["noise_std"] = noise_std_sampler

    return config


def get_gp_mix_prior_hyperparameters(config):
    return {
        "lengthscale_concentration": config["prior_lengthscale_concentration"],
        "nu": config["prior_nu"],
        "outputscale_concentration": config["prior_outputscale_concentration"],
        "categorical_data": config["prior_y_minmax_norm"],
        "y_minmax_norm": config["prior_lengthscale_concentration"],
        "noise_concentration": config["prior_noise_concentration"],
        "noise_rate": config["prior_noise_rate"],
    }


def get_gp_prior_hyperparameters(config):
    return {
        hp: (next(iter(config[hp].values())))
        if type(config[hp]) is dict
        else config[hp]
        for hp in config
    }


def get_meta_gp_prior_hyperparameters(config):
    from tabpfn.priors.utils import trunc_norm_sampler_f

    config = {
        hp: (next(iter(config[hp].values())))
        if type(config[hp]) is dict
        else config[hp]
        for hp in config
    }

    if "outputscale_mean" in config:
        outputscale_sampler = trunc_norm_sampler_f(
            config["outputscale_mean"],
            config["outputscale_mean"] * config["outputscale_std_f"],
        )
        config["outputscale"] = outputscale_sampler
    if "lengthscale_mean" in config:
        lengthscale_sampler = trunc_norm_sampler_f(
            config["lengthscale_mean"],
            config["lengthscale_mean"] * config["lengthscale_std_f"],
        )
        config["lengthscale"] = lengthscale_sampler

    return config


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


def get_model(
    config,
    device,
    should_train=True,
    verbose=False,
    state_dict=None,
    epoch_callback=None,
):
    import math

    from tabpfn import encoders, priors
    from tabpfn.train import Losses, train

    extra_kwargs = {}
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config["verbose"] = verbose_prior

    if "aggregate_k_gradients" not in config or config["aggregate_k_gradients"] is None:
        config["aggregate_k_gradients"] = math.ceil(
            config["batch_size"]
            * (
                (config["nlayers"] * config["emsize"] * config["bptt"] * config["bptt"])
                / 10824640000
            )
        )

    config["num_steps"] = math.ceil(
        config["num_steps"] * config["aggregate_k_gradients"]
    )
    config["batch_size"] = math.ceil(
        config["batch_size"] / config["aggregate_k_gradients"]
    )
    config["recompute_attn"] = config.get("recompute_attn", False)

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(
            batch_size,
            seq_len,
            num_features,
            hyperparameters,
            device,
            model_proto=model_proto,
            **kwargs,
        ):
            kwargs = {**extra_kwargs, **kwargs}  # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                hyperparameters=hyperparameters,
                num_features=num_features,
                **kwargs,
            )

        return new_get_batch

    if config["prior_type"] == "prior_bag":
        # Prior bag combines priors
        get_batch_gp = make_get_batch(priors.fast_gp)
        get_batch_mlp = make_get_batch(priors.mlp)
        if config.get("flexible"):
            get_batch_gp = make_get_batch(
                priors.flexible_categorical, get_batch=get_batch_gp
            )
            get_batch_mlp = make_get_batch(
                priors.flexible_categorical, get_batch=get_batch_mlp
            )
        prior_bag_hyperparameters = {
            "prior_bag_get_batch": (get_batch_gp, get_batch_mlp),
            "prior_bag_exp_weights_1": 2.0,
        }
        prior_hyperparameters = {
            **get_mlp_prior_hyperparameters(config),
            **get_gp_prior_hyperparameters(config),
            **prior_bag_hyperparameters,
        }
        model_proto = priors.prior_bag
    else:
        if config["prior_type"] == "mlp":
            prior_hyperparameters = get_mlp_prior_hyperparameters(config)
            model_proto = priors.mlp
        elif config["prior_type"] == "gp":
            prior_hyperparameters = get_gp_prior_hyperparameters(config)
            model_proto = priors.fast_gp
        elif config["prior_type"] == "gp_mix":
            prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
            model_proto = priors.fast_gp_mix
        else:
            raise Exception()

        if config.get("flexible"):
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs["get_batch"] = get_batch_base
            model_proto = priors.flexible_categorical

    if config.get("flexible"):
        prior_hyperparameters["normalize_labels"] = True
        prior_hyperparameters["check_is_compatible"] = True
    prior_hyperparameters["prior_mlp_scale_weights_sqrt"] = (
        config["prior_mlp_scale_weights_sqrt"]
        if "prior_mlp_scale_weights_sqrt" in prior_hyperparameters
        else None
    )
    prior_hyperparameters["rotate_normalized_labels"] = (
        config["rotate_normalized_labels"]
        if "rotate_normalized_labels" in prior_hyperparameters
        else True
    )

    use_style = False

    if config.get("differentiable"):
        get_batch_base = make_get_batch(model_proto, **extra_kwargs)
        extra_kwargs = {
            "get_batch": get_batch_base,
            "differentiable_hyperparameters": config["differentiable_hyperparameters"],
        }
        model_proto = priors.differentiable_prior
        use_style = True
    print(f"Using style prior: {use_style}")

    if (
        ("nan_prob_no_reason" in config and config["nan_prob_no_reason"] > 0.0)
        or ("nan_prob_a_reason" in config and config["nan_prob_a_reason"] > 0.0)
        or (
            "nan_prob_unknown_reason" in config
            and config["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if config["max_num_classes"] == 2:
        loss = Losses.bce
    elif config["max_num_classes"] > 2:
        loss = Losses.ce(config["max_num_classes"])

    False if "multiclass_loss_type" not in config else (
        config["multiclass_loss_type"] == "compatible"
    )
    config["multiclass_type"] = config.get("multiclass_type", "rank")
    config["mix_activations"] = config.get("mix_activations", False)

    config["bptt_extra_samples"] = config.get("bptt_extra_samples", None)
    config["eval_positions"] = (
        [int(config["bptt"] * 0.95)]
        if config["bptt_extra_samples"] is None
        else [int(config["bptt"])]
    )

    epochs = 0 if not should_train else config["epochs"]
    # print('MODEL BUILDER', model_proto, extra_kwargs['get_batch'])
    return train(
        model_proto.DataLoader,
        loss,
        encoder,
        style_encoder_generator=encoders.StyleEncoder if use_style else None,
        emsize=config["emsize"],
        nhead=config["nhead"],
        # For unsupervised learning change to NanHandlingEncoder
        y_encoder_generator=encoders.get_Canonical(config["max_num_classes"])
        if config.get("canonical_y_encoder", False)
        else encoders.Linear,
        pos_encoder_generator=None,
        batch_size=config["batch_size"],
        nlayers=config["nlayers"],
        nhid=config["emsize"] * config["nhid_factor"],
        epochs=epochs,
        warmup_epochs=20,
        bptt=config["bptt"],
        gpu_device=device,
        dropout=config["dropout"],
        steps_per_epoch=config["num_steps"],
        single_eval_pos_gen=get_uniform_single_eval_pos_sampler(
            config.get("max_eval_pos", config["bptt"]),
            min_len=config.get("min_eval_pos", 0),
        ),
        load_weights_from_this_state_dict=state_dict,
        aggregate_k_gradients=config["aggregate_k_gradients"],
        recompute_attn=config["recompute_attn"],
        epoch_callback=epoch_callback,
        bptt_extra_samples=config["bptt_extra_samples"],
        train_mixed_precision=config["train_mixed_precision"],
        extra_prior_kwargs_dict={
            "num_features": config["num_features"],
            "hyperparameters": prior_hyperparameters,
            # , 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
            "batch_size_per_gp_sample": config.get("batch_size_per_gp_sample", None),
            **extra_kwargs,
        },
        lr=config["lr"],
        verbose=verbose_train,
        weight_decay=config.get("weight_decay", 0.0),
    )


def load_model(path, filename, device, eval_positions, verbose):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    print("!! Warning: GPyTorch must be installed !!")
    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location="cpu"
    )
    if (
        "differentiable_hyperparameters" in config_sample
        and "prior_mlp_activations" in config_sample["differentiable_hyperparameters"]
    ):
        config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values_used"
        ] = config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values"
        ]
        config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values"
        ] = [
            torch.nn.Tanh
            for k in config_sample["differentiable_hyperparameters"][
                "prior_mlp_activations"
            ]["choice_values"]
        ]

    config_sample["categorical_features_sampler"] = lambda: lambda x: ([], [], [])
    config_sample["num_features_used_in_training"] = config_sample["num_features_used"]
    config_sample["num_features_used"] = lambda: config_sample["num_features"]
    config_sample["num_classes_in_training"] = config_sample["num_classes"]
    config_sample["num_classes"] = 2
    config_sample["batch_size_in_training"] = config_sample["batch_size"]
    config_sample["batch_size"] = 1
    config_sample["bptt_in_training"] = config_sample["bptt"]
    config_sample["bptt"] = 10
    config_sample["bptt_extra_samples_in_training"] = config_sample[
        "bptt_extra_samples"
    ]
    config_sample["bptt_extra_samples"] = None

    # print('Memory', str(get_gpu_memory()))

    model = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model[2].load_state_dict(model_state)
    model[2].to(device)
    # model[2].eval()

    return model, config_sample


def load_model_workflow(
    i,
    e,
    add_name,
    base_path,
    device="cpu",
    eval_addition="",
    only_inference=True,
    if_regression=False,
):
    """Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param i:
    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """

    def get_file(e):
        """Returns the different paths of model_file, model_path and results_file."""
        model_file = (
            f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt"
        )
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(
            base_path,
            f"models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl",
        )
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            print(
                "We have to download the TabPFN, as there is no checkpoint at ",
                model_path,
            )
            print("It has about 100MB, so this might take a moment.")
            import requests

            url = "https://github.com/PriorLabs/TabPFN/raw/refs/tags/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
            # print('hhh')
            r = requests.get(url, allow_redirects=True)
            # print('hhh')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, "wb").write(r.content)
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = (
                    model_file_,
                    model_path_,
                    results_file_,
                )
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception("No checkpoint found at " + str(model_path))

    # print(f'Loading {model_file}')
    if only_inference:
        # print('Loading model that can be used for inference only')
        model, c = load_model_only_inference(base_path, model_file, device)
        if if_regression:
            model, c = load_model_only_inference_regression(
                base_path, model_file, device
            )

    else:
        # until now also only capable of inference
        model, c = load_model(
            base_path, model_file, device, eval_positions=[], verbose=False
        )
    # model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file


def load_model_workflow_reg(
    i, e, add_name, base_path, device="cpu", eval_addition="", only_inference=True
):
    """Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param i:
    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """

    def get_file(e):
        """Returns the different paths of model_file, model_path and results_file."""
        model_file = (
            f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt"
        )
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(
            base_path,
            f"models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl",
        )
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            print(
                "We have to download the TabPFN, as there is no checkpoint at ",
                model_path,
            )
            print("It has about 100MB, so this might take a moment.")
            import requests

            url = "https://github.com/automl/TabPFN/raw/main/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
            print("hhh")
            r = requests.get(url, allow_redirects=True)
            print("hhh")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, "wb").write(r.content)
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = (
                    model_file_,
                    model_path_,
                    results_file_,
                )
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception("No checkpoint found at " + str(model_path))

    # print(f'Loading {model_file}')
    if only_inference:
        # print('Loading model that can be used for inference only')
        model, c = load_model_only_inference_regression(base_path, model_file, device)

    else:
        # until now also only capable of inference
        model, c = load_model(
            base_path, model_file, device, eval_positions=[], verbose=False
        )
    # model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Manager":
            from settings import Manager

            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


import time


def transformer_predict(
    model,
    eval_xs,
    eval_ys,
    eval_position,
    device="cpu",
    max_features=100,
    style=None,
    inference_mode=False,
    num_classes=2,
    extend_features=True,
    normalize_with_test=False,
    normalize_to_ranking=False,
    softmax_temperature=0.0,
    multiclass_decoder="permutation",
    preprocess_transform="mix",
    categorical_feats=None,
    feature_shift_decoder=False,
    N_ensemble_configurations=10,
    batch_size_inference=16,
    differentiable_hps_as_style=False,
    average_logits=True,
    fp16_inference=False,
    normalize_with_sqrt=False,
    seed=0,
    no_grad=True,
    return_logits=False,
    **kwargs,
):
    """:param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    if categorical_feats is None:
        categorical_feats = []
    num_classes = len(torch.unique(eval_ys))

    # N_ensemble_configurations=32
    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        # Initialize results array size S, B, Classes

        # no_grad disables inference_mode, because otherwise the gradients are lost
        inference_mode_call = (
            torch.inference_mode() if inference_mode and no_grad else NOP()
        )
        with inference_mode_call:
            time.time()
            output = model(
                (
                    used_style.repeat(eval_xs.shape[1], 1)
                    if used_style is not None
                    else None,
                    eval_xs,
                    eval_ys.float(),
                ),
                single_eval_pos=eval_position,
            )[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            # else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        # print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))
        # print(output)
        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        if eval_xs.shape[2] > max_features:
            eval_xs = eval_xs[
                :,
                :,
                sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False)),
            ]

        if preprocess_transform != "none":
            if preprocess_transform in {"power", "power_all"}:
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform in {"quantile", "quantile_all"}:
                pt = QuantileTransformer(output_distribution="normal")
            elif preprocess_transform in {"robust", "robust_all"}:
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(
            eval_xs, normalize_positions=-1 if normalize_with_test else eval_position
        )

        # Removing empty features
        eval_xs = eval_xs[:, 0, :]
        sel = [
            len(torch.unique(eval_xs[0 : eval_ys.shape[0], col])) > 1
            for col in range(eval_xs.shape[1])
        ]
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter("error")
        if preprocess_transform != "none":
            eval_xs = eval_xs.cpu().numpy()
            feats = (
                set(range(eval_xs.shape[1]))
                if "all" in preprocess_transform
                else set(range(eval_xs.shape[1])) - set(categorical_feats)
            )
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col : col + 1])
                    trans = pt.transform(eval_xs[:, col : col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col : col + 1] = trans
                except:
                    pass
            eval_xs = torch.tensor(eval_xs).float()
        warnings.simplefilter("default")

        eval_xs = eval_xs.unsqueeze(1)

        # TODO: Caution there is information leakage when to_ranking is used, we should not use it
        eval_xs = (
            remove_outliers(
                eval_xs,
                normalize_positions=-1 if normalize_with_test else eval_position,
            )
            if not normalize_to_ranking
            else normalize_data(to_ranking_low_mem(eval_xs))
        )
        # Rescale X
        eval_xs = normalize_by_used_features_f(
            eval_xs,
            eval_xs.shape[-1],
            max_features,
            normalize_with_sqrt=normalize_with_sqrt,
        )

        return eval_xs.to(device)

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]
    # print(eval_xs[eval_position:])
    model.to(device)

    model.eval()

    import itertools

    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = (
            softmax_temperature
            if softmax_temperature.shape
            else softmax_temperature.unsqueeze(0).repeat(num_styles)
        )
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(num_styles)

    def get_preprocess(i):
        if i == 0:
            return "power_all"
        #            if i == 1:
        #                return 'robust_all'
        if i == 1:
            return "none"
        return None

    preprocess_transform_configurations = (
        ["none", "power_all"]
        if preprocess_transform == "mix"
        else [preprocess_transform]
    )

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = (
        torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    )
    class_shift_configurations = (
        torch.randperm(len(torch.unique(eval_ys)))
        if multiclass_decoder == "permutation"
        else [0]
    )

    ensemble_configurations = list(
        itertools.product(class_shift_configurations, feature_shift_configurations)
    )
    # default_ensemble_config = ensemble_configurations[0]

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(
        itertools.product(
            ensemble_configurations,
            preprocess_transform_configurations,
            styles_configurations,
        )
    )
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    # if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    time.time()
    for ensemble_configuration in ensemble_configurations:
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration

        style_ = (
            style[styles_configuration : styles_configuration + 1, :]
            if style is not None
            else style
        )
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()
        # print(preprocess_transform_configuration)
        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            eval_xs_ = preprocess_input(
                eval_xs_, preprocess_transform=preprocess_transform_configuration
            )
            if no_grad:
                eval_xs_ = eval_xs_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        # eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()
        # print(class_shift_configuration)
        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat(
            [
                eval_xs_[..., feature_shift_configuration:],
                eval_xs_[..., :feature_shift_configuration],
            ],
            dim=-1,
        )

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [
                    eval_xs_,
                    torch.zeros(
                        (
                            eval_xs_.shape[0],
                            eval_xs_.shape[1],
                            max_features - eval_xs_.shape[2],
                        )
                    ).to(device),
                ],
                -1,
            )
        inputs += [eval_xs_]
        labels += [eval_ys_]
    # print(eval_xs_)
    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    # print(inputs[0].shape, labels[0].shape)
    # print('PREPROCESSING TIME', str(time.time() - start))
    outputs = []
    time.time()
    for batch_input, batch_label in zip(inputs, labels):
        # preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        print(batch_input.shape, batch_label.shape)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="None of the inputs have requires_grad=True. Gradients will be None",
            )
            warnings.filterwarnings(
                "ignore",
                message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.",
            )
            if device == "cpu":
                output_batch = checkpoint(
                    predict,
                    batch_input,
                    batch_label,
                    style_,
                    softmax_temperature_,
                    True,
                )
            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(
                        predict,
                        batch_input,
                        batch_label,
                        style_,
                        softmax_temperature_,
                        True,
                    )
        outputs += [output_batch]
    # print('MODEL INFERENCE TIME ('+str(batch_input.device)+' vs '+device+', '+str(fp16_inference)+')', str(time.time()-start))

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration
        output_ = outputs[:, i : i + 1, :]
        output_ = torch.cat(
            [
                output_[..., class_shift_configuration:],
                output_[..., :class_shift_configuration],
            ],
            dim=-1,
        )

        # output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits and not return_logits:
            # transforms every ensemble_configuration into a probability -> equal contribution of every configuration
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    # if average_logits and not return_logits:
    #     output = torch.nn.functional.softmax(output, dim=-1)

    return torch.transpose(output, 0, 1)


def get_params_from_config(c):
    return {
        "max_features": c["num_features"],
        "rescale_features": c["normalize_by_used_features"],
        "normalize_to_ranking": c["normalize_to_ranking"],
        "normalize_with_sqrt": c.get("normalize_with_sqrt", False),
    }
