from __future__ import annotations

import pathlib
import random
from typing import Literal

import numpy as np
import torch
from torch import nn

from tabrepo.benchmark.models.ag.beta.deps.tabm_utils import (
    ElementwiseAffineEnsemble,
    OneHotEncoding0d,
    _init_scaling_by_sections,
    make_efficient_ensemble,
)
from tabrepo.benchmark.models.ag.beta.deps.tabpfn.utils import load_model_workflow
from tabrepo.benchmark.models.ag.beta.deps.tabr_utils import (
    MLP,
    ResNet,
    make_module,
    make_module1,
)


def _get_first_input_scaling(backbone):
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    if isinstance(backbone, ResNet):
        return backbone.blocks[0][1] if backbone.proj is None else backbone.proj  # type: ignore[code]
    raise RuntimeError(f"Unsupported backbone: {backbone}")


class Beta(nn.Module):
    def __init__(
        self,
        *,
        d_num: int,
        d_out: int,
        backbone: dict,
        cat_cardinalities: list[int],
        num_embeddings: dict | None,
        arch_type: Literal[
            # Active
            "vanilla",  # Simple MLP
            "tabm",  # BatchEnsemble + separate heads + better initialization
            "tabm-mini",  # Minimal: * weight
            # BatchEnsemble
            "tabm-naive",
        ],
        k: None | int = None,
        device="cuda:0",
        base_path=pathlib.Path(__file__).parent.resolve(),
        model_string="",
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.d_num = d_num
        if cat_cardinalities is None:
            cat_cardinalities = []
        scaling_init_sections = []

        if d_num == 0:
            # assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            # assert bins is None
            self.num_module = None
            scaling_init_sections.extend(1 for _ in range(self.d_num))

        else:
            self.num_module = make_module(num_embeddings, n_features=d_num)
            d_num = d_num * num_embeddings["d_embedding"]
            scaling_init_sections.extend(
                num_embeddings["d_embedding"] for _ in range(self.d_num)
            )

        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        scaling_init_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.affine_ensemble = None
        # self.scaling_layer = ScaleLayer(k, d_flat)
        self.backbone = make_module1(d_in=d_flat, **backbone)
        if arch_type != "vanilla":
            assert k is not None
            scaling_init = "random-signs" if num_embeddings is None else "normal"

            if arch_type == "tabm-mini":
                # The minimal possible efficient ensemble.
                self.affine_ensemble = ElementwiseAffineEnsemble(
                    k,
                    d_flat,
                    weight=True,
                    bias=False,
                    weight_init=(
                        "random-signs" if num_embeddings is None else "normal"
                    ),
                )
                _init_scaling_by_sections(
                    self.affine_ensemble.weight,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            elif arch_type == "tabm-naive":
                # The original BatchEnsemble.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init="random-signs",
                )
            elif arch_type == "tabm":
                # Like BatchEnsemble, but all scalings, except for the first one,
                # are initialized with ones.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init="ones",
                )
                _init_scaling_by_sections(
                    _get_first_input_scaling(self.backbone).r,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            else:
                raise ValueError(f"Unknown arch_type: {arch_type}")
        self.arch_type = arch_type
        self.k = k
        self.device = device
        self.style = None
        self.TabPFN, self.c, self.results_file = load_model_workflow(
            0,
            42,
            add_name=model_string,
            base_path=base_path,
            device=device,
            eval_addition="",
            only_inference=True,
        )
        self.TabPFN = self.TabPFN[2]
        self.TabPFN.to(torch.float16)
        for param in self.TabPFN.parameters():
            param.requires_grad = False
        self.max_num_features = self.c["num_features"]
        self.max_num_classes = self.c["max_num_classes"]
        self.differentiable_hps_as_style = self.c["differentiable_hps_as_style"]
        self.index = None

    def train_step(self, x_num, x_cat, y, batch_size):
        num_samples = y.shape[0]
        if batch_size > num_samples / 2:
            batch_size = int(num_samples / 2)
        train_indices = random.sample(range(num_samples), batch_size)
        candidate_indices = list(range(num_samples))  # if i not in train_indices]

        x_num_train = x_num[train_indices] if x_num is not None else None
        x_cat_train = x_cat[train_indices] if x_cat is not None else None
        y_train = torch.tensor(y[train_indices], device=self.device).long()

        x_num_candidate = x_num[candidate_indices] if x_num is not None else None
        x_cat_candidate = x_cat[candidate_indices] if x_cat is not None else None
        y_candidate = y[candidate_indices]
        logits = self.forward(
            x_num_train,
            x_cat_train,
            x_num_candidate,
            x_cat_candidate,
            y_candidate,
            is_train=True,
        )

        return logits, y_train

    def forward(
        self,
        x_num,
        x_cat,
        candidate_x_num,
        candidate_x_cat,
        candidate_y,
        is_train=False,
        is_val=False,
        is_test=False,
        indexs=None,
    ):
        candidate_size = candidate_y.shape[0]
        input = []
        y_input = []
        index_val = []
        if is_train:
            indices = np.random.randint(0, candidate_size, (min(1024, candidate_size),))
            candidate_x_cat_sample = (
                torch.tensor(candidate_x_cat[indices], device=self.device)
                if candidate_x_cat is not None
                else None
            )
            candidate_x_num_sample = (
                torch.tensor(candidate_x_num[indices], device=self.device).float()
                if candidate_x_num is not None
                else None
            )
            candidate_y_sample = (
                torch.tensor(candidate_y[indices], device=self.device).long()
                if candidate_y is not None
                else None
            )
            x_num = (
                torch.tensor(x_num, device=self.device).float()
                if x_num is not None
                else None
            )
            x_cat = (
                torch.tensor(x_cat, device=self.device) if x_cat is not None else None
            )
            x = []
            candidate_x = []
            if x_num is not None:
                x_num_sample = x_num
                x.append(
                    x_num_sample
                    if self.num_module is None
                    else self.num_module(x_num_sample)
                )
                candidate_x.append(
                    candidate_x_num_sample
                    if self.num_module is None
                    else self.num_module(candidate_x_num_sample)
                )
            if x_cat is None:
                assert self.cat_module is None
            else:
                assert self.cat_module is not None
                x_cat_sample = x_cat
                x.append(self.cat_module(x_cat_sample))
                candidate_x.append(self.cat_module(candidate_x_cat_sample))
            x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
            candidate_x = torch.column_stack([x_.flatten(1, -1) for x_ in candidate_x])
            if self.k is not None:
                x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
                candidate_x = candidate_x[:, None].expand(
                    -1, self.k, -1
                )  # (B, D) -> (B, K, D)
                if self.affine_ensemble is not None:
                    x = self.affine_ensemble(x)
                    candidate_x = self.affine_ensemble(candidate_x)
            else:
                assert self.affine_ensemble is None

            x1 = self.backbone(x)
            candidate_x1 = self.backbone(candidate_x)
            for k in range(self.k):
                x = x1[:, k, :]
                candidate_x = candidate_x1[:, k, :]
                input.append(torch.cat([candidate_x, x], dim=0))
                zeroy_expanded = torch.zeros((x.shape[0]), device=self.device)
                y_full = torch.cat([candidate_y_sample, zeroy_expanded], dim=0)
                y_input.append(y_full)
            input = torch.stack(input)
            y_input = torch.stack(y_input)
            input = torch.permute(input, (1, 0, 2)).to(torch.float16)
            y_input = torch.permute(y_input, (1, 0)).to(torch.float16)

            from torch.cuda.amp import autocast

            with autocast():
                logits = self.TabPFN(
                    (None, input, y_input),
                    single_eval_pos=y_input.shape[0] - x.shape[0],
                )[:, :, : self.d_out].to(torch.float)
            # print(logits.dtype)
        else:
            val_logits = []
            x_num = (
                torch.tensor(x_num, device=self.device).float()
                if x_num is not None
                else None
            )
            x_cat = (
                torch.tensor(x_cat, device=self.device) if x_cat is not None else None
            )
            for k in range(self.k):
                if indexs is None:
                    indices = np.random.randint(
                        0, candidate_size, (min(1000, candidate_size),)
                    )
                    index_val.append(indices)
                else:
                    indices = indexs[k]
                candidate_x_cat_sample = (
                    torch.tensor(candidate_x_cat[indices], device=self.device)
                    if candidate_x_cat is not None
                    else None
                )
                candidate_x_num_sample = (
                    torch.tensor(candidate_x_num[indices], device=self.device).float()
                    if candidate_x_num is not None
                    else None
                )
                candidate_y_sample = (
                    torch.tensor(candidate_y[indices], device=self.device).long()
                    if candidate_y is not None
                    else None
                )

                x = []
                candidate_x = []
                if x_num is not None:
                    x_num_sample = x_num
                    x.append(
                        x_num_sample
                        if self.num_module is None
                        else self.num_module(x_num_sample)
                    )
                    candidate_x.append(
                        candidate_x_num_sample
                        if self.num_module is None
                        else self.num_module(candidate_x_num_sample)
                    )
                if x_cat is None:
                    assert self.cat_module is None
                else:
                    assert self.cat_module is not None
                    x_cat_sample = x_cat
                    x.append(self.cat_module(x_cat_sample))
                    candidate_x.append(self.cat_module(candidate_x_cat_sample))
                x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
                candidate_x = torch.column_stack(
                    [x_.flatten(1, -1) for x_ in candidate_x]
                )
                if self.k is not None:
                    x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
                    candidate_x = candidate_x[:, None].expand(
                        -1, self.k, -1
                    )  # (B, D) -> (B, K, D)
                    if self.affine_ensemble is not None:
                        x = self.affine_ensemble(x)
                        candidate_x = self.affine_ensemble(candidate_x)
                else:
                    assert self.affine_ensemble is None

                x1 = self.backbone(x)[:, k, :]
                candidate_x1 = self.backbone(candidate_x)[:, k, :]
                input = [torch.cat([candidate_x1, x1], dim=0)]
                zeroy_expanded = torch.zeros((x.shape[0]), device=self.device)
                y_input = [torch.cat([candidate_y_sample, zeroy_expanded], dim=0)]
                input = torch.stack(input)
                y_input = torch.stack(y_input)
                input = torch.permute(input, (1, 0, 2)).to(torch.float16)
                y_input = torch.permute(y_input, (1, 0)).to(torch.float16)

                val_logits.append(
                    self.TabPFN(
                        (None, input, y_input),
                        single_eval_pos=y_input.shape[0] - x.shape[0],
                    )[:, :, : self.d_out]
                )
                # Clean up for next iteration
                del x, candidate_x, x1, candidate_x1, input, y_input, zeroy_expanded
                torch.cuda.empty_cache()

            logits = torch.cat(val_logits, dim=1)
        if is_val:
            return logits, index_val
        return logits
