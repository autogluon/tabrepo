from typing import Union

import einops
import einx
import torch
import torch.nn as nn
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.checkpoint import checkpoint

from ..._internal.config.enums import Task
from ..._internal.models.base import BaseModel
from ..._internal.models.embedding import (Tab2DEmbeddingX, Tab2DEmbeddingYClasses, Tab2DEmbeddingYRegression,
                                           Tab2DQuantileEmbeddingX)


class Tab2D(BaseModel):

    def __init__(
        self,
        dim: int,
        dim_output: int, 
        n_layers: int,
        n_heads: int,
        task: Union[str, Task],
        use_pretrained_weights: bool,
        path_to_weights: str,
    ) -> None:
        
        super().__init__()

        self.dim = dim
        self.dim_output = dim_output
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.task = task

        if type(self.task) == str:
            self.task = Task[self.task]

        self.x_quantile = Tab2DQuantileEmbeddingX(dim)
        self.x_embedding = Tab2DEmbeddingX(dim)


        match self.task:
            case Task.CLASSIFICATION:
                self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)     # type: nn.Module 
            case Task.REGRESSION:
                if self.dim_output == 1:
                    self.y_embedding = Tab2DEmbeddingYRegression(dim)
                else:
                    self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
            case _:
                raise ValueError(f"Task {task} not supported")

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(Layer(dim, n_heads))

        self.final_layer_norm = nn.LayerNorm(dim)

        self.final_layer = nn.Linear(dim, dim_output, bias=True)

        if use_pretrained_weights:
            self.load_state_dict(torch.load(path_to_weights, weights_only=True))
        else:
            self.init_weights()


    def forward(
            self, 
            x_support: torch.Tensor, # (b, n_s, f)
            y_support: torch.Tensor, # (b, n_s)
            x_query: torch.Tensor, # (b, n_q, f)
            padding_features: torch.Tensor, # (b, f), "1" represents padding, "0" represents valid values
            padding_obs_support: torch.Tensor, # (b, n_s)
            padding_obs_query__: torch.Tensor, # (b, n_q)
        ):

        """
        x_support is (batch_size, n_observations_support, n_features)
        y_support is (batch_size, n_observations_support)

        x_query is (batch_size, n_observations_query, n_features)

        returns:

        y_query is (batch_size, n_observations_query, n_classes)

        syntax:
        b = batch size
        s = number of observations
        f = number of features
        d = dimension of embedding
        c = number of classes
        """

        x_query__ = x_query

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        x_support, x_query__ = self.x_quantile(x_support, x_query__, padding_obs_support, padding_features) 
        x_support = self.x_embedding(x_support) # (b, n_s, f, d)
        x_query__ = self.x_embedding(x_query__) # (b, n_q, f, d)
        y_support, y_query__ = self.y_embedding(y_support, padding_obs_support, n_obs_query__) # (b, n_s, 1, d), (b, n_q, 1, d)
        
        support, pack_support = einops.pack((y_support, x_support), 'b s * d') # (b, n_s, f+1, d)
        query__, pack_query__ = einops.pack((y_query__, x_query__), 'b s * d') # (b, n_q, f+1, d)

        padding_features_y = torch.zeros((batch_size, 1), device=padding_features.device, dtype=torch.bool) # (b, 1)
        padding_features, _ = einops.pack((padding_features_y, padding_features), 'b *') # (b, f+1)

        padder_support = Padder(support, padding_obs_support, padding_features)
        padder_query__ = Padder(query__, padding_obs_query__, padding_features)

        support = padder_support.base_to_obs(support) # (n_valid_s, d)
        query__ = padder_query__.base_to_obs(query__)  # (n_valid_q, d)
       
        for layer in self.layers:
            support, query__ = checkpoint(layer, support, query__, padder_support, padder_query__, use_reentrant=False) # (n_valid_s, d), (n_valid_q, d)

        query__ = self.final_layer_norm(query__)
        query__ = self.final_layer(query__) # (n_valid_q, d)

        query__ = padder_query__.obs_to_base(query__) # (b, n_q, f+1, c)
        
        y_query__, x_query__ = einops.unpack(query__, pack_query__, 'b s * c') # (b, n_q, 1, c), (b, n_q, f, c)

        match self.task:
            # output has shape (batch_size, n_observations_query, n_features, n_classes)
            # we want to remove the n_features dimension, and for regression, the n_classes dimension
            case Task.REGRESSION:
                if self.dim_output == 1:
                    y_query__ = y_query__[:, :, 0, 0]
                else:   
                    y_query__ = y_query__[:, :, 0, :]
            case Task.CLASSIFICATION:
                y_query__ = y_query__[:, :, 0, :]
            case _:
                raise ValueError(f"Task {self.task} not supported")

        return y_query__
    

    def init_weights(self) -> None:

        nn.init.normal_(self.x_embedding.x_embedding.weight, mean=0, std=1)
        nn.init.normal_(self.x_embedding.x_embedding.bias, mean=0, std=1)
        nn.init.normal_(self.y_embedding.y_embedding.weight, mean=0, std=1)
        nn.init.normal_(self.y_embedding.y_mask.weight, mean=0, std=1)

        # default PyTorch initialization for everything else


class Padder(torch.nn.Module):

    def __init__(self, x: torch.Tensor, padding_mask: torch.Tensor, feature_mask: torch.Tensor) -> None:
        
        super().__init__()

        self.padding_mask = padding_mask
        self.feature_mask = feature_mask

        n_obs, n_feat = x.shape[1], x.shape[2]

        # Three different compositions:
        # Base: (batch_size, n_observations, n_features, dim)
        # Obs: ((batch_size, n_observations, n_features), dim)
        # Feat: ((batch_size, n_features, n_observations), dim)

        # Obs can be used for attention over the observations,
        # while Feat can be used for attention over the features.
        # Both can be used when the model wants to compute linear layers.

        x_o, self.indices_o, self.cu_seqlens_o, self.max_seqlen_in_batch_o = unpad_input(x, ~self.padding_mask) # x_o: (b*n w/o pad, f, d)

        self.feature_mask_big = einops.repeat(self.feature_mask, 'b f -> b s f', s=n_obs) # (b, n, f)
        self.feature_mask_big, _, _, _ = unpad_input(self.feature_mask_big, ~self.padding_mask) # (b*n w/o pad, f)
        x_of, self.indices_of, self.cu_seqlens_of, self.max_seqlen_in_batch_of = unpad_input(x_o, ~self.feature_mask_big) # x_of: (b*n*f w/o pad, d)

        x_rearranged = einx.rearrange('b s f d -> b f s d', x)
        x_f, self.indices_f, self.cu_seqlens_f, self.max_seqlen_in_batch_f = unpad_input(x_rearranged, ~self.feature_mask) # x_f: (b*f w/o pad, n, d)

        self.padding_mask_big = einops.repeat(self.padding_mask, 'b s -> b f s', f=n_feat) # (b, f, n)
        self.padding_mask_big, _, _, _ = unpad_input(self.padding_mask_big, ~self.feature_mask) # (b*f w/o pad, n)
        x_fo, self.indices_fo, self.cu_seqlens_fo, self.max_seqlen_in_batch_fo = unpad_input(x_f, ~self.padding_mask_big) # x_fo: (b*f*n w/o pad, d)
        
        self.batch_size = x.shape[0]
        self.batch_size_f = x_f.shape[0]
        self.batch_size_o = x_o.shape[0]

        t = torch.arange(self.indices_fo.shape[0]).unsqueeze(1).to(x.device)
        self.obs_to_feat_indices = self.base_to_feat(self.obs_to_base(t)).squeeze(1)
        self.feat_to_obs_indices = self.base_to_obs(self.feat_to_base(t)).squeeze(1)
        pass

    
    def base_to_obs(self, x: torch.Tensor) -> torch.Tensor:

        x = einx.rearrange('b s f d -> b f s d', x)
        x, _, _, _ = unpad_input(x, ~self.feature_mask)
        x, _, _, _ = unpad_input(x, ~self.padding_mask_big)
        return x
    

    def base_to_feat(self, x: torch.Tensor) -> torch.Tensor:

        x, _, _, _ = unpad_input(x, ~self.padding_mask)
        x, _, _, _ = unpad_input(x, ~self.feature_mask_big)
        return x
    

    def obs_to_base(self, x: torch.Tensor) -> torch.Tensor:

        x = pad_input(x, self.indices_fo, self.batch_size_f, self.max_seqlen_in_batch_fo)
        x = pad_input(x, self.indices_f, self.batch_size, self.max_seqlen_in_batch_f)
        x = einx.rearrange('b f s d -> b s f d', x)
        return x
    
    def feat_to_base(self, x: torch.Tensor) -> torch.Tensor:

        x = pad_input(x, self.indices_of, self.batch_size_o, self.max_seqlen_in_batch_of)
        x = pad_input(x, self.indices_o, self.batch_size, self.max_seqlen_in_batch_o)
        return x
    
    def obs_to_feat(self, x: torch.Tensor) -> torch.Tensor:

        x = x[self.obs_to_feat_indices]
        return x
    
    def feat_to_obs(self, x: torch.Tensor) -> torch.Tensor:
            
        x = x[self.feat_to_obs_indices]
        return x
    

class Layer(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int) -> None:
        
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(dim)
        self.attention1 = MultiheadAttention(dim, n_heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim*4, bias=True)
        self.linear2 = nn.Linear(dim*4, dim, bias=True)

        self.layer_norm3 = nn.LayerNorm(dim)
        self.attention2 = MultiheadAttention(dim, n_heads)
        self.layer_norm4 = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(dim, dim*4, bias=True)
        self.linear4 = nn.Linear(dim*4, dim, bias=True)


    def forward(
            self, 
            support: torch.Tensor,
            query__: torch.Tensor, 
            padder_support: Padder,
            padder_query__: Padder,
        ) -> tuple[torch.Tensor, torch.Tensor]:

        """
        Input:
        support in 'obs' format
        query__ in 'obs' format

        Output:
        support in 'obs' format
        query__ in 'obs' format
        """

        support_residual = support
        query___residual = query__

        support = self.layer_norm1(support)
        query__ = self.layer_norm1(query__)

        # attention across rows
        support_att = self.attention1(
            support, support, support, 
            cu_seqlens_q = padder_support.cu_seqlens_fo, max_seqlen_q = padder_support.max_seqlen_in_batch_fo, 
            cu_seqlens_k = padder_support.cu_seqlens_fo, max_seqlen_k = padder_support.max_seqlen_in_batch_fo
        )
        query___att = self.attention1(
            query__, support, support, 
            cu_seqlens_q = padder_query__.cu_seqlens_fo, max_seqlen_q = padder_query__.max_seqlen_in_batch_fo,
            cu_seqlens_k = padder_support.cu_seqlens_fo, max_seqlen_k = padder_support.max_seqlen_in_batch_fo
        )
        
        support = support_residual + support_att
        query__ = query___residual + query___att
        
        support_residual = support
        query___residual = query__

        support = self.layer_norm2(support)
        query__ = self.layer_norm2(query__)

        support = self.linear1(support)
        query__ = self.linear1(query__)

        support = torch.nn.functional.gelu(support)
        query__ = torch.nn.functional.gelu(query__)

        support = self.linear2(support)
        query__ = self.linear2(query__)

        support = support_residual + support
        query__ = query___residual + query__

        support = padder_support.obs_to_feat(support)
        query__ = padder_query__.obs_to_feat(query__)

        support_residual = support
        query___residual = query__

        support = self.layer_norm3(support)
        query__ = self.layer_norm3(query__)

        # attention across features
        support = self.attention2(
            support, support, support, 
            cu_seqlens_q = padder_support.cu_seqlens_of, max_seqlen_q = padder_support.max_seqlen_in_batch_of, 
            cu_seqlens_k = padder_support.cu_seqlens_of, max_seqlen_k = padder_support.max_seqlen_in_batch_of
        )
        query__ = self.attention2(
            query__, query__, query__, 
            cu_seqlens_q = padder_query__.cu_seqlens_of, max_seqlen_q = padder_query__.max_seqlen_in_batch_of,
            cu_seqlens_k = padder_query__.cu_seqlens_of, max_seqlen_k = padder_query__.max_seqlen_in_batch_of
        )

        support = support_residual + support
        query__ = query___residual + query__

        support_residual = support
        query___residual = query__

        support = self.layer_norm4(support)
        query__ = self.layer_norm4(query__)

        support = self.linear3(support)
        query__ = self.linear3(query__)

        support = torch.nn.functional.gelu(support)
        query__ = torch.nn.functional.gelu(query__)

        support = self.linear4(support)
        query__ = self.linear4(query__)

        support = support_residual + support
        query__ = query___residual + query__

        support = padder_support.feat_to_obs(support)
        query__ = padder_query__.feat_to_obs(query__)

        return support, query__


class MultiheadAttention(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int) -> None:
        
        super().__init__()

        self.use_flash_attention = False
        self.dim = dim
        self.n_heads = n_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)

    
    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
        ) -> torch.Tensor:
        """
        b = batch size
        s = number of observations
        f = number of features
        t = flashattention-compressed sequences of (batch, observations, features)
        h = heads
        d = dimension of embedding

        input: (bsf, d)
        output: (bsf, d)
        """

        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        q = einops.rearrange(q, 't (h d) -> t h d', h=self.n_heads) # (tokens, heads, dim), tokens is b*n*f w/o pad
        k = einops.rearrange(k, 't (h d) -> t h d', h=self.n_heads)
        v = einops.rearrange(v, 't (h d) -> t h d', h=self.n_heads)

        output = flash_attn_varlen_func(
            q = q, 
            k = k, 
            v = v, 
            cu_seqlens_q = cu_seqlens_q, # num_seq+1, either b*n (w/o pad)+1, or b*f (w/o pad)+1
            cu_seqlens_k = cu_seqlens_k,
            max_seqlen_q = max_seqlen_q, # max sequence length, either n or f
            max_seqlen_k = max_seqlen_k,
        )

        output = einops.rearrange(output, 't h d -> t (h d)')
        output = self.o(output)

        return output