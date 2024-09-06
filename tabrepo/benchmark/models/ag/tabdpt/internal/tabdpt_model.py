import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

from .tabdpt_utils import maskmean, maskstd, normalize_data, clip_outliers, seed_everything


class TabDPTModel(nn.Module):
    def __init__(self, dropout: float, n_out: int, nhead: int, nhid: int, ninp: int, nlayers: int, norm_first: bool, num_features: int):
        super().__init__()
        self.n_out = n_out
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(activation="gelu", d_model=ninp, dim_feedforward=nhid, dropout=0.0, nhead=nhead, norm_first=norm_first)
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.y_encoder = nn.Linear(1, ninp)
        self.cls_head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.reg_head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, 1))
        self.task2head = {'cls': self.cls_head, 'reg': self.reg_head}

    @torch.no_grad()
    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        eval_pos = y_src.shape[0]
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=10)

        x_src = torch.nan_to_num(x_src, nan=0)
        x_src = self.encoder(x_src)

        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)
        condition = torch.arange(src.shape[0]).to(src.device) >= eval_pos
        attention_mask = condition.repeat(src.shape[0], 1)

        for layer in self.transformer_encoder:
            src = layer(src, attention_mask)
        pred = self.task2head[task](src)

        return pred[eval_pos:]

    @classmethod
    def load(cls, model_state, config):
        model = TabDPTModel(
            dropout=config['training']['dropout'],
            n_out=config['model']['max_num_classes'],
            nhead=config['model']['nhead'],
            nhid=config['model']['emsize'] * config['model']['nhid_factor'],
            ninp=config['model']['emsize'],
            nlayers=config['model']['nlayers'],
            norm_first=config['model']['norm_first'],
            num_features=config['model']['max_num_features'],
        )

        module_prefix = '_orig_mod.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(config['env']['device'])
        model.eval()
        return model