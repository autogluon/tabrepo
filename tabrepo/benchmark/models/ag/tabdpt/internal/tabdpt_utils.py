import numpy as np
import torch
import random
import os
import faiss


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data


def clip_outliers(data, eval_pos, n_sigma=4):
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def pad_x(X, num_features):
    seq_len, batch_size, n_features = X.shape
    zero_feature_padding = torch.zeros((seq_len, batch_size, num_features - n_features), device=X.device)
    return torch.cat([X, zero_feature_padding], -1)


class FAISS:
    def __init__(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)

    def get_knn_indices(self, queries, k):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs
