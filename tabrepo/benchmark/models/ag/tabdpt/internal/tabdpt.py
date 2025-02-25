import torch
import numpy as np
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .tabdpt_model import TabDPTModel
from .tabdpt_utils import convert_to_torch_tensor, pad_x, FAISS, seed_everything


class TabDPTEstimator(BaseEstimator):
    def __init__(self, path: str, mode: str, inf_batch_size: int, device: str):
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        checkpoint['cfg']["device"] = self.device
        checkpoint['cfg']["env"]["device"] = self.device
        self.model = TabDPTModel.load(model_state=checkpoint['model'], config=checkpoint['cfg'])
        self.model.eval()
        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        self.imputer = SimpleImputer(strategy='mean')
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.faiss_knn = FAISS(X)
        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.is_fitted_ = True

    def _prepare_prediction(self, X: np.ndarray):
        check_is_fitted(self)
        self.X_test = self.imputer.transform(X)
        self.X_test = self.scaler.transform(self.X_test)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA optionally to reduce the number of features
        if self.n_features > self.max_features:
            U, S, self.V = torch.pca_lowrank(train_x, q=self.max_features)
            train_x = train_x @ self.V
        else:
            self.V = None

        test_x = test_x @ self.V if self.V is not None else test_x
        return train_x, train_y, test_x


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
    def __init__(self, path: str, inf_batch_size: int = 512, device: str = 'cuda:0'):
        super().__init__(path=path, mode='cls', inf_batch_size=inf_batch_size, device=device)

    def fit(self, X, y):
        super().fit(X, y)
        self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes ** i)) % self.max_num_classes
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=0),
                y_src=y_train_digit,
                task='cls',
            )
            digit_preds.append(pred)

        full_pred = torch.zeros((X_test.shape[0], X_test.shape[1], self.num_classes), device=X_train.device)
        for class_idx in range(self.num_classes):
            class_pred = torch.zeros_like(digit_preds[0][:, :, 0])
            for digit_idx, digit_pred in enumerate(digit_preds):
                digit_value = (class_idx // (self.max_num_classes ** digit_idx)) % self.max_num_classes
                class_pred += digit_pred[:, :, digit_value]
            full_pred[:, :, class_idx] = class_pred

        return full_pred

    def predict_proba(self, X: np.ndarray, temperature: float = 0.8, context_size: int = 128):
        train_x, train_y, test_x = self._prepare_prediction(X)

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[:, None, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[:, None, :], self.max_features).to(self.device)
            y_train = train_y[:, None].float()

            if self.num_classes <= self.max_num_classes:
                pred = self.model(
                    x_src=torch.cat([X_train, X_test], dim=0),
                    y_src=y_train,
                    task=self.mode,
                )
            else:
                pred = self._predict_large_cls(X_train, X_test, y_train)

            pred = pred[..., :self.num_classes] / temperature
            pred = torch.nn.functional.softmax(pred, dim=-1)
            return pred.squeeze(1).detach().cpu().numpy()
        else:
            pred_list = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                indices_nni = self.faiss_knn.get_knn_indices(
                    self.X_test[start:end], k=context_size
                )
                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = np.swapaxes(y_nni, 0, 1)

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(0), self.max_features).to(self.device)

                if self.num_classes <= self.max_num_classes:
                    pred = self.model(
                        x_src=torch.cat([X_nni, X_eval], dim=0),
                        y_src=y_nni,
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                pred = pred[..., :self.num_classes] / temperature
                pred = torch.nn.functional.softmax(pred, dim=-1)

                pred_list.append(pred.squeeze(dim=0))

            return torch.cat(pred_list, dim=0).squeeze().detach().cpu().numpy()

    def predict(self, X, temperature: float = 0.8, context_size: int = 128):
        return self.predict_proba(X, temperature=temperature, context_size=context_size).argmax(axis=-1)


class TabDPTRegressor(TabDPTEstimator, RegressorMixin):
    def __init__(self, path: str, inf_batch_size: int = 512, device: str = 'cuda:0'):
        super().__init__(path=path, mode='reg', inf_batch_size=inf_batch_size, device=device)

    def predict(self, X: np.ndarray, context_size: int):
        train_x, train_y, test_x = self._prepare_prediction(X)
        if context_size >= self.n_instances:
            X_train = pad_x(train_x[:, None, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[:, None, :], self.max_features).to(self.device)
            y_train = train_y[:, None].float()
            y_means = y_train.mean(dim=0)
            y_stds = y_train.std(dim=0) + 1e-6
            y_train = (y_train - y_means) / y_stds

            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=0),
                y_src=y_train,
                task=self.mode,
            )

            return pred.squeeze().detach().cpu().numpy() * y_stds.detach().cpu().numpy() + y_means.detach().cpu().numpy()
        else:
            pred_list = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                indices_nni = self.faiss_knn.get_knn_indices(
                    self.X_test[start:end], k=context_size
                )
                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = np.swapaxes(y_nni, 0, 1)

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(0), self.max_features).to(self.device)
                y_means = y_nni.mean(dim=0)
                y_stds = y_nni.std(dim=0) + 1e-6
                y_nni = (y_nni - y_means) / y_stds

                pred = self.model(
                    x_src=torch.cat([X_nni, X_eval], dim=0),
                    y_src=y_nni,
                    task=self.mode,
                )

                pred = pred.squeeze() * y_stds + y_means
                pred_list.append(pred)

            return torch.cat(pred_list).squeeze().detach().cpu().numpy()
