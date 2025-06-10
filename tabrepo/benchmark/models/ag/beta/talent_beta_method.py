from __future__ import annotations

import os.path as osp
import time

import numpy as np
import torch
from autogluon.core.metrics import compute_metric
from tqdm import tqdm

from tabrepo.benchmark.models.ag.beta.deps.talent_data import (
    Dataset,
    data_enc_process,
    data_label_process,
    data_loader_process,
    data_nan_process,
    data_norm_process,
    get_categories,
)
from tabrepo.benchmark.models.ag.beta.deps.talent_methods_base import Method


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


def loss_fn(_loss_fn, y_pred, y_true):
    return _loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(y_pred.shape[1]))


class BetaMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.num_policy == "none"
        assert not is_regression

    def construct_model(self, model_config=None):
        from tabrepo.benchmark.models.ag.beta.talent_beta_model import Beta

        if model_config is None:
            model_config = self.args.config["model"]
        self.model = Beta(
            d_num=self.n_num_features,
            cat_cardinalities=self.categories,
            d_out=self.d_out,
            **model_config,
        ).to(self.args.device)
        self.trlog["best_res"] = 9999

    def data_format(self, is_train=True, N=None, C=None, y=None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = (
                data_nan_process(
                    self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy
                )
            )
            self.y, self.y_info, self.label_encoder = data_label_process(
                self.y, self.is_regression
            )
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = (
                data_enc_process(self.N, self.C, self.args.cat_policy)
            )
            self.n_num_features = self.N["train"].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C["train"].shape[1] if self.C is not None else 0
            self.N, self.normalizer = data_norm_process(
                self.N, self.args.normalization, self.args.seed
            )

            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y["train"]))
            self.C_features = self.C["train"].shape[1] if self.C is not None else 0
            self.categories = get_categories(self.C)
            (
                self.N,
                self.C,
                self.y,
                self.train_loader,
                self.val_loader,
                self.criterion,
            ) = data_loader_process(
                self.is_regression,
                (self.N, self.C),
                self.y,
                self.y_info,
                self.args.device,
                self.args.batch_size,
                is_train=True,
            )

        else:
            N_test, C_test, _, _, _ = data_nan_process(
                N,
                C,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
                self.num_new_value,
                self.imputer,
                self.cat_new_value,
            )
            y_test, _, _ = data_label_process(
                y, self.is_regression, self.y_info, self.label_encoder
            )
            N_test, C_test, _, _, _ = data_enc_process(
                N_test,
                C_test,
                self.args.cat_policy,
                None,
                self.ord_encoder,
                self.mode_values,
                self.cat_encoder,
            )
            N_test, _ = data_norm_process(
                N_test, self.args.normalization, self.args.seed, self.normalizer
            )
            _, _, _, self.test_loader, _ = data_loader_process(
                self.is_regression,
                (N_test, C_test),
                y_test,
                self.y_info,
                self.args.device,
                self.args.batch_size,
                is_train=False,
            )
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test["test"], C_test["test"]
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test["test"]
            else:
                self.N_test, self.C_test = N_test["test"], None
            self.y_test = y_test["test"]

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = (
                self.D.is_binclass,
                self.D.is_multiclass,
                self.D.is_regression,
            )
            self.n_num_features, self.n_cat_features = (
                self.D.n_num_features,
                self.D.n_cat_features,
            )

            self.data_format(is_train=True)
        if config is not None:
            self.reset_stats_withconfig(config)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.config["training"]["lr"],
            weight_decay=self.args.config["training"]["weight_decay"],
        )
        self.train_size = (
            self.N["train"].shape[0] if self.N is not None else self.C["train"].shape[0]
        )
        self.train_indices = torch.arange(self.train_size, device=self.args.device)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return None

        time_cost = 0
        try:
            self.pre_validate()
        except:
            print("Pre-validation failed.")
        self.N_train = self.N["train"].cpu().numpy() if self.N is not None else None
        self.C_train = self.C["train"].cpu().numpy() if self.C is not None else None
        self.y_train = self.y["train"].cpu().numpy()
        self.N_val = self.N["val"].cpu().numpy() if self.N is not None else None
        self.C_val = self.C["val"].cpu().numpy() if self.C is not None else None
        self.y_val = self.y["val"].cpu().numpy()

        loss_list = []

        N_train_size = len(self.y_train)

        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        self._start_time = time.time()

        for epoch in range(self.args.max_epoch):
            tic = time.time()
            import math

            steps = math.ceil(N_train_size / self.args.batch_size)
            if self.args.batch_size > N_train_size / 2:
                steps = 2
            for _step in tqdm(range(steps)):
                self.model.train()
                self.optimizer.zero_grad()
                with autocast(enabled=True, dtype=torch.float32):
                    train_logit, train_label = self.model.train_step(
                        self.N_train,
                        self.C_train,
                        self.y_train,
                        min(self.args.batch_size, N_train_size),
                    )
                    loss = loss_fn(self.criterion, train_logit, train_label)
                loss_list.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if self.early_stop_due_to_timelimit(iteration=epoch):
                    self.continue_training = False
                    break

            elapsed = time.time() - tic
            self.validate(epoch)
            time_cost += elapsed
            print(f"Epoch: {epoch}, Time cost: {elapsed}")
            if not self.continue_training:
                break

        return time_cost

    def early_stop_due_to_timelimit(self, iteration: int) -> bool:
        if iteration > 0 and self.args.time_to_fit_in_seconds is not None:
            pred_time_after_next_epoch = (
                (iteration + 1) / iteration * (time.time() - self._start_time)
            )
            if pred_time_after_next_epoch >= self.args.time_to_fit_in_seconds:
                return True

        return False

    def validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            test_logit, indexs = self.model(
                self.N_val,
                self.C_val,
                self.N_train,
                self.C_train,
                self.y_train,
                is_val=True,
            )
            test_logit = test_logit.cpu()
        test_label = self.y_val
        test_logit = test_logit.mean(1).to(torch.float32).cpu().numpy()

        if self.is_regression:
            assert 0

        validation_score = compute_metric(
            y=test_label,
            metric=self.args.early_stopping_metric,
            y_pred=test_logit if self.is_regression else test_logit.argmax(axis=-1),
            y_pred_proba=test_logit[:, 1] if self.is_binclass else test_logit,
            silent=True,
        )

        if validation_score > self.trlog["best_res"] or (epoch == 0):
            self.trlog["best_res"] = validation_score
            self.trlog["best_epoch"] = epoch
            model_state_dict = self.model.state_dict()

            filtered_state_dict = {
                k: v for k, v in model_state_dict.items() if "TabPFN" not in k
            }

            torch.save(
                {"params": filtered_state_dict},
                osp.join(self.args.save_path, f"best-val-{self.args.seed!s}.pth"),
            )
            np.save(
                osp.join(
                    self.args.save_path, f"best-val-indexs-{self.args.seed!s}.npy"
                ),
                indexs,
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 50:
                self.continue_training = False

        print(
            "best_val_res {}, best_epoch {}".format(
                self.trlog["best_res"], self.trlog["best_epoch"]
            )
        )
        torch.save(self.trlog, osp.join(self.args.save_path, "trlog"))

    def pre_validate(self):
        epoch = -1
        from tabrepo.benchmark.models.ag.beta.talent_tabpfn_model import (
            TabPFNClassifier,
        )

        self.PFN_model = TabPFNClassifier(
            device=self.args.device,
            seed=self.args.seed,
            N_ensemble_configurations=self.args.config["model"]["k"],
        )
        if self.N is not None and self.C is not None:
            X_train = np.concatenate(
                (self.N["train"].cpu().numpy(), self.C["train"].cpu().numpy()), axis=1
            )
            X_val = np.concatenate(
                (self.N["val"].cpu().numpy(), self.C["val"].cpu().numpy()), axis=1
            )
        elif self.N is None and self.C is not None:
            X_train = self.C["train"].cpu().numpy()
            X_val = self.C["val"].cpu().numpy()
        else:
            X_train = self.N["train"].cpu().numpy()
            X_val = self.N["val"].cpu().numpy()
        y_train = self.y["train"].cpu().numpy()
        y_val = self.y["val"].cpu().numpy()
        if y_train.shape[0] > 3000:
            # sampled_X and sampled_Y contain sample_size samples maintaining class proportions for the training set
            from sklearn.model_selection import train_test_split

            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=3000, stratify=y_train
            )
        self.PFN_model.fit(X_train, y_train, overwrite_warning=True)
        y_val_predict = self.PFN_model.predict_proba(X_val)
        y_val_predict = torch.tensor(y_val_predict)
        test_label = torch.tensor(y_val)
        test_logit = y_val_predict.to(torch.float32)
        vl = self.criterion(test_logit, test_label).item()

        if self.is_regression:
            assert 0
        else:
            task_type = "classification"
            # measure = np.greater_equal
            measure = np.less_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        # print(metric_name)
        # assert 0
        print(f"epoch {epoch}, val, loss={vl:.4f} {task_type} result={vres[0]:.4f}")
        if measure(vres[-2], self.trlog["best_res"]) or epoch == -1:
            self.trlog["best_res"] = vres[-2]
            self.trlog["best_epoch"] = epoch
            print("🌸 New best epoch! 🌸")
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 50:
                self.continue_training = False
        print(
            "best_val_res {}, best_epoch {}".format(
                self.trlog["best_res"], self.trlog["best_epoch"]
            )
        )

    def PFN_predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        test_logit = self.PFN_model.predict_proba(Test_X)
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        print(f"Test: loss={vl:.4f}")
        for name, res in zip(metric_name, vres):
            print(f"[{name}]={res:.4f}")
        return vl, vres, metric_name, test_logit

    def predict(self, data, info, model_name):
        if self.trlog["best_epoch"] == -1:
            return self.PFN_predict(data, info, model_name)
        model_path = osp.join(
            self.args.save_path, model_name + f"-{self.args.seed!s}.pth"
        )
        saved_state_dict = torch.load(model_path)["params"]

        filtered_saved_state_dict = {
            k: v for k, v in saved_state_dict.items() if "TabPFN" not in k
        }

        self.model.load_state_dict(filtered_saved_state_dict, strict=False)
        indexs = np.load(
            osp.join(
                self.args.save_path, model_name + f"-indexs-{self.args.seed!s}.npy"
            ),
            allow_pickle=True,
        )
        N, C, y = data
        self.data_format(False, N, C, y)
        self.model.eval()
        import time

        time.time()
        batch_size = 4096
        with torch.no_grad():
            num_test_samples = len(self.y_test)  # .size(0)

            all_test_logit = []

            for start_idx in range(0, num_test_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_test_samples)

                batch_N_test = (
                    self.N_test[start_idx:end_idx] if self.N_test is not None else None
                )
                batch_C_test = (
                    self.C_test[start_idx:end_idx] if self.C_test is not None else None
                )
                batch_logit = self.model(
                    batch_N_test,
                    batch_C_test,
                    self.N_train,
                    self.C_train,
                    self.y_train,
                    is_test=True,
                    indexs=indexs,
                ).cpu()

                all_test_logit.append(batch_logit)

            test_logit = torch.cat(all_test_logit, dim=0)
            return test_logit.mean(1).to(torch.float32)
        # print("Evaluation time cost:", time.time() - start_time)
