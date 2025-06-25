import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict
from tqdm import tqdm
import gc
import os
import random
from base_model import BaseModel
from metrics import smape1p

torch.set_num_threads(1)


class CustomDataset(Dataset):
    """Custom dataset for neural network training."""

    def __init__(self, df, cfg, aug, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.features = df[cfg.features].values
        if self.mode != "test":
            self.targets = df[self.cfg.target_column].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(df))

    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]
        patient_id = self.df["patient_id"].iloc[idx]
        feature_dict = {
            "input": torch.as_tensor(features, dtype=torch.float32),
            "target_norm": torch.as_tensor(targets, dtype=torch.float32),
            "patient_id": torch.as_tensor(patient_id, dtype=torch.long),
        }
        return feature_dict

    def __len__(self):
        return len(self.df)


### >>> DEEPEVOLVE-BLOCK-START: Replace feed-forward network with Neural CDE model for time-series dynamics
class NeuralCDEFunc(nn.Module):
    def __init__(self, hidden_channels, control_channels):
        super(NeuralCDEFunc, self).__init__()
        self.linear = nn.Linear(hidden_channels, hidden_channels * control_channels)
        self.hidden_channels = hidden_channels
        self.control_channels = control_channels

    def forward(self, t, z):
        out = self.linear(z)
        out = out.view(z.size(0), self.hidden_channels, self.control_channels)
        return out


class Net(nn.Module):
    """Neural CDE model for Parkinson's progression prediction."""

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        # DEBUG: propagate use_cat and use_protein flags from cfg to instance for optional embeddings
        self.use_cat = cfg.use_cat
        self.use_protein = cfg.use_protein
        # Assume the last feature is 'horizon_scaled'; the remaining features form the control signal.
        self.input_channels = len(cfg.features) - 1
        self.hidden_channels = cfg.n_hidden
        # Encoder: map control features (excluding horizon) to the initial hidden state.
        self.encoder = nn.Linear(self.input_channels, self.hidden_channels)
        # Neural CDE function that defines the dynamics.
        self.func = NeuralCDEFunc(self.hidden_channels, self.input_channels)
        # Final fully-connected layer to produce the forecast.
        self.fc = nn.Linear(self.hidden_channels, 1)
        # Add a dropout layer for MC dropout uncertainty estimation.
        self.dropout = nn.Dropout(self.cfg.mc_dropout_prob)

    def calibrate(self, predictions):
        return predictions * self.cfg.calib_factor

    def forward(self, batch):
        import warnings

        # If optional embeddings are enabled but the corresponding keys are missing, warn the user.
        if self.use_cat and "cat_input" not in batch:
            warnings.warn(
                "cfg.use_cat is enabled but 'cat_input' is not present in the batch."
            )
        if self.use_protein and "protein_input" not in batch:
            warnings.warn(
                "cfg.use_protein is enabled but 'protein_input' is not present in the batch."
            )
        x = batch["input"].float()  # shape: (batch, feature_dim)
        y = batch["target_norm"]
        ### <<< DEEPEVOLVE-BLOCK-END
        # Split the input: last column holds 'horizon_scaled'
        horizon = x[:, -1].unsqueeze(1)  # shape: (batch, 1)
        control_features = x[:, :-1]  # shape: (batch, input_channels)
        # Compute the initial hidden state from the control features.
        z0 = self.encoder(control_features)  # shape: (batch, hidden_channels)
        # Construct a simple 2-point control path:
        # At time t=0, use the raw control_features.
        # At time t=1, add the (scaled) horizon information to induce temporal evolution.
        p0 = control_features
        p1 = control_features + horizon.repeat(1, control_features.size(1))
        control_path = torch.stack([p0, p1], dim=1)  # shape: (batch, 2, input_channels)
        # DEBUG: Replaced Neural CDE integration with simplified Euler update to avoid torchcde dependency
        # Compute control path endpoints
        p0 = control_features  # at t=0
        p1 = control_features + horizon.repeat(1, control_features.size(1))  # at t=1
        delta_p = p1 - p0  # control increments
        # Compute derivative of state: f at initial time
        f0 = self.func(0.0, z0)  # shape: (batch, hidden_channels, control_channels)
        # Multiply f0 by control derivative to get state derivative
        dZ = (f0 * delta_p.unsqueeze(1)).sum(dim=2)  # shape: (batch, hidden_channels)
        # One-step Euler integration
        z_final = z0 + dZ
        # Compute a single forward pass prediction.
        pred_single = self.fc(z_final).squeeze(-1)
        if self.training:
            preds = pred_single
            uncertainty = None
        else:
            if self.cfg.mc_dropout:
                preds_list = []
                for i in range(self.cfg.mc_dropout_samples):
                    # Apply dropout manually (ensuring dropout is active even in eval mode)
                    z_sample = torch.nn.functional.dropout(
                        z_final, p=self.cfg.mc_dropout_prob, training=True
                    )
                    preds_list.append(self.fc(z_sample).squeeze(-1))
                preds = torch.mean(torch.stack(preds_list), dim=0)
                uncertainty = torch.std(torch.stack(preds_list), dim=0)
            else:
                preds = pred_single
                uncertainty = None
            preds = self.calibrate(preds)
        ### >>> DEEPEVOLVE-BLOCK-START: Add PINN regularization and adaptive loss weighting for Enhanced Meta-Neural CDE
        mse_loss = torch.mean((preds - y) ** 2)
        smape_loss = torch.mean(
            torch.abs(preds - y) / (torch.abs(preds) + torch.abs(y) + 1e-6)
        )
        # PINN-inspired regularization: enforce smooth state transitions by penalizing abrupt changes in state (dZ)
        physics_loss = torch.mean(torch.abs(dZ))
        # Compute adaptive loss weights using the inverse of each loss value (detached to avoid gradient flow)
        eps = 1e-6
        w_smape = 1.0 / (smape_loss.detach() + eps)
        w_mse = 1.0 / (mse_loss.detach() + eps)
        w_physics = 1.0 / (physics_loss.detach() + eps)
        total_weight = w_smape + w_mse + w_physics
        alpha = w_smape / total_weight  # weight for SMAPE loss
        gamma = w_mse / total_weight  # weight for MSE loss
        beta = w_physics / total_weight  # weight for PINN regularization term
        loss = alpha * smape_loss + gamma * mse_loss + beta * physics_loss
        return {
            "loss": loss,
            "preds": preds,
            "target_norm": y,
            "uncertainty": uncertainty,
        }


### <<< DEEPEVOLVE-BLOCK-END


### <<< DEEPEVOLVE-BLOCK-END


def worker_init_fn(worker_id):
    """Worker initialization function."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg, verbose):
    """Get training dataloader."""
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device == "cuda" else False,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg, verbose):
    """Get validation dataloader."""
    sampler = SequentialSampler(val_ds)
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device == "cuda" else False,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    """Get learning rate scheduler."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def set_seed(seed=1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def batch_to_device(batch, device):
    """Move batch to device."""
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def run_eval(model, val_dataloader, cfg, pre="val", verbose=True):
    """Run model evaluation."""
    model.eval()
    torch.set_grad_enabled(False)
    val_data = defaultdict(list)
    if verbose:
        progress_bar = tqdm(val_dataloader)
    else:
        progress_bar = val_dataloader
    for data in progress_bar:
        batch = batch_to_device(data, cfg.device)
        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)
        ### >>> DEEPEVOLVE-BLOCK-START: Correct accumulation of outputs in run_eval
        for key, val in output.items():
            val_data[key] += [val]
    ### <<< DEEPEVOLVE-BLOCK-END
    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0)

    preds = val_data["preds"].cpu().numpy()
    if (pre == "val") and verbose:
        metric = smape1p(100 * val_data["target_norm"].cpu().numpy(), 100 * preds)
        print(f"{pre}_metric 1 ", metric)
        metric = smape1p(
            100 * val_data["target_norm"].cpu().numpy(), np.round(100 * preds)
        )
        print(f"{pre}_metric 2 ", metric)

    return 100 * preds


def run_train(cfg, train_df, val_df, test_df=None, verbose=True):
    """Run model training."""

    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    if verbose:
        print("seed", cfg.seed)
    set_seed(cfg.seed)

    train_dataset = CustomDataset(train_df, cfg, aug=None, mode="train")
    train_dataloader = get_train_dataloader(train_dataset, cfg, verbose)

    if val_df is not None:
        val_dataset = CustomDataset(val_df, cfg, aug=None, mode="val")
        val_dataloader = get_val_dataloader(val_dataset, cfg, verbose)

    if test_df is not None:
        test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
        test_dataloader = get_val_dataloader(test_dataset, cfg, verbose)

    model = Net(cfg)
    model.to(cfg.device)

    total_steps = len(train_dataset)
    params = model.parameters()
    ### >>> DEEPEVOLVE-BLOCK-START: Introduce weight decay for regularization in Adam optimizer
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=1e-4)
    ### <<< DEEPEVOLVE-BLOCK-END
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        set_seed(cfg.seed + epoch)
        if verbose:
            print("EPOCH:", epoch)
            progress_bar = tqdm(range(len(train_dataloader)))
        else:
            progress_bar = range(len(train_dataloader))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()

        for itr in progress_bar:
            i += 1
            data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            batch = batch_to_device(data, cfg.device)
            if cfg.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)
            loss = output_dict["loss"]
            ### >>> DEEPEVOLVE-BLOCK-START: Incorporate meta-learning patient-level loss adaptation in training loop
            if cfg.meta_learning:
                patient_ids = batch["patient_id"]
                unique_patients = torch.unique(patient_ids)
                meta_loss = 0
                count = 0
                for pid in unique_patients:
                    mask = patient_ids == pid
                    if mask.sum() > 0:
                        loss_patient = torch.mean(
                            (output_dict["preds"][mask] - batch["target_norm"][mask])
                            ** 2
                        ) + torch.mean(
                            torch.abs(
                                output_dict["preds"][mask] - batch["target_norm"][mask]
                            )
                            / (
                                torch.abs(output_dict["preds"][mask])
                                + torch.abs(batch["target_norm"][mask])
                                + 1e-6
                            )
                        )
                        grads = torch.autograd.grad(
                            loss_patient,
                            model.parameters(),
                            retain_graph=True,
                            create_graph=True,
                        )
                        grad_norm = sum([torch.sum(g**2) for g in grads])
                        meta_loss += loss_patient + cfg.inner_lr * grad_norm
                        count += 1
                loss = meta_loss / count if count > 0 else loss
            ### <<< DEEPEVOLVE-BLOCK-END
            losses.append(loss.item())
            if cfg.mixed_precision:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        if val_df is not None:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                run_eval(model, val_dataloader, cfg, pre="val", verbose=verbose)

    if test_df is not None:
        return run_eval(model, test_dataloader, cfg, pre="test", verbose=verbose)
    else:
        return model


def run_test(model, cfg, test_df):
    """Run model testing."""
    test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
    test_dataloader = get_val_dataloader(test_dataset, cfg, verbose=False)
    return run_eval(model, test_dataloader, cfg, pre="test", verbose=False)


class NNRegModel1(BaseModel):
    """Neural network regression model."""

    def __init__(self, cfg, features=None):
        self.cfg = cfg

    def fit(self, df_train):
        self.models = [
            run_train(self.cfg, df_train, None, None, verbose=False)
            for _ in range(self.cfg.bag_size)
        ]
        return self

    def predict(self, df_valid):
        preds = np.vstack(
            [run_test(model, self.cfg, df_valid) for model in self.models]
        )
        if self.cfg.bag_agg_function == "max":
            return np.max(preds, axis=0)
        elif self.cfg.bag_agg_function == "median":
            return np.median(preds, axis=0)
        else:
            return np.mean(preds, axis=0)


