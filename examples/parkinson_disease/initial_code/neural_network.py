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

        feature_dict = {
            "input": torch.tensor(features),
            "target_norm": torch.tensor(targets),
        }
        return feature_dict

    def __len__(self):
        return len(self.df)


class Net(nn.Module):
    """Neural network model."""

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.cnn = nn.Sequential(
            *(
                [
                    nn.Linear(len(self.cfg.features), cfg.n_hidden),
                    nn.LeakyReLU(),
                ]
                + [
                    nn.Linear(cfg.n_hidden, cfg.n_hidden),
                    nn.LeakyReLU(),
                ]
                * self.cfg.n_layers
            )
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.n_hidden, self.n_classes),
            nn.LeakyReLU(),
        )

    def forward(self, batch):
        input = batch["input"].float()
        y = batch["target_norm"]
        x = input
        x = self.cnn(x)
        preds = self.head(x).squeeze(-1)
        loss = (
            torch.abs(y - preds) / (torch.abs(0.01 + y) + torch.abs(0.01 + preds))
        ).mean()
        return {"loss": loss, "preds": preds, "target_norm": y}


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
        pin_memory=False,
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
        pin_memory=False,
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
        for key, val in output.items():
            val_data[key] += [output[key]]
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
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)
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
