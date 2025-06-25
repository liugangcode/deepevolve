import sys

is_tty = sys.stdout.isatty()

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import r2_score

## dataset
from preprocessing import convert_to_pytorch_data

## training
from model import GraphEnvAug
from utils import init_weights, get_args, train_with_loss, eval


class Evaluator:
    def __init__(self):
        # These values are from the train data.
        self.MINMAX_DICT = {
            "Tg": [-148.0297376, 472.25],
            "FFV": [0.2269924, 0.77709707],
            "Tc": [0.0465, 0.524],
            "Density": [0.748691234, 1.840998909],
            "Rg": [9.7283551, 34.672905605],
        }
        self.property_names = ["Tg", "FFV", "Tc", "Density", "Rg"]

    def scaling_error(self, labels, preds, property_idx):
        """Compute scaled MAE for a single property"""
        property_name = self.property_names[property_idx]
        error = np.abs(labels - preds)
        min_val, max_val = self.MINMAX_DICT[property_name]
        label_range = max_val - min_val
        return np.mean(error / label_range)

    def get_property_weights(self, labels):
        """Get weights for each property based on valid sample counts"""
        property_weight = []
        for i, property_name in enumerate(self.property_names):
            valid_num = np.sum(~np.isnan(labels[:, i]))
            property_weight.append(valid_num)
        property_weight = np.array(property_weight)
        property_weight = np.sqrt(1 / property_weight)
        return (property_weight / np.sum(property_weight)) * len(property_weight)

    def eval(self, input_dict):
        """
        Compute weighted MAE and R² metrics.

        Args:
            input_dict: Dictionary with keys 'y_true' and 'y_pred'
                       Both should be numpy arrays of shape (n_samples, 5)

        Returns:
            Dictionary with 'wmae', 'r2', and individual 'r2_<property>' keys
        """
        y_true = input_dict["y_true"]  # shape: (n_samples, 5)
        y_pred = input_dict["y_pred"]  # shape: (n_samples, 5)

        # Compute weighted MAE
        property_maes = []
        property_weights = self.get_property_weights(y_true)

        for i, property_name in enumerate(self.property_names):
            # Find valid (non-NaN) samples for this property
            is_labeled = ~np.isnan(y_true[:, i])
            if np.sum(is_labeled) > 0:
                property_mae = self.scaling_error(
                    y_true[is_labeled, i], y_pred[is_labeled, i], i
                )
                property_maes.append(property_mae)
            else:
                property_maes.append(0.0)  # or handle as needed

        if len(property_maes) == 0:
            raise RuntimeError("No labels")

        wmae = float(np.average(property_maes, weights=property_weights))

        # Compute R² for each task and average
        r2_scores = []
        result_dict = {"wmae": wmae}

        for i, property_name in enumerate(self.property_names):
            is_labeled = ~np.isnan(y_true[:, i])
            if np.sum(is_labeled) > 1:  # Need at least 2 samples for R²
                r2 = r2_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                r2_scores.append(r2)
                result_dict[f"r2_{property_name}"] = r2
            else:
                r2_scores.append(0.0)  # or np.nan if preferred
                result_dict[f"r2_{property_name}"] = 0.0

        avg_r2 = np.mean(r2_scores)
        result_dict["r2"] = avg_r2

        return result_dict


def main(args, trial_idx=0, total_trials=1):
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_df = pd.read_csv(os.path.join(args.base_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(args.base_dir, "valid.csv"))
    test_df = pd.read_csv(os.path.join(args.base_dir, "test.csv"))

    train_smiles = train_df["SMILES"].tolist()
    valid_smiles = valid_df["SMILES"].tolist()
    test_smiles = test_df["SMILES"].tolist()

    train_properties = train_df[["Tg", "FFV", "Tc", "Density", "Rg"]].values
    valid_properties = valid_df[["Tg", "FFV", "Tc", "Density", "Rg"]].values
    test_properties = test_df[["Tg", "FFV", "Tc", "Density", "Rg"]].values

    train_data = convert_to_pytorch_data(train_smiles, train_properties)
    valid_data = convert_to_pytorch_data(valid_smiles, valid_properties)
    test_data = convert_to_pytorch_data(test_smiles, test_properties)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    evaluator = Evaluator()

    n_train_data, n_val_data, n_test_data = (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        float(len(test_loader.dataset)),
    )

    model = GraphEnvAug(
        gnn_type=args.gnn,
        num_tasks=5,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        gamma=args.gamma,
        use_linear_predictor=args.use_linear_predictor,
    ).to(device)
    init_weights(model, args.initw_name, init_gain=0.02)

    opt_separator = optim.Adam(
        model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg
    )
    opt_predictor = optim.Adam(
        list(model.graph_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.l2reg,
    )
    optimizers = {"separator": opt_separator, "predictor": opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=100, eta_min=1e-4
            )
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    best_model_state = None

    # Track metrics throughout training
    train_losses = []
    best_valid_perf = None
    final_train_perf = None
    final_valid_perf = None
    final_test_perfs = None

    # Create progress bar for training epochs with trial information
    epoch_desc = f"Trial {trial_idx+1}/{total_trials} - Epochs"
    pbar = tqdm(
        range(args.epochs),
        desc=epoch_desc,
        unit="epoch",
        position=1,
        leave=False,
        disable=not is_tty,
    )

    for epoch in pbar:
        # Update progress bar with current epoch info
        pbar.set_description(
            f"Trial {trial_idx+1}/{total_trials} - Epoch {epoch+1}/{args.epochs}"
        )

        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = "separator"
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = "predictor"

        # Get train loss
        epoch_train_loss = train_with_loss(
            args,
            model,
            device,
            train_loader,
            optimizers,
            optimizer_name,
        )
        train_losses.append(epoch_train_loss)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perfs = eval(args, model, device, train_loader, evaluator)
        valid_perfs = eval(args, model, device, valid_loader, evaluator)

        update_test = False
        if best_valid_perf is None:
            best_valid_perf = valid_perfs
            update_test = True
        else:
            if valid_perfs["wmae"] < best_valid_perf["wmae"]:
                update_test = True

        if update_test or epoch == 0:
            best_valid_perf = valid_perfs
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(args, model, device, test_loader, evaluator)
            final_train_perf = train_perfs
            final_valid_perf = valid_perfs
            final_test_perfs = test_perfs

            # Save the best model parameters
            best_model_state = {
                "separator": model.separator.state_dict(),
                "graph_encoder": model.graph_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
            }
        else:
            cnt_wait += 1
            if cnt_wait > args.patience:
                break

    pbar.close()

    # Return comprehensive metrics
    final_train_loss = (
        train_losses[best_epoch] if best_epoch < len(train_losses) else train_losses[-1]
    )

    return {
        "train_wmae": final_train_perf["wmae"],
        "valid_wmae": final_valid_perf["wmae"],
        "test_wmae": final_test_perfs["wmae"],
        "test_r2_avg": final_test_perfs["r2"],
        "test_r2_Tg": final_test_perfs["r2_Tg"],
        "test_r2_FFV": final_test_perfs["r2_FFV"],
        "test_r2_Tc": final_test_perfs["r2_Tc"],
        "test_r2_Density": final_test_perfs["r2_Density"],
        "test_r2_Rg": final_test_perfs["r2_Rg"],
    }


def config_and_run(args):
    results = {
        "train_wmae": [],
        "valid_wmae": [],
        "test_wmae": [],
        "test_r2_avg": [],
        "test_r2_Tg": [],
        "test_r2_FFV": [],
        "test_r2_Tc": [],
        "test_r2_Density": [],
        "test_r2_Rg": [],
    }

    for trial_idx in range(args.trials):
        trial_results = main(args, trial_idx, args.trials)
        for key, value in trial_results.items():
            results[key].append(value)

    final_results = {}
    for metric, values in results.items():
        final_results[f"{metric}"] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    return final_results, np.mean(results["test_wmae"]), np.mean(results["test_r2_avg"])


if __name__ == "__main__":
    args = get_args()
    args.base_dir = "../../../data_cache/polymer"

    results, wmae, r2 = config_and_run(args)
    print(results)
    print(f"wmae: {wmae:.4f}, r2: {r2:.4f}")
