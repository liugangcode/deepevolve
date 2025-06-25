import sys

is_tty = sys.stdout.isatty()

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import tqdm

## dataset
from sklearn.model_selection import train_test_split
from dataset import PolymerRegDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

## training
from model import GraphEnvAug
from utils import init_weights, get_args, train, eval, train_with_loss


def main(args, trial_idx=0, total_trials=1):
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if args.dataset.startswith("ogbg"):
        dataset = PygGraphPropPredDataset(name=args.dataset, root="data_cache")

        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=args.batch_size,
            shuffle=False,
        )
        evaluator = Evaluator(args.dataset)

    elif args.dataset.startswith("plym"):
        dataset = PolymerRegDataset(
            name=args.dataset.split("-")[1], root="data_cache"
        )  # PolymerRegDataset
        full_idx = list(range(len(dataset)))
        train_ratio = 0.6
        valid_ratio = 0.1
        test_ratio = 0.3
        train_index, test_index, _, _ = train_test_split(
            full_idx, full_idx, test_size=test_ratio, random_state=42
        )
        train_index, val_index, _, _ = train_test_split(
            train_index,
            train_index,
            test_size=valid_ratio / (valid_ratio + train_ratio),
            random_state=42,
        )

        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)

        train_loader = DataLoader(
            dataset[train_index],
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset[val_index],
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset[test_index],
            batch_size=args.batch_size,
            shuffle=False,
        )
        evaluator = Evaluator("ogbg-molesol")  # RMSE metric

    n_train_data, n_val_data, n_test_data = (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        float(len(test_loader.dataset)),
    )

    model = GraphEnvAug(
        gnn_type=args.gnn,
        num_tasks=dataset.num_tasks,
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
            dataset.task_type,
            optimizer_name,
        )
        train_losses.append(epoch_train_loss)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval(args, model, device, train_loader, evaluator)[0]
        valid_perf = eval(args, model, device, valid_loader, evaluator)[0]
        update_test = False
        if epoch != 0:
            if "classification" in dataset.task_type and valid_perf > best_valid_perf:
                update_test = True
            elif (
                "classification" not in dataset.task_type
                and valid_perf < best_valid_perf
            ):
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(args, model, device, test_loader, evaluator)
            final_train_perf = train_perf
            final_valid_perf = valid_perf
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

    if args.dataset.startswith("ogbg"):
        return {
            "train_bce_loss": final_train_loss,
            "train_auc": final_train_perf,
            "valid_auc": final_valid_perf,
            "test_auc": final_test_perfs[0],
        }
    else:
        return {
            "train_mse_loss": final_train_loss,
            "train_rmse": final_train_perf,
            "valid_rmse": final_valid_perf,
            "test_rmse": final_test_perfs[0],
            "test_r2": final_test_perfs[1],
        }


def config_and_run(args):
    """Alternative version with single progress bar showing total progress"""
    if args.by_default:
        if args.dataset == "plym-o2_prop":
            # oxygen permeability
            args.gamma = 0.2
            args.epochs = 400
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.l2reg = 1e-4
            args.lr = 1e-2
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.patience = 100
        if args.dataset == "plym-mt_prop":
            # melting temperature
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.lr = 1e-2
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
            args.patience = 50
        if args.dataset == "plym-tg_prop":
            # glass temperature
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.initw_name = "orthogonal"
            args.batch_size = 256
            args.lr = 1e-2
            args.patience = 50
        if args.dataset == "plym-density_prop":
            # polymer density
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.3
            args.num_layer = 3
            args.drop_ratio = 0.5
            if args.gnn == "gcn-virtual":
                args.l2reg = 1e-4
            args.batch_size = 32
            args.lr = 1e-3
            args.patience = 50
            args.use_clip_norm = True

        if args.dataset == "ogbg-molhiv":
            args.gamma = 0.1
            args.batch_size = 512
            args.initw_name = "orthogonal"
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.epochs = 100
                args.num_layer = 3
                args.use_clip_norm = True
                args.path_list = [2, 4]
        if args.dataset == "ogbg-molbace":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
                args.l2reg = 7e-4
                args.gamma = 0.55
                args.num_layer = 4
                args.batch_size = 64
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 100
                args.drop_ratio = 0.3
                args.initw_name = "orthogonal"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
                args.patience = 100
                args.initw_name = "orthogonal"
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 128
        if args.dataset == "ogbg-molbbbp":
            args.l2reg = 5e-6
            args.initw_name = "orthogonal"
            args.num_layer = 2
            args.emb_dim = 64
            args.batch_size = 256
            args.use_lr_scheduler = True
            args.gamma = 0.2
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn-virtual"
                args.gamma = 0.4
                args.emb_dim = 128
                args.use_lr_scheduler = False
        if args.dataset == "ogbg-molsider":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.l2reg = 1e-4
            args.patience = 100
            args.gamma = 0.65
            args.num_layer = 5
            args.epochs = 400
        if args.dataset == "ogbg-molclintox":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.use_linear_predictor = True
            args.use_clip_norm = True
            args.gamma = 0.2
            args.patience = 100
            args.batch_size = 64
            args.num_layer = 5
            args.emb_dim = 300
            args.l2reg = 1e-4
            args.epochs = 400
            args.drop_ratio = 0.5
        if args.dataset == "ogbg-moltox21":
            args.gamma = 0.8
        if args.dataset == "ogbg-moltoxcast":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.patience = 50
            args.epochs = 150
            args.l2reg = 1e-5
            args.gamma = 0.7
            args.num_layer = 2

    args.plym_prop = (
        "none"
        if args.dataset.startswith("ogbg")
        else args.dataset.split("-")[1].split("_")[0]
    )

    if args.dataset.startswith("ogbg"):
        results = {
            "train_bce_loss": [],
            "train_auc": [],
            "valid_auc": [],
            "test_auc": [],
        }
    else:
        results = {
            "train_mse_loss": [],
            "train_rmse": [],
            "valid_rmse": [],
            "test_rmse": [],
            "test_r2": [],
        }

    ### >>> DEEPEVOLVE-BLOCK-START: Rename loop variable from 'trail_idx' to 'trial_idx' for clarity
    for trial_idx in range(args.trials):
        trial_results = main(args, trial_idx, args.trials)
        ### <<< DEEPEVOLVE-BLOCK-END
        for key, value in trial_results.items():
            results[key].append(value)

    # Return comprehensive metrics with mean and std
    final_results = {}
    for metric, values in results.items():
        final_results[f"{metric}_mean"] = np.mean(values)
        final_results[f"{metric}_std"] = np.std(values)

    return final_results


if __name__ == "__main__":
    args = get_args()
    results = config_and_run(args)
    print("Results:", results)


