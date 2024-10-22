import copy
import os
import argparse
import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import QM9
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import *


def select_target(data, target):
    data = data.clone()
    data.y = data.y[:, target]
    return data


def make_complete(data):
    data = data.clone()
    device = data.edge_index.device

    row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
    col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

    row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
    col = col.repeat(data.num_nodes)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = None
    if data.edge_attr is not None:
        idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        size = list(data.edge_attr.size())
        size[0] = data.num_nodes * data.num_nodes
        edge_attr = data.edge_attr.new_zeros(size)
        edge_attr[idx] = data.edge_attr

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    data.edge_attr = edge_attr
    data.edge_index = edge_index

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D models on QM9")
    parser.add_argument(
        "--dim", type=int, default=64, help="Dimension of the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--target", type=int, default=0, help="Target property"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/qm9", help="Data directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--model", type=str, default="Net", help="Model to use"
    )
    parser.add_argument(
        "--heads", type=int, default=4, help="Number of heads in GAT"
    )
    parser.add_argument(
        "--concat", type=bool, default=True, help="Concatenate heads in GAT"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    print(f"Loading data from {args.data_dir}", flush=True)
    dataset = QM9(
        root=args.data_dir,
        transform=T.Compose(
            [
                lambda data: select_target(data, args.target),
                lambda data: make_complete(data),
                T.Distance(norm=False),
            ]
        ),
    )

    print(f"Dataset size: {len(dataset)}", flush=True)

    NUM_FEATURES = dataset.num_features

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset._data.y[:, args.target].mean()
    std = dataset._data.y[:, args.target].std()
    dataset._data.y[:, args.target] = (
        dataset._data.y[:, args.target] - mean
    ) / std
    print(f"Mean: {mean}, Std: {std}", flush=True)
    print(
        f"Normalized mean: {dataset._data.y[:, args.target].mean()}, normalized std: {dataset._data.y[:, args.target].std()}",
        flush=True,
    )

    # split datasets
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"Train size: {len(train_loader.dataset)}", flush=True)
    print(f"Validation size: {len(val_loader.dataset)}", flush=True)
    print(f"Test size: {len(test_loader.dataset)}", flush=True)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if args.model == "Net":
        model = Net(NUM_FEATURES, args.dim).to(device)
    elif args.model == "GAT":
        model = GAT(
            NUM_FEATURES, args.dim, heads=args.heads, concat=args.concat
        ).to(device)
    else:
        raise ValueError(f"Model {args.model} not found")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
    )

    def train(epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_loader.dataset)

    def test(loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                pred = model(data)
                error += (pred * std - data.y * std).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    best_test_error = float("inf")
    best_model_state = None
    pbar = tqdm(range(1, 301), desc=f"Epoch")
    for epoch in pbar:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)

        test_error = test(test_loader)

        if test_error < best_test_error:
            best_test_error = test_error
            best_model_state = copy.deepcopy(model.state_dict())

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        pbar.set_postfix(
            {
                "Epoch": epoch,
                "LR": lr,
                "Loss": loss,
                "Val MAE": val_error,
                "Test MAE": test_error,
                "Best Test MAE": best_test_error,
                "Peak Memory": peak_memory,
            }
        )

    # Save the best model
    # Create save_dir if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    model_filename = os.path.join(
        args.save_dir, f"qm9_net_best_model_{args.target}.pth"
    )
    torch.save(best_model_state, model_filename)
    print(
        f"Best model saved to {model_filename} with Test MAE: {best_test_error:.7f}"
    )
