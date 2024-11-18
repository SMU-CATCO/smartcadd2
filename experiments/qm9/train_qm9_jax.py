import os
import argparse
import sys
import jax
import jax.profiler
import jax.numpy as jnp
import jax.tree_util as tree
import haiku as hk
import jraph
import optax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from functools import partial

# jax.disable_jit()

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import *
from data_loaders import qmx_batch_to_jraph, pad_graph_to_max_size
import utils

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
    parser.add_argument(
        "--pad_to_power_of_two",
        type=bool,
        default=True,
        help="Pad to power of two",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.00001, help="Minimum learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=None, help="Maximum number of nodes"
    )
    parser.add_argument(
        "--max_edges", type=int, default=None, help="Maximum number of edges"
    )
    return parser.parse_args()

def load_and_preprocess_data(args):
    print(f"Loading data from {args.data_dir}", flush=True)
    pyg_dataset = QM9(
        root=args.data_dir,
        transform=T.Compose(
            [
                lambda data: utils.select_target(data, args.target),
                lambda data: utils.make_complete(data),
                T.Distance(norm=False),
            ]
        ),
    )

    print(f"Dataset size: {len(pyg_dataset)}", flush=True)

    # Normalize targets to mean = 0 and std = 1.
    mean = pyg_dataset._data.y[:, args.target].mean()
    std = pyg_dataset._data.y[:, args.target].std()
    pyg_dataset._data.y[:, args.target] = (
        pyg_dataset._data.y[:, args.target] - mean
    ) / std

    print(f"Mean: {mean}, Std: {std}", flush=True)
    print(
        f"Normalized mean: {pyg_dataset._data.y[:, args.target].mean()}, normalized std: {pyg_dataset._data.y[:, args.target].std()}",
        flush=True,
    )

    # split datasets
    train_dataset, test_dataset = train_test_split(
       pyg_dataset, test_size=0.2, random_state=42
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

    # get max nodes and edges
    if args.max_nodes is not None and args.max_edges is not None:
        max_nodes = args.max_nodes
        max_edges = args.max_edges
    else:
        max_nodes = max(data.x.shape[0] for data in pyg_dataset)
        max_edges = max(data.edge_index.shape[1] for data in pyg_dataset)

    print(f"Max nodes: {max_nodes}, Max edges: {max_edges}", flush=True)

    print(f"Train size: {len(train_loader) * args.batch_size}", flush=True)
    print(f"Validation size: {len(val_loader) * args.batch_size}", flush=True)
    print(f"Test size: {len(test_loader) * args.batch_size}", flush=True)

    return train_loader, val_loader, test_loader, float(std), max_nodes, max_edges


def create_model(model_name, dim, heads, concat=True):
    net_fn = None
    if model_name == "GAT":
        net_fn = lambda graph: GAT(
            dim=dim, out_dim=1, num_heads=heads, num_layers=3, concat=concat
        )(graph)

    elif model_name == "EGNN":
        net_fn = lambda graph: EGNN(
            hidden_size=dim, output_size=1, num_layers=3
        )(graph)
    
    elif model_name == "SchNet":
        net_fn = lambda graph: SchNet(
            n_atom_basis=dim, max_z=9, n_gaussians=25, n_filters=dim, mean=0.0, stddev=20.0, r_cutoff=5.0, n_interactions=2, per_atom=True
        )(graph)

    return hk.without_apply_rng(hk.transform(net_fn))


@partial(jax.jit, static_argnames=["model_fn"])
def loss_fn(
    params,
    graph,
    target,
    model_fn,
):
    pred = model_fn(params, graph)
    mask = jraph.get_graph_padding_mask(graph)
    errors = jnp.square(pred - target)
    return (errors * mask).sum() / mask.sum()


@partial(jax.jit, static_argnames=["optimizer_update", "model_fn"])
def update(
    params,
    opt_state,
    graph,
    target,
    scheduler_scale,
    optimizer_update,
    model_fn,
):
    loss, grads = jax.value_and_grad(loss_fn)(params, graph, target, model_fn)
    updates, opt_state = optimizer_update(grads, opt_state, params)
    updates = optax.tree_utils.tree_scalar_mul(
        scheduler_scale, updates
    )
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


@partial(jax.jit, static_argnames=["model_fn"])
def evaluate(params, graph, target, std, model_fn):
    pred = model_fn(params, graph)
    mask = jraph.get_graph_padding_mask(graph)
    diff = (pred - target) * mask * std
    return jnp.sum(jnp.abs(diff)) / jnp.sum(mask)


def train(args, model, train_loader, val_loader, test_loader, std, max_nodes, max_edges):
    rng = jax.random.PRNGKey(42)

    graph_transform = lambda graph: pad_graph_to_max_size(qmx_batch_to_jraph(graph), max_nodes, max_edges)

    graph_init = graph_transform(next(iter(train_loader)).to("cuda"))

    params = model.init(rng, graph_init)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    min_lr = args.min_lr
    scheduler = optax.contrib.reduce_on_plateau(
        patience=5,
        factor=0.7,
        rtol=0.0001,
        min_scale=min_lr / args.lr
    )
    scheduler_state = scheduler.init(params)

    params = jax.block_until_ready(params)
    print(f"Model compiled", flush=True)

    print(f"Training model with {args.epochs} epochs", flush=True)

    model_fn = model.apply
    optimizer_update = optimizer.update
    best_val_error = best_test_error = float("inf")
    best_params = None
    pbar = tqdm(range(1, args.epochs + 1), total=args.epochs, desc=f"Epoch")
    for epoch in pbar:

        # Training
        epoch_loss = 0
        for graph in train_loader:
            graph = graph.to("cuda")
            graph = graph_transform(graph)
            params, opt_state, loss = update(
                params,
                opt_state,
                graph,
                graph.globals["y"],
                scheduler_state.scale,
                optimizer_update,
                model_fn
            )
            epoch_loss += loss
            print(f"Loss: {loss:.7f}", flush=True)

        epoch_loss /= len(train_loader)

        # Validation
        val_error = 0
        for graph in val_loader:
            graph = graph.to("cuda")
            graph = graph_transform(graph)
            val_error += evaluate(params, graph, graph.globals["y"], std, model_fn)

        val_error /= len(val_loader)

        if val_error < best_val_error:
            best_val_error = val_error
            best_params = params

            # Test error
            test_error = 0
            for graph in test_loader:
                graph = graph.to("cuda")
                graph = graph_transform(graph)
                test_error += evaluate(best_params, graph, graph.globals["y"], std, model_fn)
            test_error /= len(test_loader)
            if test_error < best_test_error:
                best_test_error = test_error

        # Adjust learning rate
        _, scheduler_state = scheduler.update(updates=params, state=scheduler_state, value=val_error)

        peak_memory = utils.get_gpu_memory_usage()

        pbar.set_postfix({
            "Epoch": epoch,
            "LR": scheduler_state.scale * args.lr,
            "Loss": epoch_loss,
            "Val MAE": val_error,
            "Best Test MAE": best_test_error,
            "Peak Memory": peak_memory,
        })

    return best_params, best_test_error


if __name__ == "__main__":
    args = parse_args()
    print(args)

    train_loader, val_loader, test_loader, std, max_nodes, max_edges = load_and_preprocess_data(args)

    # Initialize model
    model = create_model(args.model, args.dim, args.heads, args.concat)

    # Train the model
    best_params, best_test_error = train(args, model, train_loader, val_loader, test_loader, std, max_nodes, max_edges)

    print(f"Best Test MAE: {best_test_error:.7f}")

    # Save the best model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    model_filename = os.path.join(args.save_dir, f"qm9_jax_best_model_{args.target}.pkl")
    with open(model_filename, "wb") as f:
        import pickle
        pickle.dump(best_params, f)
    print(f"Best model saved to {model_filename} with Test MAE: {best_test_error:.7f}")
