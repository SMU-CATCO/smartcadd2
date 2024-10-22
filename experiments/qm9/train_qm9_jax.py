import os
import argparse
import sys
import jax
import jax.profiler
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T

TESTING = True

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import GAT_Layer
from data_loaders import JraphLoader, qmx_batch_to_jraph
import utils

# At the beginning of your script
# jax.config.update('jax_enable_x64', False)

# print LD_LIBRARY_PATH
print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

# Configure JAX to use GPU
jax.config.update("jax_platform_name", "gpu")
print(f"JAX is using: {jax.devices()}", flush=True)

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
        "--epochs", type=int, default=300, help="Number of epochs"
    )
    return parser.parse_args()

def create_model(model_name, num_features, dim, heads, concat):
    if model_name == "GAT":

        def net_fn(graph):
            nodes, edges, receivers, senders, _, _, _ = graph
            nodes = nodes["x"]

            # Input linear layer
            x = hk.Linear(dim)(nodes)
            x = jax.nn.relu(x)

            # GAT layers with skip connections
            gat1 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            )
            gat2 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            )
            gat3 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            )

            skip1 = hk.Linear(dim * heads if concat else dim)

            # First GAT layer with skip connection
            out1 = gat1(graph._replace(nodes=x))
            out1 = jax.nn.leaky_relu(out1.nodes, negative_slope=0.2)
            out1 = out1 + skip1(x)

            # Second GAT layer with skip connection
            out2 = gat2(graph._replace(nodes=out1))
            out2 = jax.nn.leaky_relu(out2.nodes, negative_slope=0.2)
            out2 = out2 + out1

            # Third GAT layer with skip connection
            out3 = gat3(graph._replace(nodes=out2))
            out3 = jax.nn.leaky_relu(out3.nodes, negative_slope=0.2)
            out3 = out3 + out2

            # Global mean pooling
            out = jnp.mean(out3, axis=0)

            # Final MLP
            mlp = hk.Sequential(
                [
                    hk.Linear(dim),
                    lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
                    hk.Linear(dim),
                    lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
                    hk.Linear(1),
                ]
            )

            return mlp(out)

        return hk.without_apply_rng(hk.transform(net_fn))
    else:
        raise ValueError(f"Model {model_name} not found")


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
    ).to("cuda")

    print(f"Dataset size: {len(pyg_dataset)}", flush=True)

    NUM_FEATURES = pyg_dataset.num_features

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
    train_indices, test_indices = train_test_split(
        np.arange(len(pyg_dataset)), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, random_state=42
    )

    train_loader = JraphLoader(
        pyg_dataset[train_indices],
        batch_size=args.batch_size,
        shuffle=True,
        pad_to_power_of_two=args.pad_to_power_of_two,
        pyg_to_jraph_fn=qmx_batch_to_jraph,
    )
    val_loader = JraphLoader(
        pyg_dataset[val_indices],
        batch_size=args.batch_size,
        shuffle=False,
        pad_to_power_of_two=args.pad_to_power_of_two,
        pyg_to_jraph_fn=qmx_batch_to_jraph,
    )
    test_loader = JraphLoader(
        pyg_dataset[test_indices],
        batch_size=args.batch_size,
        shuffle=False,
        pad_to_power_of_two=args.pad_to_power_of_two,
        pyg_to_jraph_fn=qmx_batch_to_jraph,
    )

    print(f"Train size: {len(train_loader) * args.batch_size}", flush=True)
    print(f"Validation size: {len(val_loader) * args.batch_size}", flush=True)
    print(f"Test size: {len(test_loader) * args.batch_size}", flush=True)

    return train_loader, val_loader, test_loader, NUM_FEATURES, jnp.float32(std)


def train(args, model, train_loader, val_loader, test_loader, std):
    graph_init = next(iter(train_loader))
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, graph_init)

    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),  # Add gradient clipping
        optax.adam(learning_rate=args.lr)
    )
    opt_state = optimizer.init(params)

    scheduler = optax.contrib.reduce_on_plateau(
        patience=5,
        factor=0.7,
        rtol=0.001,
    )
    scheduler_state = scheduler.init(params)

    params = jax.block_until_ready(params)
    print(f"Model compiled", flush=True)

    @jax.jit
    def loss_fn(params, graph, target):
        pred = model.apply(params, graph)
        return jnp.mean((pred - target) ** 2)

    @jax.jit
    def update(params, opt_state, scheduler_state, graph, target):
        loss, grads = jax.value_and_grad(loss_fn)(params, graph, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        updates = optax.tree_utils.tree_scalar_mul(scheduler_state.scale, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def evaluate(params, graph, target, std):
        pred = model.apply(params, graph)
        return jnp.mean(jnp.abs(pred * std - target * std))

    print(f"Training model with {args.epochs} epochs", flush=True)
    print(f"JAX is using: {jax.devices()}", flush=True)

    best_val_error = float("inf")
    best_params = None
    pbar = tqdm(range(1, args.epochs + 1), desc=f"Epoch")
    for epoch in pbar:
        print(f"Epoch {epoch}", flush=True)

        # Training
        epoch_loss = 0
        for graph in train_loader:
            params, opt_state, loss = update(params, opt_state, scheduler_state, graph, graph.globals["y"])
            epoch_loss += loss

        epoch_loss /= len(train_loader)

        # Validation
        val_error = 0
        for graph in val_loader:
            val_error += evaluate(params, graph, graph.globals["y"], std)

        val_error /= len(val_loader)

        if val_error < best_val_error:
            best_val_error = val_error
            best_params = params

        # Adjust learning rate
        _, scheduler_state = scheduler.update(updates=params, state=scheduler_state, value=val_error)

        # Test
        test_error = 0
        for graph in test_loader:
            test_error += evaluate(best_params, graph, graph.globals["y"], std)

        test_error /= len(test_loader)

        peak_memory = utils.get_gpu_memory_usage()
        print(f"Peak Memory: {peak_memory}", flush=True)

        pbar.set_postfix({
            "Epoch": epoch,
            "LR": scheduler_state.scale * args.lr,
            "Loss": loss,
            "Val MAE": val_error,
            "Test MAE": test_error,
            "Peak Memory": peak_memory,
        })

    # Final evaluation on test set with best params
    best_test_error = 0
    for graph in test_loader:
        best_test_error += evaluate(best_params, graph, graph.globals["y"], std)

    best_test_error /= len(test_loader)

    return best_params, best_test_error


if __name__ == "__main__":
    args = parse_args()
    print(args)

    train_loader, val_loader, test_loader, num_features, std = load_and_preprocess_data(args)

    # Initialize model
    model = create_model(args.model, num_features, args.dim, args.heads, args.concat)

    # Train the model
    best_params, best_test_error = train(args, model, train_loader, val_loader, test_loader, std)

    print(f"Best Test MAE: {best_test_error:.7f}")

    # Save the best model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    model_filename = os.path.join(args.save_dir, f"qm9_jax_best_model_{args.target}.pkl")
    with open(model_filename, "wb") as f:
        import pickle
        pickle.dump(best_params, f)
    print(f"Best model saved to {model_filename} with Test MAE: {best_test_error:.7f}")
