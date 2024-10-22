import os
import argparse
import sys
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import GAT_Layer
from data_loaders import QMXJraphLoader


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
    parser.add_argument(
        "--pad_to_power_of_two", type=bool, default=True, help="Pad to power of two"
    )
    return parser.parse_args()


def create_model(model_name, num_features, dim, heads, concat):
    if model_name == "GAT":

        def net_fn(graph):
            nodes, edges, receivers, senders, _, _, _ = graph

            # Input linear layer
            x = hk.Linear(dim)(nodes)
            x = jax.nn.relu(x)

            # GAT layers with skip connections
            gat1 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(
                    x, negative_slope=0.2
                ),
            )
            gat2 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(
                    x, negative_slope=0.2
                ),
            )
            gat3 = GAT_Layer(
                attention_query_fn=lambda x: hk.Linear(dim * heads)(x),
                attention_logit_fn=lambda s, r, e: hk.Linear(1)(
                    jnp.concatenate([s, r], axis=-1)
                ),
                node_update_fn=lambda x: jax.nn.leaky_relu(
                    x, negative_slope=0.2
                ),
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
                    jax.nn.leaky_relu(negative_slope=0.2),
                    hk.Linear(dim),
                    jax.nn.leaky_relu(negative_slope=0.2),
                    hk.Linear(1),
                ]
            )

            return mlp(out)

        return hk.transform(net_fn)
    else:
        raise ValueError(f"Model {model_name} not found")


@jax.jit
def loss_fn(params, model, graph, target):
    pred = model.apply(params, graph)
    return jnp.mean((pred - target) ** 2)


@jax.jit
def update(params, opt_state, model, graph, target):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, graph, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def evaluate(params, model, graph, target, std):
    pred = model.apply(params, graph)
    return jnp.mean(jnp.abs(pred * std - target * std))


if __name__ == "__main__":
    args = parse_args()
    print(args)

    print(f"Loading data from {args.data_dir}", flush=True)
    pyg_dataset = QM9(
        root=args.data_dir,
        transform=T.Compose(
            [
                lambda data: select_target(data, args.target),
                lambda data: make_complete(data),
                T.Distance(norm=False),
            ]
        ),
    )
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
    train_dataset, test_dataset = train_test_split(
        pyg_dataset, test_size=0.2, random_state=42
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2, random_state=42
    )

    train_loader = QMXJraphLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pad_to_power_of_two=args.pad_to_power_of_two
    )
    val_loader = QMXJraphLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pad_to_power_of_two=args.pad_to_power_of_two
    )
    test_loader = QMXJraphLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pad_to_power_of_two=args.pad_to_power_of_two
    )

    print(f"Train size: {len(train_loader)}", flush=True)
    print(f"Validation size: {len(val_loader)}", flush=True)
    print(f"Test size: {len(test_loader)}", flush=True)

    graph_init = train_loader.get_graph_at_idx(0)

    # Initialize model, optimizer, and loss function
    model = create_model(
        args.model,
        NUM_FEATURES,
        args.dim,
        args.heads,
        args.concat,
    )

    rng = jax.random.PRNGKey(42)
    params = model.init(rng, next(iter(train_dataset)))

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    best_val_error = float("inf")
    best_params = None

    for epoch in tqdm(range(1, 301), desc="Epoch"):
        # Training
        epoch_loss = 0
        for graph in train_dataset:
            params, opt_state, loss = update(
                params, opt_state, model, graph, graph.targets
            )
            epoch_loss += loss
        epoch_loss /= len(train_dataset)

        # Validation
        val_error = 0
        for graph in val_dataset:
            val_error += evaluate(params, model, graph, graph.targets, std)
        val_error /= len(val_dataset)

        if val_error < best_val_error:
            best_val_error = val_error
            best_params = params

        print(
            f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Val MAE: {val_error:.4f}"
        )

    # Test
    test_error = 0
    for graph in test_dataset:
        test_error += evaluate(best_params, model, graph, graph.targets, std)
    test_error /= len(test_dataset)

    print(f"Test MAE: {test_error:.4f}")

    # Save the best model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    model_filename = os.path.join(
        args.save_dir, f"qm9_jax_best_model_{args.target}.pkl"
    )
    with open(model_filename, "wb") as f:
        import pickle

        pickle.dump(best_params, f)
    print(
        f"Best model saved to {model_filename} with Test MAE: {test_error:.7f}"
    )
