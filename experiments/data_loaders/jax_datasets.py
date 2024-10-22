from typing import Callable, Iterable, Any
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

import jax
import jraph
import jax.numpy as jnp

from torch_jax_interop import torch_to_jax as torch_to_jax_src


def torch_to_jax(x):
    try:
        return torch_to_jax_src(x)
    except:
        return torch_to_jax_src(x.contiguous())


def pyg_to_jraph(pyg_data):

    if isinstance(pyg_data, Batch):
        n_node = torch_to_jax(pyg_data.ptr[1:] - pyg_data.ptr[:-1])
        n_edge = torch_to_jax(pyg_data.batch[pyg_data.edge_index[0]].unique(return_counts=True)[1])
    elif isinstance(pyg_data, Data):
        n_node = jnp.array([pyg_data.num_nodes])
        n_edge = jnp.array([pyg_data.num_edges])
    
    return jraph.GraphsTuple(
        nodes=torch_to_jax(pyg_data.x),
        edges=torch_to_jax(pyg_data.edge_attr),
        senders=torch_to_jax(pyg_data.edge_index[0]),
        receivers=torch_to_jax(pyg_data.edge_index[1]),
        n_node=n_node,
        n_edge=n_edge,
        globals={"y": torch_to_jax(pyg_data.y)},
    )


def qmx_batch_to_jraph(pyg_batch):
    node_features = {
        "x": torch_to_jax(pyg_batch.x),
        "z": torch_to_jax(pyg_batch.z),
        "pos": torch_to_jax(pyg_batch.pos),
    }
    edge_features = {
        "edge_attr": torch_to_jax(pyg_batch.edge_attr),
    }
    if hasattr(pyg_batch, "edge_attr2"):
        edge_features["edge_attr2"] = torch_to_jax(pyg_batch.edge_attr2)

    global_features = {
        "y": torch_to_jax(pyg_batch.y),
    }
    if hasattr(pyg_batch, "idx"):
        global_features["idx"] = torch_to_jax(pyg_batch.idx)

    if isinstance(pyg_batch, Batch):
        n_node = torch_to_jax(pyg_batch.ptr[1:] - pyg_batch.ptr[:-1])
        n_edge = torch_to_jax(pyg_batch.batch[pyg_batch.edge_index[0]].unique(return_counts=True)[1])
    elif isinstance(pyg_batch, Data):
        n_node = jnp.array([pyg_batch.num_nodes])
        n_edge = jnp.array([pyg_batch.num_edges])

    return jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=torch_to_jax(pyg_batch.edge_index[0]),
        receivers=torch_to_jax(pyg_batch.edge_index[1]),
        n_node=n_node,
        n_edge=n_edge,
        globals=global_features,
    )


def nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y

def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
    (Note: this is done because pad_with_graphs throws an error if the graph is already the size of pad_nodes_to)

    Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
    A graphs_tuple batched to the nearest power of two.
    """
    pad_nodes_to = nearest_bigger_power_of_two(
        jnp.sum(graphs_tuple.n_node) + 1
    )
    pad_edges_to = nearest_bigger_power_of_two(
        jnp.sum(graphs_tuple.n_edge)
    )
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


class JraphLoader:
    def __init__(
        self,
        pyg_dataset: Dataset,
        pyg_to_jraph_fn: Callable[[Any], jraph.GraphsTuple] = pyg_to_jraph,
        batch_size: int = 1,
        shuffle: bool = False,
        pad_to_power_of_two: bool = False,
    ):
        self.pyg_loader = DataLoader(pyg_dataset, batch_size=batch_size, shuffle=shuffle)
        self.pyg_to_jraph_fn = pyg_to_jraph_fn
        self.pad_to_power_of_two = pad_to_power_of_two
        

    def __len__(self):
        return len(self.pyg_loader)

    def __iter__(self):
        self._iter_loader = iter(self.pyg_loader)
        return self
    
    def __next__(self):
        data = self.pyg_to_jraph_fn(next(self._iter_loader))
        if self.pad_to_power_of_two:
            data = pad_graph_to_nearest_power_of_two(data)
        return data

