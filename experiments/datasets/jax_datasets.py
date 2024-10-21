from torch_geometric.data import Dataset
import jax
import jraph
import jax.numpy as jnp
import numpy as np

# Inspired by https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/train.py
class PyG2JraphDataset:
    def __init__(
        self, pyg_dataset, batch_size: int = 1, dynamically_batch: bool = False, shuffle: bool = False
    ):
        self.pyg_dataset = pyg_dataset
        self.batch_size = batch_size
        self.dynamically_batch = dynamically_batch
        self.shuffle = shuffle
        self.max_num_nodes = max(data.num_nodes for data in pyg_dataset)
        self.max_num_edges = max(data.num_edges for data in pyg_dataset)
        self._indices = np.arange(len(pyg_dataset))
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._current_index = 0

        if dynamically_batch:
            self.generator = jraph.dynamically_batch(
                self._get_generator(),
                n_node=self.batch_size * (self.max_num_nodes),
                n_edge=self.batch_size * (self.max_num_edges),
                n_graph=self.batch_size,
            )
        else:
            self.generator = self._get_generator()

    def __len__(self):
        return len(self.pyg_dataset)

    def __iter__(self):
        self._current_index = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self):
        if self._current_index >= len(self._indices):
            raise StopIteration

        if self.dynamically_batch:
            batch = next(self.generator)
            self._current_index += self.batch_size
        else:
            batch = []
            for _ in range(self.batch_size):
                if self._current_index >= len(self._indices):
                    break
                idx = self._indices[self._current_index]
                batch.append(self.get_graph_at_idx(idx))
                self._current_index += 1
            batch = jraph.batch(batch)

        return self._pad_graph_to_nearest_power_of_two(batch)

    def get_graph_at_idx(self, idx):
        pyg_data = self.pyg_dataset[idx]
        return jraph.GraphsTuple(
            nodes=jnp.array(pyg_data.x),
            edges=jnp.array(pyg_data.edge_attr),
            senders=jnp.array(pyg_data.edge_index[0]),
            receivers=jnp.array(pyg_data.edge_index[1]),
            n_node=jnp.array([pyg_data.num_nodes]),
            n_edge=jnp.array([pyg_data.num_edges]),
            globals={"y": jnp.array(pyg_data.y)},
        )

    def _get_generator(self):
        idx = 0
        while True:
            if idx == self.__len__():
                return
            graph = self.get_graph_at_idx(idx)
            idx += 1
            yield graph

    @staticmethod
    def _nearest_bigger_power_of_two(x: int) -> int:
        """Computes the nearest power of two greater than x for padding."""
        y = 2
        while y < x:
            y *= 2
        return y

    @staticmethod
    def _pad_graph_to_nearest_power_of_two(
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

        Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

        Returns:
        A graphs_tuple batched to the nearest power of two.
        """
        # Add 1 since we need at least one padding node for pad_with_graphs.
        pad_nodes_to = (
            PyG2JraphDataset._nearest_bigger_power_of_two(
                jnp.sum(graphs_tuple.n_node)
            )
            + 1
        )
        pad_edges_to = PyG2JraphDataset._nearest_bigger_power_of_two(
            jnp.sum(graphs_tuple.n_edge)
        )
        # Add 1 since we need at least one padding graph for pad_with_graphs.
        # We do not pad to nearest power of two because the batch size is fixed.
        pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
        return jraph.pad_with_graphs(
            graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
        )


class QMXJraphDataset(PyG2JraphDataset):
    def __init__(self, pyg_dataset, batch_size: int = 1, dynamically_batch: bool = False, shuffle: bool = False):
        super().__init__(pyg_dataset, batch_size, dynamically_batch, shuffle)

    def get_graph_at_idx(self, idx):
        pyg_data = self.pyg_dataset[idx]
        node_features = {
            "x": jnp.array(pyg_data.x),
            "z": jnp.array(pyg_data.z),
            "pos": jnp.array(pyg_data.pos),
        }
        edge_features = {
            "edge_attr": jnp.array(pyg_data.edge_attr),
        }
        if hasattr(pyg_data, "edge_attr2"):
            edge_features["edge_attr2"] = jnp.array(pyg_data.edge_attr2)
        
        global_features = {
            "y": jnp.array(pyg_data.y),
        }
        if hasattr(pyg_data, "name"):
            global_features["name"] = np.array([pyg_data.name])
        if hasattr(pyg_data, "smiles"):
            global_features["smiles"] = np.array([pyg_data.smiles])

        return jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=jnp.array(pyg_data.edge_index[0]),
            receivers=jnp.array(pyg_data.edge_index[1]),
            n_node=jnp.array([pyg_data.num_nodes]),
            n_edge=jnp.array([pyg_data.num_edges]),
            globals=global_features,
        )
