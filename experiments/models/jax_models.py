from typing import Callable, Optional, Tuple, List
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import jax.tree_util as tree
# import e3nn_jax as e3nn


from .jax_layers import (
    EGNNLayer, Interaction, GaussianSmearing, #allegro_call
)
from .model_utils import shifted_softplus


# Adapted from https://github.com/gerkone/egnn-jax/blob/main/egnn_jax/egnn.py
class EGNN(hk.Module):
    r"""
    E(n) Graph Neural Network (https://arxiv.org/abs/2102.09844).

    Original implementation: https://github.com/vgsatorras/egnn
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        act_fn: Callable = jax.nn.silu,
        num_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        r"""
        Initialize the network.

        Args:
            hidden_size: Number of hidden features
            output_size: Number of features for 'h' at the output
            act_fn: Non-linearity
            num_layers: Number of layer for the EGNN
            residual: Use residual connections, we recommend not changing this one
            attention: Whether using attention or not
            normalize: Normalizes the coordinates messages such that:
                x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})\|x_i - x_j\|
                It may help in the stability or generalization. Not used in the paper.
            tanh: Sets a tanh activation function at the output of \phi_x(m_{ij}). It
                bounds the output of \phi_x(m_{ij}) which definitely improves in
                stability but it may decrease in accuracy. Not used in the paper.
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = output_size
        self._act_fn = act_fn
        self._num_layers = num_layers
        self._residual = residual
        self._attention = attention
        self._normalize = normalize
        self._tanh = tanh

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
        self.input_layer = hk.Linear(self._hidden_size, w_init=self.init_fn, name="embedding")
        self.readout = hk.Linear(self._output_size, w_init=self.init_fn, name="readout")

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply EGNN.

        Args:
            graph: Input graph
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)

        Returns:
            Tuple of updated node features and positions
        """
        nodes, _, _, _, _, _, _ = graph

        # input node embedding
        nodes["x"] = self.input_layer(nodes["x"])

        graph = graph._replace(nodes=nodes)

        # message passing
        for n in range(self._num_layers):
            graph = EGNNLayer(
                layer_num=n,
                hidden_size=self._hidden_size,
                output_size=self._hidden_size,
                act_fn=self._act_fn,
                residual=self._residual,
                attention=self._attention,
                normalize=self._normalize,
                tanh=self._tanh,
            )(
                graph,
                edge_attribute=edge_attribute,
                node_attribute=node_attribute,
            )
        # node readout
        h = self.readout(graph.nodes["x"])
        return h, graph.nodes["pos"]


class SchNet(hk.Module):

    def __init__(
        self,
        n_atom_basis: int,
        max_z: int,
        n_interactions: int,
        n_gaussians: int,
        n_filters: int,
        r_cutoff: float = 6.0,
        mean: float = 0.0,
        stddev: float = 1.0,
        per_atom: bool = False,
        aggr_type: str = "sum",
    ):

        self.n_atom_basis = n_atom_basis
        self.max_z = max_z
        self.n_gaussians = n_gaussians
        self.n_filters = n_filters
        self.mean = mean
        self.stddev = stddev
        self.aggr_type = aggr_type
        super().__init__(name="SchNet")
        self.n_interactions = n_interactions
        self.per_atom = per_atom

        self.embedding = hk.Embed(
            self.max_z, self.n_atom_basis, name="embeddings"
        )
        self.distance_expansion = GaussianSmearing(
            0.0, r_cutoff, self.n_gaussians
        )

        self.interactions = hk.Sequential(
            [
                Interaction(
                    idx=i,
                    n_atom_basis=self.n_atom_basis,
                    n_filters=self.n_filters,
                    r_cutoff=r_cutoff,
                )
                for i in range(self.n_interactions)
            ]
        )

        self.atomwise = hk.nets.MLP(
            output_sizes=[32, 1], activation=shifted_softplus, name="atomwise"
        )

    @staticmethod
    def standardize(yi: jnp.ndarray, mean: float, stddev: float):
        return yi * stddev + mean

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        # Extract node and edge features
        atoms = graph.nodes["z"]
        senders = graph.senders
        receivers = graph.receivers
        num_graphs = graph.n_node.shape[0]
        dR = graph.edges["edge_attr"]
        sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
        node_mask = jraph.get_node_padding_mask(graph)
        edge_mask = jraph.get_edge_padding_mask(graph)

        # Get embedding for atomic numbers
        x = self.embedding(atoms)

        dR_expanded = self.distance_expansion(dR)

        # Compute interactions
        for interaction in self.interactions.layers:
            v = interaction(x, dR, senders, receivers, edge_mask, dR_expanded)
            x = x + v

        # Compute energy contributions
        yi = self.atomwise(x)
        # yi = self.standardize(yi, self.mean, self.stddev)

        # mask padded nodes
        yi = jnp.squeeze(yi) * node_mask

        if self.per_atom:
            return yi

        # global pooling
        graph_idx = jnp.repeat(
            jnp.arange(num_graphs), graph.n_node, total_repeat_length=sum_n_node
        )
        if self.aggr_type == "sum":
            return jraph.segment_sum(yi, segment_ids=graph_idx, num_segments=num_graphs)
        elif self.aggr_type == "mean":
            return jraph.segment_mean(yi, segment_ids=graph_idx, num_segments=num_graphs)


### Allegro ###


# class AllegroHaiku(hk.Module):

#     def __init__(
#         self,
#         avg_num_neighbors: float,
#         max_ell: int = 3,
#         irreps: e3nn.Irreps = 128
#         * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
#         mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
#         mlp_n_hidden: int = 1024,
#         mlp_n_layers: int = 3,
#         p: int = 6,
#         n_radial_basis: int = 8,
#         radial_cutoff: float = 1.0,
#         output_irreps: e3nn.Irreps = e3nn.Irreps("0e"),
#         num_layers: int = 3,
#     ):
#         super().__init__()
#         self.avg_num_neighbors = avg_num_neighbors
#         self.max_ell = max_ell
#         self.irreps = irreps
#         self.mlp_activation = mlp_activation
#         self.mlp_n_hidden = mlp_n_hidden
#         self.mlp_n_layers = mlp_n_layers
#         self.p = p
#         self.n_radial_basis = n_radial_basis
#         self.radial_cutoff = radial_cutoff
#         self.output_irreps = output_irreps
#         self.num_layers = num_layers

#     def __call__(
#         self,
#         node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
#         vectors: e3nn.IrrepsArray,  # [n_edges, 3]
#         senders: jnp.ndarray,  # [n_edges]
#         receivers: jnp.ndarray,  # [n_edges]
#         edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
#     ) -> e3nn.IrrepsArray:
#         return allegro_call(
#             e3nn.haiku.Linear,
#             e3nn.haiku.MultiLayerPerceptron,
#             self,
#             node_attrs,
#             vectors,
#             senders,
#             receivers,
#             edge_feats,
#         )
