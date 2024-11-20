import functools
from typing import List, Optional, Callable, Any, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import jax.tree_util as tree
import haiku as hk
import jraph
# import e3nn_jax as e3nn


from .model_utils import (
    shifted_softplus,
    cosine_cutoff,
)  # , normalized_bessel, u


# Adapted from https://github.com/gerkone/egnn-jax/blob/main/egnn_jax/egnn.py
class EGNNLayer(hk.Module):
    """EGNN layer.

    Args:
        layer_num: layer number
        hidden_size: hidden size
        output_size: output size
        blocks: number of blocks in the node and edge MLPs
        act_fn: activation function
        pos_aggregate_fn: position aggregation function
        msg_aggregate_fn: message aggregation function
        residual: whether to use residual connections
        attention: whether to use attention
        normalize: whether to normalize the coordinates
        tanh: whether to use tanh in the position update
        dt: position update step size
        eps: small number to avoid division by zero
    """

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        output_size: int,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        pos_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        msg_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
    ):
        super().__init__(f"layer_{layer_num}")

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # message network
        self._edge_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [hidden_size],
            w_init=self.init_fn,
            activation=act_fn,
            activate_final=True,
        )

        # update network
        self._node_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [output_size],
            w_init=self.init_fn,
            activation=act_fn,
            activate_final=False,
        )

        # position update network
        net = [hk.Linear(hidden_size, w_init=self.init_fn), act_fn]

        # NOTE: from https://github.com/vgsatorras/egnn/blob/main/models/gcl.py#L254
        net += [
            hk.Linear(
                1, with_bias=False, w_init=hk.initializers.UniformScaling(dt)
            )
        ]
        if tanh:
            net.append(jax.nn.tanh)
        self._pos_correction_mlp = hk.Sequential(net)

        # attention
        self._attention_mlp = None
        if attention:
            self._attention_mlp = hk.Sequential(
                [hk.Linear(hidden_size, w_init=self.init_fn), jax.nn.sigmoid]
            )

        self.pos_aggregate_fn = pos_aggregate_fn
        self.msg_aggregate_fn = msg_aggregate_fn
        self._residual = residual
        self._normalize = normalize
        self._eps = eps

    def _pos_update(
        self,
        graph: jraph.GraphsTuple,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = coord_diff * self._pos_correction_mlp(graph.edges["edge_attr"])
        # NOTE: was in the original code
        trans = jnp.clip(trans, -100, 100)
        return self.pos_aggregate_fn(
            trans, graph.senders, num_segments=graph.nodes["pos"].shape[0]
        )

    def _message(
        self,
        radial: jnp.ndarray,
        edge_attribute: jnp.ndarray,
        edge_features: Any,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = edge_features
        _ = globals_
        msg = jnp.concatenate([incoming, outgoing, radial], axis=-1)
        if edge_attribute is not None:
            msg = jnp.concatenate([msg, edge_attribute], axis=-1)
        msg = self._edge_mlp(msg)
        if self._attention_mlp:
            att = self._attention_mlp(msg)
            msg = msg * att
        return msg

    def _update(
        self,
        node_attribute: jnp.ndarray,
        nodes: jnp.ndarray,
        senders: Any,
        msg: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = senders
        _ = globals_
        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = self._node_mlp(x)
        if self._residual:
            x = nodes + x
        return x

    def _coord2radial(
        self, graph: jraph.GraphsTuple
    ) -> Tuple[jnp.array, jnp.array]:
        nodes, _, senders, receivers, _, _, _ = graph
        coord = nodes["pos"]

        coord_diff = coord[senders] - coord[receivers]
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self._normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self._eps)
        return radial, coord_diff

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """
        Apply EGNN layer.

        Args:
            graph: Graph from previous step
            pos: Node position, updated separately
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)
        """

        radial, coord_diff = self._coord2radial(graph)

        graph = GraphNetwork(
            update_edge_fn=Partial(self._message, radial, edge_attribute),
            update_node_fn=Partial(self._update, node_attribute),
            aggregate_edges_for_nodes_fn=self.msg_aggregate_fn,
        )(graph)

        graph.nodes["pos"] = graph.nodes["pos"] + self._pos_update(
            graph, coord_diff
        )

        return graph


class GraphNetwork(hk.Module):
    def __init__(
        self,
        update_edge_fn: Optional[Callable],
        update_node_fn: Optional[Callable],
        update_global_fn: Optional[Callable] = None,
        aggregate_edges_for_nodes_fn: Callable = jraph.segment_sum,
        aggregate_nodes_for_globals_fn: Callable = jraph.segment_sum,
        aggregate_edges_for_globals_fn: Callable = jraph.segment_sum,
        attention_logit_fn: Optional[Callable] = None,
        attention_normalize_fn: Optional[Callable] = jraph.segment_softmax,
        attention_reduce_fn: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.update_global_fn = update_global_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.attention_logit_fn = attention_logit_fn
        self.attention_normalize_fn = attention_normalize_fn
        self.attention_reduce_fn = attention_reduce_fn

        if (attention_logit_fn is None) != (attention_reduce_fn is None):
            raise ValueError(
                "attention_logit_fn and attention_reduce_fn must both be supplied."
            )

    def __call__(self, graph):
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_edge = senders.shape[0]

        if not tree.tree_all(
            tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)
        ):
            raise ValueError(
                "All node arrays in nest must contain the same number of nodes."
            )

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        global_edge_attributes = tree.tree_map(
            lambda g: jnp.repeat(
                g, n_edge, axis=0, total_repeat_length=sum_n_edge
            ),
            globals_,
        )

        if self.update_edge_fn:
            edges = self.update_edge_fn(
                edges,
                sent_attributes,
                received_attributes,
                global_edge_attributes,
            )

        if self.attention_logit_fn:
            logits = self.attention_logit_fn(
                edges,
                sent_attributes,
                received_attributes,
                global_edge_attributes,
            )
            tree_calculate_weights = functools.partial(
                self.attention_normalize_fn,
                segment_ids=receivers,
                num_segments=sum_n_node,
            )
            weights = tree.tree_map(tree_calculate_weights, logits)
            edges = self.attention_reduce_fn(edges, weights)

        if self.update_node_fn:
            sent_attributes = tree.tree_map(
                lambda e: self.aggregate_edges_for_nodes_fn(
                    e, senders, sum_n_node
                ),
                edges,
            )
            received_attributes = tree.tree_map(
                lambda e: self.aggregate_edges_for_nodes_fn(
                    e, receivers, sum_n_node
                ),
                edges,
            )
            global_attributes = tree.tree_map(
                lambda g: jnp.repeat(
                    g, n_node, axis=0, total_repeat_length=sum_n_node
                ),
                globals_,
            )
            nodes = self.update_node_fn(
                nodes, sent_attributes, received_attributes, global_attributes
            )

        if self.update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node
            )
            edge_gr_idx = jnp.repeat(
                graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge
            )
            node_attributes = tree.tree_map(
                lambda n: self.aggregate_nodes_for_globals_fn(
                    n, node_gr_idx, n_graph
                ),
                nodes,
            )
            edge_attributes = tree.tree_map(
                lambda e: self.aggregate_edges_for_globals_fn(
                    e, edge_gr_idx, n_graph
                ),
                edges,
            )
            globals_ = self.update_global_fn(
                node_attributes, edge_attributes, globals_
            )

        return graph._replace(nodes=nodes, edges=edges, globals=globals_)

### SchNet Layers from https://github.com/fabiannagel/schnax ###


class FilterNetwork(hk.Module):
    def __init__(self, n_filters: int):
        super().__init__(name="FilterNetwork")

        self.linear_0 = hk.Sequential(
            [hk.Linear(n_filters, name="linear_0"), shifted_softplus]
        )  # n_spatial_basis -> n_filters
        self.linear_1 = hk.Linear(
            n_filters, name="linear_1"
        )  # n_filters -> n_filters

    def __call__(self, x: jnp.ndarray):
        x = self.linear_0(x)
        x = self.linear_1(x)
        return x


class CFConv(hk.Module):

    def __init__(
        self,
        n_filters: int,
        n_out: int,
        r_cutoff: float,
        activation,

    ):
        super().__init__(name="CFConv")

        self.filter_network = FilterNetwork(n_filters)
        self.cutoff_network = lambda dR: cosine_cutoff(dR, r_cutoff)

        self.in2f = hk.Linear(n_filters, with_bias=False, name="in2f")
        self.f2out = hk.Sequential(
            [hk.Linear(n_out, with_bias=True, name="f2out"), activation]
        )

    def __call__(
        self,
        x: jnp.ndarray,
        dR: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        dR_expanded: jnp.ndarray,
    ):
        sum_n_node = x.shape[0]

        # pass expanded interactomic distances through filter block
        if dR_expanded.ndim == 3:
            W = self.filter_network(dR_expanded).squeeze(1)
        else:
            W = self.filter_network(dR_expanded)

        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(dR)
            W = W * C #jnp.expand_dims(C, axis=-1)

        # pass initial embeddings through dense layer. reshape y for element-wise multiplication by W.
        y = self.in2f(x)
        y = y[receivers]

        # element-wise multiplication, aggregation and dense output layer.
        y = y * W

        # aggregate over neighborhoods, skip padded indices.
        y = jraph.segment_sum(y, segment_ids=receivers, num_segments=sum_n_node)

        y = self.f2out(y)

        return y


class Interaction(hk.Module):
    def __init__(
        self,
        idx: int,
        n_atom_basis: int,
        n_filters: int,
        r_cutoff: float,
    ):
        super().__init__(name="Interaction_{}".format(idx))

        self.cfconv = CFConv(
            n_filters,
            n_atom_basis,
            r_cutoff,
            activation=shifted_softplus,
        )
        self.dense = hk.Linear(n_atom_basis, name="Output")

    def __call__(
        self,
        x: jnp.ndarray,
        dR: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        dR_expanded=None,
    ):
        """Compute convolution block.

        Args:
            x: input representation/embedding of atomic environments with (N_a, n_in) shape.
            dR: interatomic distances of (N_a, N_nbh) shape.
            neighbors: neighbor list with neighbor indices in (N_a, N_nbh) shape.
            pairwise_mask: mask to filter out non-existing neighbors introduced via padding.
            dR_expanded (optional): expanded interatomic distances in a basis.
                If None, dR.unsqueeze(-1) is used.

        Returns:
            jnp.ndarray: block output with (N_a, n_out) shape.

        """
        x = self.cfconv(x, dR, senders, receivers, dR_expanded)
        x = self.dense(x)
        return x


class GaussianSmearing(hk.Module):
    def __init__(
        self,
        start=0.0,
        stop=5.0,
        n_gaussians=50,
        centered=False,
        trainable=False,
    ):
        super().__init__(name="GaussianSmearing")
        self.offset = jnp.linspace(start, stop, n_gaussians)
        self.widths = (self.offset[1] - self.offset[0]) * jnp.ones_like(
            self.offset
        )
        self.centered = centered

    def _smearing(self, distances: jnp.ndarray) -> jnp.ndarray:
        """Smear interatomic distance values using Gaussian functions."""

        if not self.centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / jnp.power(self.widths, 2)
            # Use advanced indexing to compute the individual components
            # diff = distances[:, :, :, None] - self.offset[None, None, None, :]
            diff = (
                distances[:, :, None] - self.offset[None, None, :]
            )  # skip batches for now
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / jnp.power(self.offset, 2)
            # if Gaussian functions are centered, no offset is subtracted
            # diff = distances[:, :, :, None]
            diff = distances[:, :, None]  # skip batches for now

        # compute smear distance values
        gauss = jnp.exp(coeff * jnp.power(diff, 2))
        return gauss

    def __call__(self, distances: jnp.ndarray, *args, **kwargs):
        smearing = self._smearing(distances)
        return smearing


### Allegro Layers ###


# def filter_layers(
#     layer_irreps: List[e3nn.Irreps], max_ell: int
# ) -> List[e3nn.Irreps]:
#     layer_irreps = list(layer_irreps)
#     filtered = [e3nn.Irreps(layer_irreps[-1])]
#     for irreps in reversed(layer_irreps[:-1]):
#         irreps = e3nn.Irreps(irreps)
#         irreps = irreps.filter(
#             keep=e3nn.tensor_product(
#                 filtered[0],
#                 e3nn.Irreps.spherical_harmonics(lmax=max_ell),
#             ).regroup()
#         )
#         filtered.insert(0, irreps)
#     return filtered


# def allegro_layer_call(
#     Linear,
#     MultiLayerPerceptron,
#     output_irreps: e3nn.Irreps,
#     self,
#     vectors: e3nn.IrrepsArray,  # [n_edges, 3]
#     x: jnp.ndarray,  # [n_edges, features]
#     V: e3nn.IrrepsArray,  # [n_edges, irreps]
#     senders: jnp.ndarray,  # [n_edges]
# ) -> e3nn.IrrepsArray:
#     num_edges = vectors.shape[0]
#     assert vectors.shape == (num_edges, 3)
#     assert x.shape == (num_edges, x.shape[-1])
#     assert V.shape == (num_edges, V.irreps.dim)
#     assert senders.shape == (num_edges,)

#     irreps_out = e3nn.Irreps(output_irreps)

#     w = MultiLayerPerceptron((V.irreps.mul_gcd,), act=None)(x)
#     Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)
#     wY = e3nn.scatter_sum(
#         w[:, :, None] * Y[:, None, :], dst=senders, map_back=True
#     ) / jnp.sqrt(self.avg_num_neighbors)
#     assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

#     V = e3nn.tensor_product(
#         wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out
#     ).axis_to_mul()

#     if "0e" in V.irreps:
#         x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
#         V = V.filter(drop="0e")

#     x = MultiLayerPerceptron(
#         (self.mlp_n_hidden,) * self.mlp_n_layers,
#         self.mlp_activation,
#         output_activation=False,
#     )(x)
#     lengths = e3nn.norm(vectors).array
#     x = u(lengths, self.p) * x
#     assert x.shape == (num_edges, self.mlp_n_hidden)

#     V = Linear(irreps_out)(V)
#     assert V.shape == (num_edges, V.irreps.dim)

#     return (x, V)


# def allegro_call(
#     Linear,
#     MultiLayerPerceptron,
#     self,
#     node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
#     vectors: e3nn.IrrepsArray,  # [n_edges, 3]
#     senders: jnp.ndarray,  # [n_edges]
#     receivers: jnp.ndarray,  # [n_edges]
#     edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
# ) -> e3nn.IrrepsArray:
#     num_edges = vectors.shape[0]
#     num_nodes = node_attrs.shape[0]
#     assert vectors.shape == (num_edges, 3)
#     assert node_attrs.shape == (num_nodes, node_attrs.shape[-1])
#     assert senders.shape == (num_edges,)
#     assert receivers.shape == (num_edges,)

#     assert vectors.irreps in ["1o", "1e"]
#     irreps = e3nn.Irreps(self.irreps)
#     irreps_out = e3nn.Irreps(self.output_irreps)

#     irreps_layers = [irreps] * self.num_layers + [irreps_out]
#     irreps_layers = filter_layers(irreps_layers, self.max_ell)

#     vectors = vectors / self.radial_cutoff

#     d = e3nn.norm(vectors).array.squeeze(1)
#     x = jnp.concatenate(
#         [
#             normalized_bessel(d, self.n_radial_basis),
#             node_attrs[senders],
#             node_attrs[receivers],
#         ],
#         axis=1,
#     )
#     assert x.shape == (
#         num_edges,
#         self.n_radial_basis + 2 * node_attrs.shape[-1],
#     )

#     # Protection against exploding dummy edges:
#     x = jnp.where(d[:, None] == 0.0, 0.0, x)

#     x = MultiLayerPerceptron(
#         (
#             self.mlp_n_hidden // 8,
#             self.mlp_n_hidden // 4,
#             self.mlp_n_hidden // 2,
#             self.mlp_n_hidden,
#         ),
#         self.mlp_activation,
#         output_activation=False,
#     )(x)
#     x = u(d, self.p)[:, None] * x
#     assert x.shape == (num_edges, self.mlp_n_hidden)

#     irreps_Y = irreps_layers[0].filter(
#         keep=lambda mir: vectors.irreps[0].ir.p ** mir.ir.l == mir.ir.p
#     )
#     V = e3nn.spherical_harmonics(irreps_Y, vectors, True)

#     if edge_feats is not None:
#         V = e3nn.concatenate([V, edge_feats])
#     w = MultiLayerPerceptron((V.irreps.num_irreps,), act=None)(x)
#     V = w * V
#     assert V.shape == (num_edges, V.irreps.dim)

#     for irreps in irreps_layers[1:]:
#         y, V = allegro_layer_call(
#             Linear,
#             MultiLayerPerceptron,
#             irreps,
#             self,
#             vectors,
#             x,
#             V,
#             senders,
#         )

#         alpha = 0.5
#         x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

#     x = MultiLayerPerceptron((128,), act=None)(x)

#     xV = Linear(irreps_out)(e3nn.concatenate([x, V]))

#     if xV.irreps != irreps_out:
#         raise ValueError(f"output_irreps {irreps_out} is not reachable")

#     return xV
