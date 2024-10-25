import functools
from typing import Optional, Callable, Any, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import jax.tree_util as tree
import haiku as hk
import jraph


class GATLayer(hk.Module):
    """Implements GATv2Layer"""
    def __init__(
        self,
        dim,
        num_heads,
        concat=True,
        share_weights=False,
        name=None,
    ):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.concat = concat
        self.share_weights = share_weights
        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # node update function
        self.node_update_fn = lambda x: x 

        # query functions
        self.attention_query_l = lambda x: hk.Linear(
            dim * num_heads, w_init=self.init_fn, name="attention_query_l"
        )(x)

        self.attention_query_r = (
            self.attention_query_l
            if self.share_weights
            else lambda x: hk.Linear(
                dim * num_heads, w_init=self.init_fn, name="attention_query_r"
            )(x)
        )

        self.attention_logit_fn = lambda q, k: hk.Linear(
            1, w_init=self.init_fn, name="attention_logit_fn"
        )(
            jax.nn.leaky_relu(q + k, negative_slope=0.2)
        )

    def __call__(self, graph):
        nodes, _, receivers, senders, _, _, _ = graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

        # Linear transformation 
        nodes_transformed_l = self.attention_query_l(nodes).reshape(
            -1, self.num_heads, self.dim
        )
        if not self.share_weights:
            nodes_transformed_r = self.attention_query_r(nodes).reshape(
                -1, self.num_heads, self.dim
            )
        else:
            nodes_transformed_r = nodes_transformed_l

        # Compute attention logits
        sent_attributes = nodes_transformed_l[senders]
        received_attributes = nodes_transformed_r[receivers]
        attention_logits = self.attention_logit_fn(
            sent_attributes, received_attributes
        )

        # Apply softmax to get attention coefficients
        alpha = jraph.segment_softmax(
            attention_logits, segment_ids=receivers, num_segments=sum_n_node
        )

        # Apply attention coefficients
        out = sent_attributes * alpha

        # Aggregate messages
        out = jraph.segment_sum(
            out, segment_ids=receivers, num_segments=sum_n_node
        )

        # # Concatenate or average the multi-head results
        if self.concat:
            out = out.reshape(sum_n_node, self.dim * self.num_heads)
        else:
            out = jnp.mean(out, axis=1)

        # Apply final update function
        nodes = self.node_update_fn(out)

        return graph._replace(nodes=nodes)


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
